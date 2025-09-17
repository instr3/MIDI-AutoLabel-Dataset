import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.roformer_inject import RoFormerEncoder as RoFormerEncoderInject
from cp_transformer import RoFormerSymbolicTransformer, FramedDataset, fill_with_neg_inf
import pytorch_lightning as L
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from peft import LoraConfig, get_peft_model
import sys
from generator_helper import end_generator
from cp_transformer_fine_tune import RoFormerSymbolicTransformerInjected, train
from yield_tags import Tags

TRAIN_LENGTH = 384
MAX_STEPS = 400000


class BaseProber(L.LightningModule):

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        gen = self.base_model(x, use_causal_mask=self.use_causal_mask)
        data = next(gen); assert data[0] == Tags.SIMUNOTE_EMBEDDING
        data = next(gen); assert data[0] == Tags.PE_POSITIONS
        hidden_states = []
        for layer in range(self.n_layers):
            data = next(gen); assert data[0] == Tags.HIDDEN_STATES
            hidden_states.append(data[1])
            data = next(gen); assert data[0] == Tags.PRENORM_OUTPUT
        hidden_states = torch.stack(hidden_states, dim=1)  # [batch_size, n_layers, seq_len, hidden_size]
        weights = F.softmax(self.weights, dim=0)
        hidden_states = torch.sum(weights.view(1, -1, 1, 1) * hidden_states, dim=1)
        return self.linear(hidden_states)

    def preprocess_label_midi(self, x, pitch_shift):
        labels = x[:, :, 1]
        is_valid = labels < 255
        quality = labels // 12
        root = (labels + pitch_shift.unsqueeze(-1) + 12) % 12
        result = quality * 12 + root
        result[~is_valid] = -1  # invalid
        return result

    def preprocess(self, x, pitch_shift, preprocess_args=None):
        batch_size, seq_len, subseq_len = x.shape
        x1_processed = self.base_model.preprocess(x[:, :, :-4], pitch_shift)
        y = self.preprocess_label_midi(x[:, :, -4:], pitch_shift)
        return x1_processed, y

    def loss(self, x, pitch_shift):
        x, y = self.preprocess(x, pitch_shift)
        y_hat = self(x)
        return F.cross_entropy(y_hat.view(-1, y_hat.shape[-1]), y.view(-1), ignore_index=-1)

    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        print('Training with learning rate:', self.lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        # Get the full state dict from the parent class
        full_state_dict = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)

        # Filter to keep only trainable parameters
        trainable_state_dict = {
            k: v for k, v in full_state_dict.items()
            if self._is_trainable_param(k)
        }
        return trainable_state_dict

    def _is_trainable_param(self, param_name):
        # Check if the parameter is trainable
        for name, param in self.named_parameters():
            if name == param_name and param.requires_grad:
                return True
        return False

    def inference(self, x):
        return F.softmax(self(x), dim=-1)



class RoformerProber(BaseProber):

    def __init__(self, model_fp, hidden_size=512, output_size=12, dropout=0.1, lora_r=32, lora_alpha=64, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        if not os.path.exists(model_fp):
            # Use relative path
            model_fp = os.path.join('ckpt', os.path.basename(model_fp))
        base_model = RoFormerSymbolicTransformerInjected.load_from_checkpoint(model_fp, strict=False)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=0.1,
        )
        self.base_model = get_peft_model(base_model, lora_config)
        self.n_layers = base_model.num_layers
        self.weights = nn.Parameter(torch.randn(self.n_layers))  # weights of layers
        self.linear = nn.Sequential(
            nn.Linear(base_model.hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.lr = lr
        self.use_causal_mask = False

class RoformerProberPlain(BaseProber):

    def __init__(self, model_fp, hidden_size=512, output_size=12, dropout=0.1, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        if not os.path.exists(model_fp):
            # Use relative path
            model_fp = os.path.join('ckpt', os.path.basename(model_fp))
        base_model = RoFormerSymbolicTransformerInjected.load_from_checkpoint(model_fp, strict=False)
        base_model.freeze()
        base_model.eval()
        self.base_model = base_model
        self.n_layers = base_model.num_layers
        self.weights = nn.Parameter(torch.randn(self.n_layers))  # weights of layers
        self.linear = nn.Sequential(
            nn.Linear(base_model.hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.lr = lr
        self.use_causal_mask = True

def main():
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int)
    args.add_argument('--fp_path', type=str, default='ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt')
    args.add_argument('--dataset_name', type=str)
    args.add_argument('--weights_path', type=str, default=None)
    args.add_argument('--early_stopping_patience', type=int, default=MAX_STEPS)
    args.add_argument('--lora_r', type=int, default=32)
    args.add_argument('--lora_alpha', type=int, default=64)
    args.add_argument('--lr', type=float, default=1e-4)
    args = args.parse_args()
    train_task = args.dataset_name
    n_gpus = max(torch.cuda.device_count(), 1)
    model_name = f'cp_transformer_yinyang_probe_encoder_v0.2_batch_{args.batch_size * n_gpus}_{train_task}'
    if args.weights_path is not None:
        net = RoformerProber.load_from_checkpoint(args.weights_path, strict=False, lr=args.lr)
        print('Loaded from', args.weights_path)
    else:
        net = RoformerProber(args.fp_path, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lr=args.lr)
    train_set_loader = DataLoader(FramedDataset(f'data/{args.dataset_name}.pt', TRAIN_LENGTH, args.batch_size, split='train'), batch_size=None, num_workers=1, persistent_workers=True)
    val_set_loader = DataLoader(FramedDataset(f'data/{args.dataset_name}.pt', TRAIN_LENGTH, args.batch_size, split='val'), batch_size=None, num_workers=1, persistent_workers=True)
    train(net, model_name, args.early_stopping_patience, MAX_STEPS, train_set_loader, val_set_loader)

if __name__ == '__main__':
    main()