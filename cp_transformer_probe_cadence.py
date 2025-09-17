import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.roformer_inject import RoFormerEncoder as RoFormerEncoderInject
from cp_transformer import RoFormerSymbolicTransformer, FramedDataset, fill_with_neg_inf
import pytorch_lightning as L
from torch.utils.data import DataLoader, IterableDataset
from yield_tags import Tags
from cp_transformer_fine_tune import train
from cp_transformer_probe import RoformerProber

TRAIN_LENGTH = 384
MAX_STEPS = 400000

class RoformerProberCadence(RoformerProber):

    def __init__(self, model_fp, hidden_size=512, lora_r=32, lora_alpha=64, dropout=0.1, lr=1e-4, use_causal_mask=True):
        super().__init__(model_fp, hidden_size=hidden_size, output_size=24, dropout=dropout, lr=lr, lora_r=lora_r, lora_alpha=lora_alpha)
        self.save_hyperparameters()
        self.n_chroma_groups = 2
        self.use_causal_mask = use_causal_mask

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
        batch_size, seq_len, _ = x.shape # [batch_size, seq_len, 24]
        x = x.view(batch_size, seq_len, self.n_chroma_groups, 12)

        base_idx = torch.arange(12, device=x.device)  # [12]
        base_idx = base_idx.unsqueeze(0).expand(batch_size, 12)  # [batch_size, 12]
        # shifts: [batch_size]
        roll_idx = (base_idx - pitch_shift.unsqueeze(1)) % 12  # [batch_size, 12]
        # x: [batch_size, seq_len, n_chroma_groups, 12]
        # Needs roll_idx to be [batch_size, 1, 1, 12]
        roll_idx = roll_idx[:, None, None, :]  # [batch_size, 1, 1, 12]
        roll_idx = roll_idx.expand(-1, seq_len, self.n_chroma_groups, -1)  # [batch_size, seq_len, n_chroma_groups, 12]
        rolled_x = torch.gather(x, -1, roll_idx)
        return rolled_x.view(batch_size, seq_len, -1)  # [batch_size, seq_len, 24]

    def preprocess(self, x, pitch_shift, preprocess_args=None):
        batch_size, seq_len, subseq_len = x.shape
        x1_processed = self.base_model.preprocess(x[:, :, :-self.n_chroma_groups * 12], pitch_shift)
        y = self.preprocess_label_midi(x[:, :, -self.n_chroma_groups * 12:], pitch_shift)
        return x1_processed, y

    def loss(self, x, pitch_shift):
        x, y = self.preprocess(x, pitch_shift)
        y_hat = self(x)
        y = y.type(y_hat.dtype) / 255.0
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def inference(self, x):
        y = self(x)
        y = F.sigmoid(y)
        return y

def main():
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int)
    args.add_argument('--fp_path', type=str, default='ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt')
    args.add_argument('--dataset_name', type=str)
    args.add_argument('--weights_path', type=str, default=None)
    args.add_argument('--early_stopping_patience', type=int, default=MAX_STEPS)
    args.add_argument('--lr', type=float, default=1e-4)
    args.add_argument('--lora_r', type=int, default=32)
    args.add_argument('--lora_alpha', type=int, default=64)
    args.add_argument('--use_causal_mask', action='store_true', default=True)
    args.add_argument('--no_causal_mask', action='store_false', dest='use_causal_mask')
    args = args.parse_args()
    train_task = args.dataset_name
    n_gpus = max(torch.cuda.device_count(), 1)
    model_type = 'probe' if args.use_causal_mask else 'probe_encoder'
    model_name = f'cp_transformer_yinyang_{model_type}_cadence_v0.2_batch_{args.batch_size * n_gpus}_{train_task}_lora{args.lora_r}-{args.lora_alpha}'
    if args.weights_path is not None:
        net = RoformerProberCadence.load_from_checkpoint(args.weights_path, strict=False, lr=args.lr, use_causal_mask=args.use_causal_mask)
        print('Loaded from', args.weights_path)
    else:
        net = RoformerProberCadence(args.fp_path, lr=args.lr, lora_r=args.lora_r, lora_alpha=args.lora_alpha, use_causal_mask=args.use_causal_mask)
    train_set_loader = DataLoader(FramedDataset(f'data/{args.dataset_name}.pt', TRAIN_LENGTH, args.batch_size, split='train'), batch_size=None, num_workers=1, persistent_workers=True)
    val_set_loader = DataLoader(FramedDataset(f'data/{args.dataset_name}.pt', TRAIN_LENGTH, args.batch_size, split='val'), batch_size=None, num_workers=1, persistent_workers=True)
    train(net, model_name, args.early_stopping_patience, MAX_STEPS, train_set_loader, val_set_loader)

if __name__ == '__main__':
    main()