import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.roformer_inject import RoFormerEncoder as RoFormerEncoderInject
from cp_transformer import RoFormerSymbolicTransformer, FramedDataset, fill_with_neg_inf
import pytorch_lightning as L
from torch.utils.data import DataLoader, IterableDataset
from cp_transformer_fine_tune import train
from cp_transformer_probe import RoformerProber

TRAIN_LENGTH = 384
MAX_STEPS = 400000

class RoformerProberChord(RoformerProber):

    def __init__(self, model_fp, hidden_size=512, dropout=0.1, lora_r=32, lora_alpha=64, chroma_tuple_size=24, lr=1e-4, freeze_all=False):
        super().__init__(model_fp, hidden_size=hidden_size, output_size=chroma_tuple_size * 2, lora_r=lora_r, lora_alpha=lora_alpha, dropout=dropout, lr=lr)
        self.save_hyperparameters()
        self.extra_save_parameters = set()
        self.chroma_tuple_size = chroma_tuple_size
        if freeze_all:
            for name, param in self.named_parameters():
                if param.requires_grad and 'linear.3' not in name:
                    param.requires_grad = False
                    self.extra_save_parameters.add(name)
            self.base_model.eval()


    def preprocess_label_midi(self, x, pitch_shift):
        batch_size, seq_len, _ = x.shape # [batch_size, seq_len, 24]
        x = x.view(batch_size, seq_len, self.chroma_tuple_size // 12, 12)

        base_idx = torch.arange(12, device=x.device)  # [12]
        base_idx = base_idx.unsqueeze(0).expand(batch_size, 12)  # [batch_size, 12]
        # shifts: [batch_size]
        roll_idx = (base_idx - pitch_shift.unsqueeze(1)) % 12  # [batch_size, 12]
        # x: [batch_size, seq_len, self.chroma_tuple_size // 12, 12]
        # Needs roll_idx to be [batch_size, 1, 1, 12]
        roll_idx = roll_idx[:, None, None, :]  # [batch_size, 1, 1, 12]
        roll_idx = roll_idx.expand(-1, seq_len, self.chroma_tuple_size // 12, -1)  # [batch_size, seq_len, self.chroma_tuple_size // 12, 12]
        rolled_x = torch.gather(x, -1, roll_idx)
        return rolled_x.view(batch_size, seq_len, -1)  # [batch_size, seq_len, 24]

    def preprocess(self, x, pitch_shift, preprocess_args=None):
        batch_size, seq_len, subseq_len = x.shape
        x1_processed = self.base_model.preprocess(x[:, :, :-self.chroma_tuple_size], pitch_shift)
        y = self.preprocess_label_midi(x[:, :, -self.chroma_tuple_size:], pitch_shift)
        return x1_processed, y

    def loss(self, x, pitch_shift):
        x, y = self.preprocess(x, pitch_shift)
        y_hat = self(x)
        return F.cross_entropy(y_hat.view(-1, 2), y.view(-1))

    def inference(self, x):
        y = self(x)
        y = y.view(y.shape[0], y.shape[1], y.shape[2] // 2, 2)
        return F.softmax(y, dim=-1)[..., 1]

    def _is_trainable_param(self, param_name):
        if param_name in self.extra_save_parameters:
            return True
        return super()._is_trainable_param(param_name)

def main():
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int)
    args.add_argument('--fp_path', type=str, default='ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt')
    args.add_argument('--dataset_name', type=str)
    args.add_argument('--weights_path', type=str, default=None)
    args.add_argument('--lora_r', type=int, default=32)
    args.add_argument('--lora_alpha', type=int, default=64)
    args.add_argument('--chroma_tuple_size', type=int, default=24)
    args.add_argument('--early_stopping_patience', type=int, default=MAX_STEPS)
    args.add_argument('--train_task', type=str, default=None)
    args.add_argument('--lr', type=float, default=1e-4)
    args.add_argument('--freeze_all', action='store_true', default=False)
    args = args.parse_args()
    train_task = args.dataset_name if args.train_task is None else args.train_task
    n_gpus = max(torch.cuda.device_count(), 1)
    model_name = f'cp_transformer_yinyang_probe_encoder_chroma_v0.5_batch_{args.batch_size * n_gpus}_{train_task}'
    if args.weights_path is not None:
        net = RoformerProberChord.load_from_checkpoint(args.weights_path, strict=False, lr=args.lr, freeze_all=args.freeze_all)
        print('Loaded from', args.weights_path)
    else:
        net = RoformerProberChord(args.fp_path, lr=args.lr, freeze_all=args.freeze_all, lora_r=args.lora_r, lora_alpha=args.lora_alpha, chroma_tuple_size=args.chroma_tuple_size)
    train_set_loader = DataLoader(FramedDataset(f'data/{args.dataset_name}.pt', TRAIN_LENGTH, args.batch_size, split='train'), batch_size=None, num_workers=1, persistent_workers=True)
    val_set_loader = DataLoader(FramedDataset(f'data/{args.dataset_name}.pt', TRAIN_LENGTH, args.batch_size, split='val'), batch_size=None, num_workers=1, persistent_workers=True)
    train(net, model_name, args.early_stopping_patience, MAX_STEPS, train_set_loader, val_set_loader)

if __name__ == '__main__':
    main()