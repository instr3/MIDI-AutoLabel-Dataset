import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from cp_transformer import RoFormerSymbolicTransformer, FramedDataset, fill_with_neg_inf
from cp_transformer_fine_tune import RoformerFineTune, PreprocessingParameters, RoFormerSymbolicTransformerInjected, train
import pytorch_lightning as L
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from peft import LoraConfig, get_peft_model
from modules.yinyang_cross_attn import LowRankMultiheadAttention
from modules.hubert import _compute_mask
import sys
from generator_helper import end_generator, injection
from yield_tags import Tags

MAX_STEPS = 60000

class RoformerSymbolicTransformerInjectedZiyu(RoFormerSymbolicTransformerInjected):

    def global_sampling(self, x, max_seq_len=384, temperature=1.0, sampling_func=None):
        batch_size, seq_len, subseq_len = x.shape
        h, _ = self.local_encode(x)
        h = h.view(batch_size, seq_len, h.shape[-1])
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 2, 1)
        h = torch.cat([sos, h], dim=1)
        y = [x[:, i, :] for i in range(seq_len)]  # y will be returned by a list
        past_key_values = None
        for i in range(seq_len, max_seq_len):
            yield [Tags.GENERATION_STEP, i]
            if i % 10 == 0:
                print('Sampling', i, '/', max_seq_len)
            attention_mask = self.buffered_future_mask_extra(h) if past_key_values is None else None
            h_out, past_key_values = yield from self.model(h, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True, return_dict=False)
            y_next = self.local_sampling(h_out[:, -1], temperature=temperature, global_step=i, sampling_func=sampling_func)
            injected_array = [Tags.SAMPLED_TOKEN, y_next]
            yield injected_array
            y_next = injected_array[1]
            y.append(y_next)
            h = self.local_encode(y_next.unsqueeze(1))[0].unsqueeze(1)
        return y

    def forward(self, x, use_causal_mask=True):
        # x: [batch, seq, subseq]
        # Use local encoder to encode subsequences
        batch_size, seq_len, subseq_len = x.shape
        h, emb = self.local_encode(x)
        h = h.view(batch_size, seq_len, -1)
        [_, h] = yield from injection(Tags.SIMUNOTE_EMBEDDING, h)
        # Prepend SOS token and remove the last token
        sos = self.global_sos.view(1, 1, -1).repeat(h.shape[0], 2, 1)
        h = torch.cat([sos, h[:, :-2]], dim=1)
        # Use global transformer to decode
        h = (yield from self.model(h, attention_mask=self.buffered_future_mask_extra(h) if use_causal_mask else None))[0]
        return self.local_decode(h, emb)

    def buffered_future_mask_extra(self, tensor):
        dim = tensor.size(1)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
                self._future_mask.size(0) == 0
                or (not self._future_mask.device == tensor.device)
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
            idx = torch.arange(len(self._future_mask) // 2) * 2
            self._future_mask[idx, idx + 1] = 0.0
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

class RoformerZiyu(RoformerFineTune):

    def __init__(self, model_fp, train_task, mask_prob=None, mask_length=None, max_position_embeddings=768, n_skip=2,
                 compress_ratio_l=1, compress_ratio_r=1, lr=None):
        super().__init__(compress_ratio_l=compress_ratio_l, compress_ratio_r=compress_ratio_r, lr=lr)
        self.save_hyperparameters()
        if not os.path.exists(model_fp):
            # Use relative path
            model_fp = os.path.join('ckpt', os.path.basename(model_fp))
        base_model1 = RoformerSymbolicTransformerInjectedZiyu.load_from_checkpoint(model_fp,
                                                                                   max_position_embeddings=max_position_embeddings,
                                                                                   strict=False)
        base_model2 = RoformerSymbolicTransformerInjectedZiyu.load_from_checkpoint(model_fp,
                                                                                   max_position_embeddings=max_position_embeddings,
                                                                                   strict=False)

        lora_config1 = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1,
        )
        lora_config2 = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1,
        )
        self.wrapped_model = get_peft_model(base_model1, lora_config1)
        self.wrapped_model2 = get_peft_model(base_model2, lora_config2)
        self.n_layers = base_model1.num_layers
        self.masked_embedding = nn.Parameter(torch.randn(1, 1, base_model1.hidden_size) * 0.1, requires_grad=True)
        self.n_skip = n_skip
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.yinyang_mask_ratio = 0.0
        self.preprocess_args = PreprocessingParameters(train_task)

    def get_yinyang_attn(self, layer):
        return self.yinyang_attn[layer]

    def forward(self, x_concat, indices_replace):
        gen1 = self.wrapped_model(x_concat)
        gen2 = self.wrapped_model2(x_concat)
        data1 = next(gen1); assert data1[0] == Tags.SIMUNOTE_EMBEDDING
        data2 = next(gen2); assert data2[0] == Tags.SIMUNOTE_EMBEDDING
        data1 = next(gen1); assert data1[0] == Tags.PE_POSITIONS
        data1[1] = indices_replace
        data2 = next(gen2); assert data2[0] == Tags.PE_POSITIONS
        data2[1] = indices_replace
        for layer in range(self.n_layers):
            data1 = next(gen1); assert data1[0] == Tags.HIDDEN_STATES
            data2 = next(gen2); assert data2[0] == Tags.HIDDEN_STATES
            data1 = next(gen1); assert data1[0] == Tags.PRENORM_OUTPUT
            data2 = next(gen2); assert data2[0] == Tags.PRENORM_OUTPUT
            avg = (data1[1] + data2[1]) / 2
            data1[1] = avg
            data2[1] = avg
        result1 = end_generator(gen1)
        result2 = end_generator(gen2)
        return result1, result2

    def global_sampling(self, x1, x2, temperature=1.0, sampling_func=None, max_seq_length=384, direction=1):
        batch_size, seq_len1, subseq_len = x1.shape
        _, seq_len2, _ = x2.shape
        assert self.compress_ratio_l == 1
        assert self.compress_ratio_r == 1
        min_seq_length = min(seq_len1, seq_len2)
        indices_replace = torch.arange(2 * max_seq_length, dtype=torch.long, device=x1.device) // 2  # [0, 0, 1, 1, 2, 2, ..., max_seq_length-1, max_seq_length-1]
        x1_short = x1[:, :min_seq_length, :]
        x2_short = x2[:, :min_seq_length, :]
        if direction == 1:
            x_concat = torch.stack([x1_short, x2_short], dim=2)  # [batch_size, min_seq_length, 2, subseq_len]
        else:
            x_concat = torch.stack([x2_short, x1_short], dim=2)
        x_concat = x_concat.view(x_concat.shape[0], -1, x_concat.shape[-1])  # [batch_size, min_seq_length * 2, subseq_len]
        gen1 = self.wrapped_model.global_sampling(x_concat, max_seq_len=max_seq_length * 2, temperature=temperature, sampling_func=sampling_func)
        gen2 = self.wrapped_model2.global_sampling(x_concat, max_seq_len=max_seq_length * 2, temperature=temperature, sampling_func=sampling_func)

        for i in range(min_seq_length * 2, max_seq_length * 2):
            data1 = next(gen1); assert data1[0] == Tags.GENERATION_STEP
            data2 = next(gen2); assert data2[0] == Tags.GENERATION_STEP
            data1 = next(gen1); assert data1[0] == Tags.PE_POSITIONS
            data1[1] = indices_replace[i:i + data1[1].shape[0]]
            data2 = next(gen2); assert data2[0] == Tags.PE_POSITIONS
            data2[1] = indices_replace[i:i + data2[1].shape[0]]
            for layer in range(self.n_layers):
                data1 = next(gen1); assert data1[0] == Tags.HIDDEN_STATES
                data2 = next(gen2); assert data2[0] == Tags.HIDDEN_STATES
                data1 = next(gen1); assert data1[0] == Tags.PRENORM_OUTPUT
                data2 = next(gen2); assert data2[0] == Tags.PRENORM_OUTPUT
                avg = (data1[1] + data2[1]) / 2
                data1[1] = avg
                data2[1] = avg
            data1 = next(gen1); assert data1[0] == Tags.SAMPLED_TOKEN
            data2 = next(gen2); assert data2[0] == Tags.SAMPLED_TOKEN
            if (i % 2 == 0) == (direction == 1):
                if seq_len1 > i // 2:
                    # Override from the input sequence
                    data1[1] = x1[:, i // 2, :]
                # Prediction from gen1
                data2[1] = data1[1]
            else:
                if seq_len2 > i // 2:
                    # Override from the input sequence
                    data2[1] = x2[:, i // 2, :]
                data1[1] = data2[1]
        result1 = end_generator(gen1)  # max_seq_length * 2 of [batch_size, subseq_len]
        result2 = end_generator(gen2)  # max_seq_length * 2 of [batch_size, subseq_len]
        result1 = result1[::2]  # [batch_size, max_seq_length, subseq_len]
        result2 = result2[1::2]  # [batch_size, max_seq_length, subseq_len]
        return result1, result2

    def loss(self, x, pitch_shift):
        x1, x2, _, _ = self.preprocess(x, pitch_shift, preprocess_args=self.preprocess_args)
        batch_size, seq_len, subseq_len = x1.shape
        # Generate a random {0, 1}
        direction = torch.randint(0, 2, (1,), device=x1.device).item()  # 0 or 1
        if direction == 1:
            x_concat = torch.stack([x1, x2], dim=2) # [batch_size, seq_len, 2, subseq_len]
        else:
            x_concat = torch.stack([x2, x1], dim=2) # [batch_size, seq_len, 2, subseq_len]
        x_concat = x_concat.view(batch_size, seq_len * 2, subseq_len)  # [batch_size, seq_len * 2, subseq_len]
        indices_replace = torch.arange(x_concat.shape[1], device=x_concat.device) // 2 # [0, 0, 1, 1, 2, 2, ..., seq_len-1, seq_len-1]

        y1, y2 = self(x_concat, indices_replace)
        if direction == 1:
            x1_pred = y1.view(batch_size, seq_len, 2, subseq_len, self.wrapped_model.tokenizer.n_tokens)[:, ::2]
            x2_pred = y2.view(batch_size, seq_len, 2, subseq_len, self.wrapped_model.tokenizer.n_tokens)[:, 1::2]
        else:
            x1_pred = y1.view(batch_size, seq_len, 2, subseq_len, self.wrapped_model.tokenizer.n_tokens)[:, 1::2]
            x2_pred = y2.view(batch_size, seq_len, 2, subseq_len, self.wrapped_model.tokenizer.n_tokens)[:, ::2]
        loss_x1 = F.cross_entropy(x1_pred.reshape(-1, self.wrapped_model.tokenizer.n_tokens), x1.reshape(-1), ignore_index=self.wrapped_model.tokenizer.pad_token)
        loss_x2 = F.cross_entropy(x2_pred.reshape(-1, self.wrapped_model.tokenizer.n_tokens), x2.reshape(-1), ignore_index=self.wrapped_model.tokenizer.pad_token)
        return loss_x1, loss_x2


    def training_step(self, batch, batch_idx):
        loss_x1, loss_x2 = self.loss(*batch)
        loss = loss_x1 + loss_x2
        self.log('train_loss', loss)
        self.log('training/loss_x1', loss_x1)
        self.log('training/loss_x2', loss_x2)
        return loss

    def validation_step(self, batch, batch_idx):
        loss_x1, loss_x2 = self.loss(*batch)
        loss = loss_x1 + loss_x2
        self.log('val_loss', loss, sync_dist=True)
        self.log('validation/loss_x1', loss_x1, sync_dist=True)
        self.log('validation/loss_x2', loss_x2, sync_dist=True)
        return loss

def main():
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int)
    args.add_argument('--fp_path', type=str, default='ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt')
    args.add_argument('--dataset_name', type=str)
    args.add_argument('--train_task', type=str, default=None)
    args.add_argument('--weights_path', type=str, default=None)
    args.add_argument('--mask_prob', type=float, default=0.25)
    args.add_argument('--mask_length', type=int, default=10)
    args.add_argument('--compress_ratio_l', type=int, default=1)
    args.add_argument('--compress_ratio_r', type=int, default=1)
    args.add_argument('--train_length', type=int, default=384)
    args.add_argument('--lr', type=float, default=1e-4)
    args.add_argument('--n_skip', type=int, default=2)
    args.add_argument('--early_stopping_patience', type=int, default=MAX_STEPS)
    args = args.parse_args()
    sample_step = max(args.compress_ratio_l, args.compress_ratio_r)
    train_length = args.train_length
    n_gpus = max(torch.cuda.device_count(), 1)
    skip_tag = f'-skip{args.n_skip}' if args.n_skip != 2 else ''
    train_task = args.dataset_name if args.train_task is None else args.train_task
    model_name = f'cp_transformer_ziyu_v0.45_batch_{args.batch_size * n_gpus}_{train_task}_mask{args.mask_prob}-{args.mask_length}-step{sample_step}{skip_tag}'
    if args.weights_path is not None:
        net = RoformerZiyu.load_from_checkpoint(args.weights_path, strict=False, lr=args.lr)
        print('Loaded from', args.weights_path)
    else:
        net = RoformerZiyu(args.fp_path, train_task=train_task, mask_prob=args.mask_prob,
                              mask_length=args.mask_length,
                              compress_ratio_l=args.compress_ratio_l, compress_ratio_r=args.compress_ratio_r,
                              lr=args.lr, n_skip=args.n_skip)
    train_set_loader = DataLoader(FramedDataset(f'data/{args.dataset_name}.pt', train_length, args.batch_size, split='train', sample_step=sample_step), batch_size=None, num_workers=1, persistent_workers=True)
    val_set_loader = DataLoader(FramedDataset(f'data/{args.dataset_name}.pt', train_length, args.batch_size, split='val', sample_step=sample_step), batch_size=None, num_workers=1, persistent_workers=True)
    train(net, model_name, args.early_stopping_patience, MAX_STEPS, train_set_loader, val_set_loader)


if __name__ == '__main__':
    main()