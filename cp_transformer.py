import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roformer.modeling_roformer import RoFormerModel, RoFormerConfig, RoFormerEncoder
import pytorch_lightning as L
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import sys

TRAIN_LENGTH = 384
MAX_STEPS = 2000000

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

class CPTokenizer(object):
    def __init__(self, with_velocity=False, n_frame_id_embeddings=0):
        if with_velocity:
            self.n_normal_tokens = 24 * 128 + 128 * 16 + n_frame_id_embeddings  # 24 durations, 128 pitches, 128 instruments, 16 velocities
        else:
            self.n_normal_tokens = 24 * 128 + 256 + n_frame_id_embeddings  # 24 durations, 128 pitches, 128 instruments (padded to 256)
        self.frame_token_start = self.n_normal_tokens - n_frame_id_embeddings
        self.n_tokens = self.n_normal_tokens + 3
        self.sos_token = self.n_normal_tokens
        self.eos_token = self.n_normal_tokens + 1
        self.pad_token = self.n_normal_tokens + 2
        self.with_velocity = with_velocity

class RoFormerSymbolicTransformer(L.LightningModule):

    def __init__(self, size=1, max_position_embeddings=1536, with_velocity=False, max_lr=None, n_frame_id_embeddings=0):
        super().__init__()
        self.hidden_size = [512, 768, 1024, 1280][size]
        self.num_layers = [6, 12, 24, 32][size]
        self.num_attention_heads = [8, 12, 16, 16][size]
        self.intermediate_size = [1024, 3072, 4096, 5120][size]
        self.local_model_num_layers = 3
        self.local_model_num_attention_heads = 8
        self.local_model_intermediate_size = 768
        main_roformer_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=max_position_embeddings,
            is_decoder=True,
        )
        self.model = self.get_base_model(main_roformer_config)
        local_encoder_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.local_model_num_layers,
            num_attention_heads=self.local_model_num_attention_heads,
            intermediate_size=self.local_model_intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        local_decoder_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.local_model_num_layers,
            num_attention_heads=self.local_model_num_attention_heads,
            intermediate_size=self.local_model_intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            is_decoder=True
        )
        self.tokenizer = CPTokenizer(with_velocity=with_velocity, n_frame_id_embeddings=n_frame_id_embeddings)
        self.local_embedding = nn.Embedding(self.tokenizer.n_tokens, self.hidden_size)
        self.local_encoder = RoFormerEncoder(local_encoder_config)
        self.local_decoder = RoFormerEncoder(local_decoder_config)
        self.final_decoder = nn.Linear(self.hidden_size, self.tokenizer.n_tokens)
        self.global_sos = nn.Parameter(torch.randn(self.hidden_size))
        self._future_mask = torch.empty(0)
        self.with_velocity = with_velocity
        self.max_lr = max_lr

    def get_base_model(self, config):
        return RoFormerEncoder(config)

    def local_encode(self, x):
        batch_size, seq_len, subseq_len = x.shape
        x = x.view(-1, subseq_len)
        # prepend a <sos> token
        x = torch.cat([torch.full((x.shape[0], 1), self.tokenizer.sos_token, dtype=torch.long, device=x.device), x], dim=-1)
        mask = x != self.tokenizer.pad_token
        emb = self.local_embedding(x)
        h = self.local_encoder(emb, encoder_attention_mask=mask)[0]
        # get representation of the first token
        return h[:, 0], emb[:, :-1]

    def local_decode(self, h, emb):
        batch_size, subseq_len, _ = emb.shape
        # Add h as the first token of emb
        h = h.view(batch_size, 1, -1)
        emb = torch.cat([h, emb[:, 1:]], dim=1)
        # Create an autoregressive mask

        h = self.local_decoder(emb, attention_mask=self.buffered_future_mask(emb))[0]
        return self.final_decoder(h)

    def local_sampling(self, h, max_subseq_len=32, temperature=1.0, global_step=None, sampling_func=None):
        batch_size, _ = h.shape
        y = torch.zeros((batch_size, 0), dtype=torch.long, device=h.device)
        emb = h[:, None, :]
        eos_triggered = torch.zeros(batch_size, dtype=torch.bool, device=h.device)
        past_key_values = None
        local_emb = emb
        for i in range(max_subseq_len):
            h, past_key_values = self.local_decoder(local_emb, past_key_values=past_key_values, use_cache=True, return_dict=False)
            # h_out_ref = self.local_decoder(emb, attention_mask=self.buffered_future_mask(emb))[0]
            # assert torch.allclose(h[:, -1:], h_out_ref[:, -1:], rtol=1e-3, atol=1e-5)
            p = self.final_decoder(h[:, -1])
            if sampling_func is not None:
                p = sampling_func(global_step, i, p)
            if temperature == 0:
                p = F.one_hot(p.argmax(dim=-1), self.tokenizer.n_tokens).float()
            else:
                p = F.softmax(p / temperature, dim=-1)
            y_next = torch.multinomial(p, 1)
            y_next[eos_triggered, :] = self.tokenizer.pad_token  # If EOS has been triggered, pad the rest
            eos_triggered = eos_triggered | (y_next.squeeze(1) == self.tokenizer.eos_token)
            y = torch.cat([y, y_next], dim=1)
            if torch.all(eos_triggered):
                # Pad remaining tokens
                y = torch.cat([y, torch.full((batch_size, max_subseq_len - i - 1), self.tokenizer.pad_token, dtype=torch.long, device=h.device)], dim=1)
                break
            local_emb = self.local_embedding(y_next)
            # emb = torch.cat([emb, self.local_embedding(y_next)], dim=1)
        return y

    def global_sampling(self, x, max_seq_len=384, temperature=1.0, sampling_func=None):
        # import time
        # start_time = time.time()
        batch_size, seq_len, subseq_len = x.shape
        h, _ = self.local_encode(x)
        h = h.view(batch_size, seq_len, self.hidden_size)
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h], dim=1)
        y = [x[:, i, :] for i in range(seq_len)]  # y will be returned by a list
        past_key_values = None
        h_next = h
        for i in range(seq_len, max_seq_len):
            if i % 10 == 0:
                print('Sampling', i, '/', max_seq_len)
                # print('Time passed', time.time() - start_time)
            attention_mask = self.buffered_future_mask(h) if past_key_values is None else None
            h_out, past_key_values = self.model(h_next, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True, return_dict=False)
            # h_out_ref = self.model(h, attention_mask=self.buffered_future_mask(h))[0]
            # print('assert')
            # assert torch.allclose(h_out[:, -1:], h_out_ref[:, -1:], rtol=1e-3, atol=1e-5)
            # h_out = h_out_ref
            y_next = self.local_sampling(h_out[:, -1], temperature=temperature, global_step=i, sampling_func=sampling_func)
            y.append(y_next)
            h_next = self.local_encode(y_next.unsqueeze(1))[0].unsqueeze(1)
            # h = torch.cat([h, self.local_encode(y_next.unsqueeze(1))[0].unsqueeze(1)], dim=1)
        return y

    def buffered_future_mask(self, tensor):
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
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def forward(self, x):
        # x: [batch, seq, subseq]
        # Use local encoder to encode subsequences
        batch_size, seq_len, subseq_len = x.shape
        h, emb = self.local_encode(x)
        h = h.view(batch_size, seq_len, -1)
        # Prepend SOS token and remove the last token
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h[:, :-1]], dim=1)
        # Use global transformer to decode
        h = self.model(h, attention_mask=self.buffered_future_mask(h))[0]
        return self.local_decode(h, emb)


    def preprocess(self, x, pitch_shift, tuple_size=4, chroma_wrap=False):
        batch_size, seq_length, subseq_length = x.shape
        x = x.long().view(batch_size, seq_length, subseq_length // tuple_size, tuple_size)
        x_processed = torch.zeros(batch_size, seq_length, subseq_length // tuple_size, 2, dtype=torch.long, device=x.device)
        pad_indices = x[:, :, :, 1] == 255
        eos_indices = x[:, :, :, 0] == 254
        is_not_drum = x[:, :, :, 0] != 127
        if self.with_velocity:
            x_processed[:, :, :, 0] = x[:, :, :, 0] + 128 * (x[:, :, :, 3] // 8)  # there are 128 instruments
            if chroma_wrap:
                octave = x[:, :, :, 1] // 12
                pitch = (x[:, :, :, 1] + pitch_shift[:, None, None] * is_not_drum) % 12
                x_processed[:, :, :, 1] = octave * 12 + pitch + (x[:, :, :, 2] + 16) * 128
            else:
                x_processed[:, :, :, 1] = x[:, :, :, 1] + (x[:, :, :, 2] + 16) * 128 + pitch_shift[:, None, None] * is_not_drum
        else:
            x_processed[:, :, :, 0] = x[:, :, :, 0]
            if chroma_wrap:
                octave = x[:, :, :, 1] // 12
                pitch = (x[:, :, :, 1] + pitch_shift[:, None, None] * is_not_drum) % 12
                x_processed[:, :, :, 1] = octave * 12 + pitch + (x[:, :, :, 2] + 1) * 128
            else:
                x_processed[:, :, :, 1] = x[:, :, :, 1] + (x[:, :, :, 2] + 1) * 128 + pitch_shift[:, None, None] * is_not_drum
        x_processed[pad_indices] = self.tokenizer.pad_token
        x_processed[:, :, :, 0][eos_indices] = self.tokenizer.eos_token
        return x_processed.view(batch_size, seq_length, subseq_length // tuple_size * 2)
    def loss(self, x, pitch_shift):
        x = self.preprocess(x, pitch_shift)
        y = self(x)
        return F.cross_entropy(y.view(-1, self.tokenizer.n_tokens), x.view(-1), ignore_index=self.tokenizer.pad_token)

    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log('train_loss', loss)
        # scheduler step
        scheduler = self.lr_schedulers()
        scheduler.step()
        self.log('training/lr', scheduler.get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        max_lr = self.max_lr
        optimizer = torch.optim.AdamW(self.parameters(), lr=max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=MAX_STEPS, pct_start=0.005)
        return [optimizer], [scheduler]

    def inference_perplexity(self, x):
        x = self.preprocess(x, torch.zeros(x.shape[0], device=x.device, dtype=torch.long))
        y = self(x)
        result = F.cross_entropy(y.view(-1, self.tokenizer.n_tokens), x.view(-1), ignore_index=self.tokenizer.pad_token, reduction='none')
        batch_size = x.shape[0]
        result = result.view(batch_size, -1).sum(dim=1) / torch.count_nonzero(x.view(batch_size, -1) != self.tokenizer.pad_token, dim=1)
        # Calculate mean and variance along the batch dimension
        return result.mean().item(), result.std().item() * (batch_size / (batch_size - 1)) ** 0.5

class FramedDataset(IterableDataset):

    def __init__(self, file_path, target_length, batch_size, split='all', split_ratio=10, sample_step=1, random_order=True,
                 repeat=True):
        self.file_path = file_path
        self.length = torch.load(file_path[:-3] + '.length.pt', weights_only=True)
        self.start = torch.cumsum(self.length, dim=0) - self.length
        # Invalid samples are those whose length is less than min_length
        is_valid = self.length >= target_length
        self.song_indices = torch.arange(len(self.start))
        # Get training or validation split
        if split == 'all':
            self.valid_indices = self.song_indices[is_valid]
        elif split == 'train':
            self.valid_indices = self.song_indices[torch.logical_and(self.song_indices % split_ratio > 1, is_valid)]
        elif split == 'val':
            self.valid_indices = self.song_indices[torch.logical_and(self.song_indices % split_ratio == 1, is_valid)]
        elif split == 'test':
            self.valid_indices = self.song_indices[torch.logical_and(self.song_indices % split_ratio == 0, is_valid)]
        self.split = split
        self.valid_song_count = len(self.valid_indices)
        self.target_length = target_length
        self.batch_size = batch_size
        self.sample_step = sample_step
        self.random_order = random_order
        self.repeat = repeat
        print('Metadata for dataset', file_path, 'split', split, 'loaded. Number of valid songs:', self.valid_song_count, 'first 20:', self.valid_indices[:20])
        self.data = None
        self.pitch_shift_range = None
    def __iter__(self):
        if self.data is None:
            self.data = torch.load(self.file_path, weights_only=True)
            self.pitch_shift_range = torch.load(self.file_path[:-3] + '.pitch_shift_range.pt', weights_only=True).reshape(-1, 2)
            self.pitch_shift_range[self.pitch_shift_range[:, 0] < -5, 0] = -5
            self.pitch_shift_range[self.pitch_shift_range[:, 1] > 6, 1] = 6
            if self.split == 'val' or self.split == 'test':
                self.pitch_shift_range = torch.zeros_like(self.pitch_shift_range)  # No pitch shift for validation
            print('Data for dataset', self.file_path, 'loaded.')
        while True:
            if self.random_order:
                indices = torch.randperm(len(self.valid_indices))
            else:
                indices = torch.arange(len(self.valid_indices))
            for i in range(0, len(self.valid_indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_pitch_shift_range = self.pitch_shift_range[self.valid_indices[batch_indices]]
                raw_ids = self.valid_indices[batch_indices]
                # starts = torch.randint(self.start[raw_ids], self.start[raw_ids] + self.length[raw_ids] - self.target_length)
                starts = torch.floor(torch.rand(len(raw_ids)) * (self.length[raw_ids] - self.target_length) / self.sample_step).long() * self.sample_step + self.start[raw_ids]
                index_matrix = torch.arange(self.target_length).view(1, -1) + starts.view(-1, 1)
                # Shift the pitch in range [min, max], inclusive
                batch_pitch_shift = torch.floor(torch.rand(len(raw_ids)) * (batch_pitch_shift_range[:, 1] - batch_pitch_shift_range[:, 0] + 1)).long() + batch_pitch_shift_range[:, 0]
                yield self.data[index_matrix], batch_pitch_shift
            if not self.repeat:
                break


if __name__ == '__main__':
    batch_size = int(sys.argv[1])
    model_size = int(sys.argv[2])
    with_velocity = False
    if model_size < 0:  # with velocity
        model_size = -model_size - 1
        with_velocity = True
    assert model_size in [0, 1, 2, 3]
    gradient_clip = 1.0 if model_size >= 2 else None
    max_lr = 5e-5 if model_size >= 2 else 1e-4
    n_gpus = max(torch.cuda.device_count(), 1)
    suffix = 'vel' if with_velocity else ''
    model_name = f'cp_transformer_v0.42{suffix}_size{model_size}_batch_{batch_size * n_gpus}_schedule'
    net = RoFormerSymbolicTransformer(size=model_size, max_lr=max_lr, with_velocity=with_velocity)
    train_set_loader = DataLoader(FramedDataset('data/la_cp16_v2.pt', TRAIN_LENGTH, batch_size), batch_size=None, num_workers=1, persistent_workers=True)
    val_set_loader = DataLoader(FramedDataset('data/rwc_cp16_v2.pt', TRAIN_LENGTH, batch_size), batch_size=None, num_workers=1, persistent_workers=True)
    checkpoint_callback = L.callbacks.ModelCheckpoint(monitor='val_loss',
                                                      save_top_k=10,
                                                      save_last=True,
                                                      enable_version_counter=False,
                                                      dirpath=f'ckpt/{model_name}',
                                                      filename=model_name + '.{epoch:02d}.{val_loss:.5f}')

    # load from checkpoint
    checkpoint_path = None
    if len(sys.argv) > 3:
        checkpoint_path = sys.argv[3]
        if not os.path.exists(checkpoint_path):
            checkpoint_path = None
    if n_gpus > 1:
        import pytorch_lightning.strategies as strategies
        import datetime
        strategy = strategies.DDPStrategy(timeout=datetime.timedelta(hours=2))
    else:
        strategy = 'auto'
    trainer = L.Trainer(devices=-1,
                        precision="bf16-mixed" if torch.cuda.is_available() else 32,
                        max_steps=MAX_STEPS,
                        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                        callbacks=[checkpoint_callback],
                        val_check_interval=2500,
                        limit_val_batches=25,
                        check_val_every_n_epoch=None,
                        gradient_clip_val=gradient_clip,
                        logger=TensorBoardLogger("tb_logs", name=model_name),
                        num_sanity_val_steps=0 if checkpoint_path is not None else 2,
                        strategy=strategy)
    trainer.fit(net, train_set_loader, val_set_loader, ckpt_path=checkpoint_path)
    # save the model (parameters only)
    torch.save(net.state_dict(), f'ckpt/{model_name}.pt')