import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.roformer_inject import RoFormerEncoder as RoFormerEncoderInject
from cp_transformer import RoFormerSymbolicTransformer
from generator_helper import end_generator, next_tag, injection
from typing import Dict, Any
from yield_tags import Tags
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import os
import shutil

def decompress(model, byte_arr):
    x = torch.tensor(byte_arr).unsqueeze(0)
    x = x.cuda()
    return model.preprocess(x, pitch_shift=torch.zeros(1, dtype=torch.int8).cuda(), preprocess_args=PreprocessingParameters(''))[:2]


class RoFormerSymbolicTransformerInjected(RoFormerSymbolicTransformer):

    def __init__(self, size=1, max_position_embeddings=512, n_frame_id_embeddings=0):
        super().__init__(size=size, max_position_embeddings=max_position_embeddings, n_frame_id_embeddings=n_frame_id_embeddings)

    def get_base_model(self, config):
        return RoFormerEncoderInject(config)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Remove the embedding positions
        del checkpoint['state_dict']['model.embed_positions.weight']

    def forward(self, x, use_causal_mask=True):
        # x: [batch, seq, subseq]
        # Use local encoder to encode subsequences
        batch_size, seq_len, subseq_len = x.shape
        h, emb = self.local_encode(x)
        h = h.view(batch_size, seq_len, -1)
        [_, h] = yield from injection(Tags.SIMUNOTE_EMBEDDING, h)
        # Prepend SOS token and remove the last token
        sos = self.global_sos.view(1, 1, -1).repeat(h.shape[0], 1, 1)
        h = torch.cat([sos, h[:, :-1]], dim=1)
        # Use global transformer to decode
        h = (yield from self.model(h, attention_mask=self.buffered_future_mask(h) if use_causal_mask else None))[0]
        return self.local_decode(h, emb)


    def global_sampling(self, x, max_seq_len=384, temperature=1.0, sampling_func=None):
        batch_size, seq_len, subseq_len = x.shape
        h, _ = self.local_encode(x)
        h = h.view(batch_size, seq_len, h.shape[-1])
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h], dim=1)
        y = [x[:, i, :] for i in range(seq_len)]  # y will be returned by a list
        past_key_values = None
        for i in range(seq_len, max_seq_len):
            yield [Tags.GENERATION_STEP, i]
            if i % 10 == 0:
                print('Sampling', i, '/', max_seq_len)

            attention_mask = self.buffered_future_mask(h) if past_key_values is None else None
            h_out, past_key_values = yield from self.model(h, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True, return_dict=False)
            y_next = self.local_sampling(h_out[:, -1], temperature=temperature, global_step=i, sampling_func=sampling_func)
            y.append(y_next)
            h = self.local_encode(y_next.unsqueeze(1))[0].unsqueeze(1)
        return y

    def global_model_with_kv_cache(self, seq_len, max_seq_len=384, h=None):
        past_key_values = None
        for i in range(seq_len, max_seq_len):
            [_, _, h] = yield from injection(Tags.GENERATION_STEP, i, h)
            attention_mask = self.buffered_future_mask(h) if past_key_values is None else None
            h_out, past_key_values = yield from self.model(h, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True, return_dict=False)
            yield [Tags.HIDDEN_STATES, h]

    def loss(self, x, pitch_shift):
        x = self.preprocess(x, pitch_shift)
        y = yield from self(x)
        return F.cross_entropy(y.view(-1, self.tokenizer.n_tokens), x.view(-1), ignore_index=self.tokenizer.pad_token)

def shrink_duration(x, value):
    x[x < 255] = torch.max(x[x < 255], torch.tensor(value, dtype=x.dtype, device=x.device)) - value

def clip_sequence(x_processed, pad_token):
    min_values = x_processed.reshape(-1, x_processed.shape[-1]).min(dim=0).values
    return x_processed[:, :, min_values != pad_token]

class PreprocessingParameters:
    def __init__(self, task_name):
        self.task_name = task_name
        task_name_split = task_name.split('_')
        self.is_reverse = 'rev' in task_name_split
        self.is_self = 'self' in task_name_split
        self.left_mask = 'lmask' in task_name_split
        self.label_chroma_shift = 'chordlab' in task_name_split


class RoformerFineTune(L.LightningModule):

    def __init__(self, compress_ratio_l, compress_ratio_r, lr):
        super().__init__()
        self.save_hyperparameters()
        self.wrapped_model = None
        self.compress_ratio_l = compress_ratio_l
        self.compress_ratio_r = compress_ratio_r
        self.lr = lr

    def preprocess(self, x, pitch_shift, preprocess_args: PreprocessingParameters, clip_subseq=False, tuple_size=4):
        batch_size, seq_len, subseq_len = x.shape
        if self.compress_ratio_l == 1 and self.compress_ratio_r == 1:
            if preprocess_args.is_self:
                x_processed = self.wrapped_model.preprocess(x, pitch_shift, tuple_size=tuple_size)
                x_processed1 = x_processed
                # this ensures that the tensors are not shared, but currently it is not necessary
                x_processed2 = x_processed.clone()
            else:
                if preprocess_args.label_chroma_shift:
                    x = x.view(batch_size, seq_len, 2, subseq_len // 2)
                    x1 = x[:, :, 0, :]
                    x2 = x[:, :, 1, :]
                    x_processed1 = self.wrapped_model.preprocess(x1, pitch_shift, tuple_size=tuple_size, chroma_wrap=False)
                    x_processed2 = self.wrapped_model.preprocess(x2, pitch_shift, tuple_size=tuple_size, chroma_wrap=True)
                else:
                    x = x.view(batch_size, seq_len * 2, subseq_len // 2)
                    x_processed = self.wrapped_model.preprocess(x, pitch_shift, tuple_size=tuple_size)
                    x_processed = x_processed.view(batch_size, seq_len, 2, -1)
                    x_processed1 = x_processed[:, :, 0, :]
                    x_processed2 = x_processed[:, :, 1, :]
            if preprocess_args.is_reverse:
                x_processed1, x_processed2 = x_processed2, x_processed1
            indices1 = indices2 = torch.arange(seq_len, dtype=torch.long, device=x.device)
        else:
            x = x.view(batch_size, seq_len, 2, subseq_len // 2)
            if preprocess_args.is_reverse:
                x1 = x[:, ::self.compress_ratio_l, 1, :].reshape(-1, tuple_size)
                x2 = x[:, ::self.compress_ratio_r, 0, :].reshape(-1, tuple_size)
            else:
                x1 = x[:, ::self.compress_ratio_l, 0, :].reshape(-1, tuple_size)
                x2 = x[:, ::self.compress_ratio_r, 1, :].reshape(-1, tuple_size)
            # shrink duration
            shrink_ratio_l = int(np.log2(self.compress_ratio_l)) * 2
            shrink_ratio_r = int(np.log2(self.compress_ratio_r)) * 2
            shrink_duration(x1[:, 2], shrink_ratio_l)
            shrink_duration(x2[:, 2], shrink_ratio_r)
            indices1 = torch.arange(0, seq_len, self.compress_ratio_l, dtype=torch.long, device=x.device)
            indices2 = torch.arange(0, seq_len, self.compress_ratio_r, dtype=torch.long, device=x.device)
            x_processed1 = self.wrapped_model.preprocess(x1.reshape(batch_size, -1, subseq_len // 2), pitch_shift, tuple_size=tuple_size, chroma_wrap=False)
            x_processed2 = self.wrapped_model.preprocess(x2.reshape(batch_size, -1, subseq_len // 2), pitch_shift, tuple_size=tuple_size, chroma_wrap=preprocess_args.label_chroma_shift)
        if clip_subseq:
            raise Exception('Clip subseq is currently not supported')
        # if clip_subseq:
        #     x_processed1 = clip_sequence(x_processed1, self.wrapped_model.tokenizer.pad_token).contiguous()
        #     x_processed2 = clip_sequence(x_processed2, self.wrapped_model.tokenizer.pad_token).contiguous()
        return x_processed1, x_processed2, indices1, indices2

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        # Get the full state dict from the parent class
        full_state_dict = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)

        # Filter to keep only trainable parameters
        trainable_state_dict = {
            k: v for k, v in full_state_dict.items()
            if self._is_trainable_param(k)
        }
        return trainable_state_dict

    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def _is_trainable_param(self, param_name):
        # Check if the parameter is trainable
        for name, param in self.named_parameters():
            if name == param_name and param.requires_grad:
                return True
        return False

def train(net, model_name, early_stopping_patience, max_steps, train_set_loader, val_set_loader, ckpt_path=None):
    n_gpus = max(torch.cuda.device_count(), 1)
    checkpoint_callback = L.callbacks.ModelCheckpoint(monitor='val_loss',
                                                      save_top_k=10,
                                                      save_last=True,
                                                      dirpath=f'ckpt/{model_name}',
                                                      enable_version_counter=False,
                                                      filename=model_name + '.{epoch:02d}.{val_loss:.5f}')
    early_stopping = L.callbacks.EarlyStopping(monitor='val_loss',
                                               mode='min',
                                               patience=early_stopping_patience)
    trainer = L.Trainer(devices=n_gpus,
                        precision="bf16-mixed" if torch.cuda.is_available() else 32,
                        max_steps=max_steps,
                        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                        callbacks=[checkpoint_callback, early_stopping],
                        val_check_interval=500,
                        limit_val_batches=25,
                        check_val_every_n_epoch=None,
                        logger=TensorBoardLogger("tb_logs", name=model_name),
                        strategy='auto' if n_gpus == 1 else 'ddp_find_unused_parameters_true')
    net.strict_loading = False
    trainer.fit(net, train_set_loader, val_set_loader, ckpt_path=ckpt_path)
    shutil.copy(checkpoint_callback.best_model_path, f'ckpt/{model_name}.epoch=best.ckpt')
    os.chmod(f'ckpt/{model_name}.epoch=best.ckpt', 0o666)
    shutil.copy(f'ckpt/{model_name}/last.ckpt', f'ckpt/{model_name}.epoch=last.ckpt')
    os.chmod(f'ckpt/{model_name}.epoch=last.ckpt', 0o666)