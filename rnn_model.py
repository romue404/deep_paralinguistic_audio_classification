import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence
import pytorch_lightning as pl
import torchmetrics as metrics
from modules.hybrid_pooling import HybridPooler
from modules.transformer import MySpecTf, MyEncoder, MySpecTfMsmEncoder, MySpecTfMsmDecoder
from modules.patchmerger import FrequencyPatchMerger
from modules.spec_utils import SpecMixup, SpecDrloc
from utils import NormalizedLinear, exclude_from_wt_decay


class MySpecTFMR(pl.LightningModule):
    def __init__(self,
                 num_freq_patches, mode='msm',
                 lr=3e-4, weight_decay=1e-4,
                 num_layers=8, num_heads=4, dropout=0.2,
                 input_dim=128,  hidden_size=256, mask_prob=0.2,
                 label_smoothing=0.1, num_mem_kv=32, f_patchmerger=False,
                 ff_mult=4, num_warmup_epochs=10):
        super(MySpecTFMR, self).__init__()
        self.mode = mode
        self.weight_decay = weight_decay
        self.hidden_size = hidden_size
        self.num_freq_patches = num_freq_patches
        self.label_smoothing = label_smoothing
        self.f_patchmerger = f_patchmerger
        self.input_dim = input_dim

        self.lr = lr
        self.num_warmup_epochs = num_warmup_epochs
        self.num_layers = num_layers
        self.dropout =  dropout
        self.__mask_prob = mask_prob

        fpm = FrequencyPatchMerger(num_freq_patches, hidden_size) if f_patchmerger else nn.Identity()

        Wrapper = MySpecTfMsmEncoder if mode == 'msm' else MySpecTf

        self.transformer = Wrapper(
            dim_in=input_dim,
            dim_out=None,
            max_seq_len=1024,
            emb_dropout=dropout,
            num_freq_patches=num_freq_patches,
            attn_layers=MyEncoder(
                dim_head=128,
                dim=self.hidden_size,
                depth=self.num_layers,
                heads=num_heads,
                attn_dropout=dropout,
                attn_num_mem_kv=num_mem_kv,
                ff_dropout=dropout,
                ff_relu_squared=True,
                use_rezero=True,
                #shift_tokens=1,
                ff_mult=ff_mult,
                f_patchmerger=fpm
            ))

        self.decoder = MySpecTfMsmDecoder(
            dim_in=input_dim,
            dim_out=input_dim,
            max_seq_len=1024,
            emb_dropout=0.0,
            num_freq_patches=num_freq_patches,
            attn_layers=MyEncoder(
                dim=self.hidden_size,
                depth=self.num_layers // 2,
                heads=num_heads,
                attn_dropout=.1,
                ff_dropout=.1,
                ff_glu=True,
                ff_swish=True,
                use_rmsnorm=True,
                ff_mult=ff_mult
            )) if mode == 'msm' else nn.Identity()


    @property
    def mask_prob(self):
        return self.__mask_prob if self.training else 0

    def forward(self, x_pad, lengths):
        embeddings_flat, keep_idxs, num_kept = self.transformer(x_pad, lengths=lengths, mask_prob=self.mask_prob)
        decoded = self.decoder(embeddings_flat, keep_idxs, lengths)
        return decoded, keep_idxs, num_kept

    def var_len_mse(self, predictions, targets, lengths):
        delta = (predictions - targets).pow(2)
        cumsum_lens = [0] + lengths.cumsum(0).tolist()
        losses = torch.stack([delta[i:j].mean() for i, j in zip(cumsum_lens, cumsum_lens[1:])], 0)
        return losses.mean()

    def training_step(self, batch, batch_idx=None):
        data, labels, _ = batch

        x_pad, lengths = pad_packed_sequence(data, batch_first=True)
        decoded, keep_idxs, num_kept = self.forward(x_pad, lengths)

        pad_mask = self.decoder.make_mask(lengths, False).flatten()
        pad_mask[keep_idxs] = False
        target_idxs = pad_mask.nonzero().squeeze().flatten()

        predictions = decoded.view(-1, decoded.shape[-1])[target_idxs]
        target      = x_pad.view(-1, self.input_dim)[target_idxs]

        return self.var_len_mse(predictions, target, lengths - num_kept)

    def configure_optimizers(self):
        def lr_lambda(current_step):
            if current_step < self.num_warmup_epochs:
                return float(current_step) / float(max(1, self.num_warmup_epochs))
            progress = float(current_step - self.num_warmup_epochs) / \
                       float(max(1, self.trainer.max_epochs - self.num_warmup_epochs))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress)))
        params = exclude_from_wt_decay(self.named_parameters(), self.weight_decay,
                                       ['temp', 'temperature', 'sn_alpha', 'scale', 'norm'])
        optimizer = optim.AdamW(params,  weight_decay=self.weight_decay, lr=self.lr)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'frequency': 1, 'interval': 'epoch'}}


class ClassificationTFMR(MySpecTFMR):
    def __init__(self, *args, num_classes, **kwargs):
        super(ClassificationTFMR, self).__init__(*args, mode='clf', **kwargs)
        self.micro_f1  =  metrics.classification.f_beta.F1Score(compute_on_step=False, num_classes=num_classes, average='micro')
        self.macro_f1  =  metrics.classification.f_beta.F1Score(compute_on_step=False, num_classes=num_classes, average='macro')
        self.uar       =  metrics.Recall(compute_on_step=False,  num_classes=num_classes, average='macro')
        self.confusion =  metrics.ConfusionMatrix(num_classes=num_classes, compute_on_step=False)
        self.best_uar = 0, 0
        self.num_classes = num_classes

        self.pooler = HybridPooler(self.hidden_size, self.dropout, self.num_freq_patches if self.f_patchmerger else -1)
        self.clf = NormalizedLinear(2*self.hidden_size, self.num_classes)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)  # PolyLoss(epsilon=-.5)

        self.mixup = SpecMixup(num_classes=self.num_classes, alpha=0.3)
        self.drloc = SpecDrloc(self.num_freq_patches, 100, self.hidden_size, m=16)

    def forward(self, data, labels):
        x_pad, lengths = pad_packed_sequence(data, batch_first=True)
        if self.training:
            x_pad, labels = self.mixup(x_pad, labels)
        tokens = self.transformer(x_pad, lengths=lengths, mask_prob=self.mask_prob)
        return self.pooler(tokens, lengths), labels, tokens, lengths

    def training_step(self, batch, batch_idx=None):
        data, labels, _ = batch
        pooled, labels, tokens, lengths = self.forward(data, labels)
        logits = self.clf(pooled)
        loss = self.criterion(logits, labels)
        ssl_loss = self.drloc(tokens[:, 1:], lengths)
        loss = loss + 1*ssl_loss
        return loss

    def validation_step(self, batch, *args, **kwargs):
        data, labels, _ = batch
        pooled, labels, _, _ = self.forward(data, labels)
        logits = self.clf(pooled)
        predictions = logits.argmax(-1)
        self.micro_f1(predictions, labels)
        self.macro_f1(predictions, labels)
        self.uar(predictions, labels)
        self.confusion(predictions, labels)

    def validation_epoch_end(self, outs):
        mic_f1 = self.micro_f1.compute()
        mac_f1 = self.macro_f1.compute()
        uar    = self.uar.compute()
        self.log('UAR', uar)
        #confusion_matrix = self.confusion.compute()
        print(f'UAR: {uar:.3f}\tMicro F1: {mic_f1:.3f}\tMacro F1: {mac_f1:.3f}')
        self.micro_f1.reset()
        self.macro_f1.reset()
        self.uar.reset()
        self.confusion.reset()
        if self.best_uar[0] < uar:
            self.best_uar = uar, self.current_epoch

    def test_step(self, *args, **kwargs):
        self.validation_step(*args, **kwargs)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)