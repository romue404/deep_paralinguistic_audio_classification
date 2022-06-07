import torch
import torch.nn as nn
import random
from x_transformers.x_transformers import exists, LayerIntermediates, Encoder, ContinuousTransformerWrapper
from modules.patchmerger import FrequencyPatchMerger
from modules.spec_pos_emb import PatchyfiedSpecPosEmbs
from modules.attention import AttnFn
from torch.nn.utils.rnn import pad_sequence
from vector_quantize_pytorch import VectorQuantize, ResidualVQ

class MyEncoder(Encoder):
    def __init__(self, f_patchmerger=nn.Identity(), **kwargs):
        super().__init__(**kwargs)

        for ll in self.layers:
            for l in ll:
                try:
                    l.fn.attn_fn = AttnFn()
                except AttributeError as e:
                    pass

    def forward(
            self,
            x,
            context=None,
            mask=None,
            context_mask=None,
            attn_mask=None,
            mems=None,
            return_hiddens=False
    ):
        assert not (self.cross_attend ^ exists(context)), 'context must be passed in if cross_attend is set to True'

        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            max_rotary_emb_length = max(list(map(lambda m: (m.shape[1] if exists(m) else 0) + x.shape[1], mems)))
            rotary_pos_emb = self.rotary_pos_emb(max_rotary_emb_length, x.device)

        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            is_last = ind == (len(self.layers) - 1)

            if layer_type == 'a':
                hiddens.append(x)
                layer_mem = mems.pop(0) if mems else None

            #if (ind == int(0.5*self.num_attn_layers)):
            #    x, indices, commit_loss = self.vq(x)  # quantize it

            residual = x

            pre_branch_norm, post_branch_norm, post_main_norm = norm

            if exists(pre_branch_norm):
                x = pre_branch_norm(x)

            if layer_type == 'a':
                out, inter = block(x, mask=mask, attn_mask=attn_mask, sinusoidal_emb=self.pia_pos_emb,
                                   rel_pos=self.rel_pos, rotary_pos_emb=rotary_pos_emb, prev_attn=prev_attn,
                                   mem=layer_mem)
            elif layer_type == 'c':
                out, inter = block(x, context=context, mask=mask, context_mask=context_mask, prev_attn=prev_cross_attn)
            elif layer_type == 'f':
                out = block(x)

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            x = residual_fn(out, residual)

            if layer_type in ('a', 'c'):
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hiddens,
                attn_intermediates=intermediates
            )

            return x, intermediates

        return x


class MySpecTf(ContinuousTransformerWrapper):
    def __init__(
        self,
        *args,  num_freq_patches, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dim = dim = self.attn_layers.dim
        self.num_freq_patches = num_freq_patches
        self.mask_token = nn.Parameter(torch.randn(1, dim))
        self.clf_tokens = nn.Parameter(torch.randn(1, dim))
        self.pos_emb = PatchyfiedSpecPosEmbs(num_freq_patches, dim)

        self.vq = VectorQuantize(dim=self.dim, codebook_size=512, codebook_dim=16, decay=0.9, heads=2)
        self.l = nn.Linear(self.dim, self.dim)


    def cat_clf_token(self, x):
        return torch.cat((self.clf_tokens.expand(x.shape[0], -1, -1), x), 1)

    def make_mask(self, lengths, include_clf=True):
        offset = 0 if not include_clf else 1
        max_len = lengths.max().item() + offset
        return (torch.arange(max_len).expand(len(lengths), max_len)
                < (lengths + offset).unsqueeze(1))

    def random_masking(self, lengths, p):
        max_ = max(lengths)
        chunks = [i*max_ for i in range(len(lengths))]
        mask_out_idxs = [random.sample(range(chunks[i], chunks[i]+lengths[i]),
                                       k=int((chunks[i]+lengths[i] - chunks[i]) * p))
                         for i, c in enumerate(chunks)]
        return mask_out_idxs

    def forward(
        self,
        data,
        return_embeddings = False,
        lengths = None,
        return_attn = False,
        mems = None,
        mask_prob=None,
        return_masked_frames=False,
        **kwargs
    ):
        b, n, _, device = *data.shape, data.device

        spec_pos_emb = self.pos_emb(data)
        x = self.project_in(data)



        mask_token = None
        if mask_prob > 0 and self.training:
            mask_idxs = self.random_masking(lengths, mask_prob)
            mask_idxs = torch.tensor(sum(mask_idxs, [])).squeeze().to(lengths.device)
            x.view(-1, x.shape[-1])[mask_idxs] = self.mask_token

        x = x + spec_pos_emb
        x = self.cat_clf_token(x)
        x = self.emb_dropout(x)

        x, intermediates = self.attn_layers(x, mask=self.make_mask(lengths, True).to(device),
                                            mems=mems, return_hiddens=True, **kwargs)
        x = self.norm(x)

        x = self.l(self.vq(x)[0])
        out = self.project_out(x) if not return_embeddings else x

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        if return_masked_frames:
            masked_in = data.view(-1, data.shape[-1])[mask_idxs]
            masked_out = x[:, 1:].reshape(-1, x[:, 1:].shape[-1])[mask_idxs]
            return out, masked_in, masked_out
        return out


class MySpecTfMsmEncoder(MySpecTf):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vq = VectorQuantize(dim=self.dim, codebook_size=256, codebook_dim=32, decay=0.9,
                                 use_cosine_sim=True, heads=4, threshold_ema_dead_code=2)
        self.l = nn.Linear(self.dim, self.dim)


    def forward(
        self,
        data,
        return_embeddings = False,
        lengths = None,
        return_attn = False,
        mems = None,
        mask_prob=.75,
        **kwargs
    ):
        b, n, d, device = *data.shape, data.device

        spec_pos_emb = self.pos_emb(data).expand(b, -1, -1).reshape(-1, self.dim)  # flatten pos embs
        keep_idxs    = self.random_masking(lengths, 1. - mask_prob)  # idxs of kept tokens
        num_kept     = torch.tensor([len(i) for i in keep_idxs])  # new lens
        keep_idxs    = torch.tensor(sum(keep_idxs, [])).squeeze().to(lengths.device)  # flatten idxs of kept tokens
        leftovers    = data.view(-1, d)[keep_idxs]  # flat remaining tokens

        x = self.project_in(leftovers)  # project remaining tokens

        x = x + spec_pos_emb[keep_idxs]  # add pos embs to remaining tokens
        x = self.emb_dropout(x)

        cumsum_lens = [0] + num_kept.cumsum(0).tolist()  # compute ranges for new sequences
        sequences = [x[i:j] for i, j in zip(cumsum_lens, cumsum_lens[1:])]  # list of variable len sequences
        sequences = pad_sequence(sequences, batch_first=True)  # pad sequences

        # pass through transformer
        x, intermediates = self.attn_layers(sequences, mask=self.make_mask(num_kept, False).to(device),
                                            mems=mems, return_hiddens=True, **kwargs)
        x = self.norm(x)
        x = self.l(self.vq(x)[0])
        out = self.project_out(x) if not return_embeddings else x

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        kept_mask = self.make_mask(num_kept, False).flatten().nonzero().squeeze()  # indices of embeddings only
        out = out.view(-1, self.dim)[kept_mask]  # select embeddings
        return out, keep_idxs, num_kept


class MySpecTfMsmDecoder(MySpecTf):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings_flat, keep_idxs, old_lengths, **kwargs):
        n, d, device = *embeddings_flat.shape, embeddings_flat.device
        b, t = len(old_lengths), max(old_lengths)  # use original shape before masking

        x = self.mask_token.expand(b, t, -1)  # batch filled completely with mask token
        spec_pos_emb = self.pos_emb(x).expand(b, -1, -1).reshape(-1, self.pos_emb.dim)  # flatten pos embs
        x = x.reshape(-1, x.shape[-1])   # flatten mask tokens
        # mask_idxs = [i for i in range(0, x.shape[0]) if i not in keep_idxs]

        m1 = torch.ones_like(x)
        m1[keep_idxs] = 0
        x = x*m1  # removes grad of mask, allows us to replace with embeddings
        x[keep_idxs] = self.project_in(embeddings_flat)  # replace mask with embeddings
        x = x + spec_pos_emb  # add pos embs
        x = self.emb_dropout(x)

        x = x.view(b, t, x.shape[-1])  # prepare shape for transformer

        # pass through transformer
        x, intermediates = self.attn_layers(x, mask=self.make_mask(old_lengths, False).to(device),
                                            mems=None, return_hiddens=True, **kwargs)
        x = self.norm(x)
        x = self.project_out(x)  # reconstruct

        return x