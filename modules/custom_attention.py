import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        n_layers,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        normalize_before,
        dropout,
    ) -> None:
        super().__init__()

        self.attention_layers_stack = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    normalize_before=normalize_before,
                )
                for _ in range(n_layers)
            ]
        )


class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout=0,
        normalize_before=True,
    ):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
            normalize_before=normalize_before,
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model,
            d_inner,
            dropout=dropout,
            normalize_before=normalize_before,
        )

    def forward(self, q, non_pad_mask=None, slf_attn_mask=None, softmax=False):
        enc_output, enc_slf_attn = self.slf_attn(
            q, q, q, mask=slf_attn_mask, softmax=softmax
        )
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class MultiHeadAttention(nn.Module):

    def __init__(
        self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True
    ):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(
            temperature=d_k ** 0.5, attn_dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, softmax=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(1), q.size(2), k.size(2), v.size(2)
        n_samples = q.size(0)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # before view: n_samples x b x lq x (n*dv)
        # after view: n_samples x b x lq x n_head x dv
        q = self.w_qs(q).view(n_samples, sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(n_samples, sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(n_samples, sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: n_samples x b x n_head x lq x dv
        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)

        if mask is not None:
            mask = mask.unsqueeze(2)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask, softmax=softmax)

        # Transpose to move the head dimension back: n_samples x b x lq x n_head x dv
        # Combine the last two dimensions to concatenate all the heads together: n_samples x b x lq x (n*dv)
        output = (
            output.transpose(2, 3)
            .contiguous()
            .view(n_samples, sz_b, len_q, -1)
        )
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, softmax=None):
        # q: n_samples(1) x b x n_head x lq x dk
        # k: n_samples x b x n_head x lk x dk
        # v: n_samples x b x n_head x lk x dv
        # mask : n_samples x b x n_head(1) x lk x 1
        attn = torch.matmul(
            q / self.temperature, k.transpose(-2, -1)
        )  # n_samples x b x n_head x lq x lk

        if softmax:
            if mask is not None:
                attn = attn.masked_fill(~mask.transpose(-2, -1), -1e9)

            attn = F.softmax(attn, dim=-1)
            output = torch.matmul(attn, v)
        else:
            assert mask is not None
            max_attn, _ = torch.max(attn, dim=-1, keepdim=True)
            numerator = torch.exp(attn - max_attn) * mask.transpose(
                -2, -1
            )  # n_samples x b x n_head x lq x lk

            denominator = torch.sum(
                numerator, dim=-1, keepdim=True
            )  # n_samples x b x n_head x lq x 1

            denominator_zeros_mask = denominator <= 1e-10
            denominator = denominator.masked_fill(denominator_zeros_mask, 1)

            custom_softmax = numerator / denominator
            output = torch.matmul(
                custom_softmax, v
            )  # n_samples x b x n_head x lq x dv
            # output = (
            #     numerator / denominator
            # )  # n_samples x b x n_head x lq x dv

            output = output.masked_fill(
                denominator_zeros_mask.expand_as(output), 0
            )
        return output, attn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x
