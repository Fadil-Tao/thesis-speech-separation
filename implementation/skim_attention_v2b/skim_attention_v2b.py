# SkiM Attention v2b: keeps MemLSTM, runs MemAttention in parallel on segment-level h/c states,
# fuses outputs with a learnable scalar gate. Cross-segment recurrent + attention combined.

import torch
import torch.nn as nn

from espnet2.enh.layers.dprnn import SingleRNN, merge_feature, split_feature
from espnet2.enh.layers.tcn import choose_norm


class MemLSTM(nn.Module):
    """Mem-LSTM of base SkiM, kept intact."""

    def __init__(
        self,
        hidden_size,
        dropout=0.0,
        bidirectional=False,
        mem_type="hc",
        norm_type="cLN",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.input_size = (int(bidirectional) + 1) * hidden_size
        self.mem_type = mem_type

        assert mem_type in ["hc", "h", "c", "id"], (
            f"only support 'hc', 'h', 'c' and 'id', current type: {mem_type}"
        )

        if mem_type in ["hc", "h"]:
            self.h_net = SingleRNN(
                "LSTM",
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.h_norm = choose_norm(
                norm_type=norm_type, channel_size=self.input_size, shape="BTD"
            )
        if mem_type in ["hc", "c"]:
            self.c_net = SingleRNN(
                "LSTM",
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.c_norm = choose_norm(
                norm_type=norm_type, channel_size=self.input_size, shape="BTD"
            )

    def extra_repr(self) -> str:
        return f"Mem_type: {self.mem_type}, bidirectional: {self.bidirectional}"

    def forward(self, hc, S):
        if self.mem_type == "id":
            ret_val = hc
            h, c = hc
            d, BS, H = h.shape
            B = BS // S
        else:
            h, c = hc
            d, BS, H = h.shape
            B = BS // S
            h = h.transpose(1, 0).contiguous().view(B, S, d * H)
            c = c.transpose(1, 0).contiguous().view(B, S, d * H)
            if self.mem_type == "hc":
                h = h + self.h_norm(self.h_net(h)[0])
                c = c + self.c_norm(self.c_net(c)[0])
            elif self.mem_type == "h":
                h = h + self.h_norm(self.h_net(h)[0])
                c = torch.zeros_like(c)
            elif self.mem_type == "c":
                h = torch.zeros_like(h)
                c = c + self.c_norm(self.c_net(c)[0])

            h = h.view(B * S, d, H).transpose(1, 0).contiguous()
            c = c.view(B * S, d, H).transpose(1, 0).contiguous()
            ret_val = (h, c)

        if not self.bidirectional:
            causal_ret_val = []
            for x in ret_val:
                x = x.transpose(1, 0).contiguous().view(B, S, d * H)
                x_ = torch.zeros_like(x)
                x_[:, 1:, :] = x[:, :-1, :]
                x_ = x_.view(B * S, d, H).transpose(1, 0).contiguous()
                causal_ret_val.append(x_)
            ret_val = tuple(causal_ret_val)

        return ret_val


class MemAttention(nn.Module):
    """Multi-Head Self-Attention over segment-level h/c states.

    Same body as v1's MemAttention. Output is shaped to match the input's `d`
    (number of LSTM directions), so it can be summed with MemLSTM's output in
    the FusedMem block under both bidirectional and causal configs.
    """

    def __init__(
        self, hidden_size, num_heads=4, dropout=0.0, mem_type="hc", norm_type="cLN"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mem_type = mem_type
        self.norm_type = norm_type

        assert mem_type in ["hc", "h", "c", "id"], (
            f"only support 'hc', 'h', 'c' and 'id', got: {mem_type}"
        )
        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )

        if mem_type in ["hc", "h"]:
            self.h_attn = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=dropout, batch_first=True
            )
            self.h_ffn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(dropout),
            )
            self.h_norm1 = nn.LayerNorm(hidden_size)
            self.h_norm2 = nn.LayerNorm(hidden_size)

        if mem_type in ["hc", "c"]:
            self.c_attn = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=dropout, batch_first=True
            )
            self.c_ffn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(dropout),
            )
            self.c_norm1 = nn.LayerNorm(hidden_size)
            self.c_norm2 = nn.LayerNorm(hidden_size)

    def extra_repr(self):
        return f"mem_type={self.mem_type}, num_heads={self.num_heads}"

    def forward(self, hc, S, causal=False):
        if self.mem_type == "id":
            return hc

        h, c = hc
        d, BS, H = h.shape
        B = BS // S

        # Take last LSTM layer of input states: (d, B*S, H) -> (B, S, H)
        h = h.transpose(1, 0).contiguous().view(B, S, d, H)[:, :, -1, :]
        c = c.transpose(1, 0).contiguous().view(B, S, d, H)[:, :, -1, :]

        attn_mask = None
        if causal:
            attn_mask = torch.triu(
                torch.full((S, S), float("-inf"), device=h.device), diagonal=1
            )

        if self.mem_type in ["hc", "h"]:
            h_out, _ = self.h_attn(h, h, h, attn_mask=attn_mask)
            h = self.h_norm1(h + h_out)
            h = self.h_norm2(h + self.h_ffn(h))
        if self.mem_type in ["hc", "c"]:
            c_out, _ = self.c_attn(c, c, c, attn_mask=attn_mask)
            c = self.c_norm1(c + c_out)
            c = self.c_norm2(c + self.c_ffn(c))
        if self.mem_type == "h":
            c = torch.zeros_like(c)
        if self.mem_type == "c":
            h = torch.zeros_like(h)

        # Reshape back to (d, B*S, H), matching input d for shape compatibility
        # with MemLSTM's output in FusedMem.
        h = h.view(B * S, H).unsqueeze(0).expand(d, -1, -1).contiguous()
        c = c.view(B * S, H).unsqueeze(0).expand(d, -1, -1).contiguous()
        return (h, c)


class FusedMem(nn.Module):
    """Runs MemLSTM and MemAttention in parallel on segment-level states, fuses with learnable scalar gates.

    h_out = h_lstm + alpha_h * h_attn
    c_out = c_lstm + alpha_c * c_attn

    For mem_type='id', returns hc unchanged.
    For mem_type='h' or 'c', only the active state is fused; the other is zeroed (matches base semantics).
    """

    def __init__(
        self,
        hidden_size,
        num_heads=4,
        dropout=0.0,
        bidirectional=False,
        mem_type="hc",
        norm_type="cLN",
    ):
        super().__init__()
        self.mem_type = mem_type

        if mem_type == "id":
            return

        self.mem_lstm = MemLSTM(
            hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
            mem_type=mem_type,
            norm_type=norm_type,
        )
        self.mem_attn = MemAttention(
            hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            mem_type=mem_type,
            norm_type=norm_type,
        )

        if mem_type in ["hc", "h"]:
            self.alpha_h = nn.Parameter(torch.tensor(0.5))
        if mem_type in ["hc", "c"]:
            self.alpha_c = nn.Parameter(torch.tensor(0.5))

    def extra_repr(self):
        return f"mem_type={self.mem_type}"

    def forward(self, hc, S):
        if self.mem_type == "id":
            return hc

        h_lstm, c_lstm = self.mem_lstm(hc, S)
        h_attn, c_attn = self.mem_attn(hc, S)

        if self.mem_type == "hc":
            h_out = h_lstm + self.alpha_h * h_attn
            c_out = c_lstm + self.alpha_c * c_attn
        elif self.mem_type == "h":
            h_out = h_lstm + self.alpha_h * h_attn
            c_out = torch.zeros_like(c_lstm)
        elif self.mem_type == "c":
            h_out = torch.zeros_like(h_lstm)
            c_out = c_lstm + self.alpha_c * c_attn
        return (h_out, c_out)


class SegLSTM(nn.Module):
    """Seg-LSTM of base SkiM, kept intact."""

    def __init__(
        self, input_size, hidden_size, dropout=0.0, bidirectional=False, norm_type="cLN"
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        self.lstm = nn.LSTM(
            input_size, hidden_size, 1, batch_first=True, bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)
        self.norm = choose_norm(
            norm_type=norm_type, channel_size=input_size, shape="BTD"
        )

    def forward(self, input, hc):
        B, T, H = input.shape
        if hc is None:
            d = self.num_direction
            h = torch.zeros(d, B, self.hidden_size, dtype=input.dtype, device=input.device)
            c = torch.zeros(d, B, self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            h, c = hc
        output, (h, c) = self.lstm(input, (h, c))
        output = self.dropout(output)
        output = self.proj(output.contiguous().view(-1, output.shape[2])).view(
            input.shape
        )
        output = input + self.norm(output)
        return output, (h, c)


class SkiM(nn.Module):
    """SkiM v2b — base SkiM where MemLSTM is wrapped in FusedMem (MemLSTM + MemAttention, gated sum)."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=0.0,
        num_blocks=2,
        segment_size=20,
        bidirectional=True,
        mem_type="hc",
        norm_type="gLN",
        seg_overlap=False,
        num_heads=4,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.segment_size = segment_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.mem_type = mem_type
        self.norm_type = norm_type
        self.seg_overlap = seg_overlap
        self.num_heads = num_heads

        assert mem_type in ["hc", "h", "c", "id", None], (
            f"only support 'hc', 'h', 'c', 'id', and None, current type: {mem_type}"
        )

        self.seg_lstms = nn.ModuleList(
            [
                SegLSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    norm_type=norm_type,
                )
                for _ in range(num_blocks)
            ]
        )
        if self.mem_type is not None:
            self.fused_mems = nn.ModuleList(
                [
                    FusedMem(
                        hidden_size,
                        num_heads=num_heads,
                        dropout=dropout,
                        bidirectional=bidirectional,
                        mem_type=mem_type,
                        norm_type=norm_type,
                    )
                    for _ in range(num_blocks - 1)
                ]
            )
        self.output_fc = nn.Sequential(
            nn.PReLU(), nn.Conv1d(input_size, output_size, 1)
        )

    def _padfeature(self, input):
        B, T, D = input.shape
        rest = self.segment_size - T % self.segment_size
        if rest > 0:
            input = torch.nn.functional.pad(input, (0, 0, 0, rest))
        return input, rest

    def forward(self, input):
        B, T, D = input.shape

        if self.seg_overlap:
            input, rest = split_feature(
                input.transpose(1, 2), segment_size=self.segment_size
            )
            input = input.permute(0, 3, 2, 1).contiguous()
        else:
            input, rest = self._padfeature(input=input)
            input = input.view(B, -1, self.segment_size, D)
        B, S, K, D = input.shape

        assert K == self.segment_size

        output = input.view(B * S, K, D).contiguous()
        hc = None
        for i in range(self.num_blocks):
            output, hc = self.seg_lstms[i](output, hc)
            if self.mem_type and i < self.num_blocks - 1:
                hc = self.fused_mems[i](hc, S)

        if self.seg_overlap:
            output = output.view(B, S, K, D).permute(0, 3, 2, 1)
            output = merge_feature(output, rest)
            output = self.output_fc(output).transpose(1, 2)
        else:
            output = output.view(B, S * K, D)[:, :T, :]
            output = self.output_fc(output.transpose(1, 2)).transpose(1, 2)
        return output

    def forward_stream(self, input_frame, states):
        raise NotImplementedError(
            "Streaming inference is not supported for SkiM v2b "
            "(parallel MemAttention branch needs full segment context). "
            "This variant targets offline non-causal training."
        )
