import torch
import torch.nn as nn
from espnet2.enh.layers.dprnn import SingleRNN, merge_feature, split_feature
from espnet2.enh.layers.tcn import choose_norm


class MemAttention(nn.Module):
    """Multi-Head Self-Attention replacing Mem-LSTM in SkiM."""

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

        # Reshape: attention operates on the hidden dimension H, not d*H
        # h/c shape: (d, B*S, H) -> (B, S, d, H) -> (B, S, H) by taking last layer
        h = (
            h.transpose(1, 0).contiguous().view(B, S, d, H)[:, :, -1, :]
        )  # Take last layer: (B, S, H)
        c = (
            c.transpose(1, 0).contiguous().view(B, S, d, H)[:, :, -1, :]
        )  # Take last layer: (B, S, H)

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

        # Reshape back: (B, S, H) -> (B*S, H) -> (2, B*S, H) for bidirectional
        # Duplicate the attended output for both directions
        h = h.view(B * S, H)
        c = c.view(B * S, H)
        h = torch.stack([h, h], dim=0).contiguous()  # (2, B*S, H)
        c = torch.stack([c, c], dim=0).contiguous()  # (2, B*S, H)
        return (h, c)

    def forward_one_step(self, hc, state, causal=True):
        if self.mem_type == "id":
            return hc, state
        h, c = hc
        d, B, H = h.shape
        # Take last layer only: (d, B, H) -> (B, 1, H)
        h = h[-1:, :, :].transpose(1, 0).contiguous().view(B, 1, H)
        c = c[-1:, :, :].transpose(1, 0).contiguous().view(B, 1, H)
        if self.mem_type in ["hc", "h"]:
            h_out, _ = self.h_attn(h, h, h)
            h = self.h_norm1(h + h_out)
            h = self.h_norm2(h + self.h_ffn(h))
        if self.mem_type in ["hc", "c"]:
            c_out, _ = self.c_attn(c, c, c)
            c = self.c_norm1(c + c_out)
            c = self.c_norm2(c + self.c_ffn(c))
        if self.mem_type == "h":
            c = torch.zeros_like(c)
        if self.mem_type == "c":
            h = torch.zeros_like(h)
        # Reshape back: (B, 1, H) -> (2, B, H) for bidirectional
        h = h.squeeze(1)  # (B, H)
        c = c.squeeze(1)  # (B, H)
        h = torch.stack([h, h], dim=0).contiguous()  # (2, B, H)
        c = torch.stack([c, c], dim=0).contiguous()  # (2, B, H)
        return (h, c), state


class SegLSTM(nn.Module):
    """Seg-LSTM block of SkiM (unchanged from original)."""

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
            h = torch.zeros(
                d, B, self.hidden_size, dtype=input.dtype, device=input.device
            )
            c = torch.zeros(
                d, B, self.hidden_size, dtype=input.dtype, device=input.device
            )
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
    """SkiM with MemAttention (Multi-Head Self-Attention) replacing Mem-LSTM."""

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
        self.num_blocks = num_blocks
        self.mem_type = mem_type
        self.seg_overlap = seg_overlap
        self.num_heads = num_heads

        self.seg_lstms = nn.ModuleList(
            [
                SegLSTM(
                    input_size,
                    hidden_size,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    norm_type=norm_type,
                )
                for _ in range(num_blocks)
            ]
        )
        if mem_type is not None:
            self.mem_lstms = nn.ModuleList(
                [
                    MemAttention(
                        hidden_size,
                        num_heads=num_heads,
                        dropout=dropout,
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
            input, rest = self._padfeature(input)
            input = input.view(B, -1, self.segment_size, D)
        B, S, K, D = input.shape

        output = input.view(B * S, K, D).contiguous()
        hc = None
        for i in range(self.num_blocks):
            output, hc = self.seg_lstms[i](output, hc)
            if self.mem_type and i < self.num_blocks - 1:
                hc = self.mem_lstms[i](hc, S, causal=False)

        if self.seg_overlap:
            output = output.view(B, S, K, D).permute(0, 3, 2, 1)
            output = merge_feature(output, rest)
            output = self.output_fc(output).transpose(1, 2)
        else:
            output = output.view(B, S * K, D)[:, :T, :]
            output = self.output_fc(output.transpose(1, 2)).transpose(1, 2)
        return output

    def forward_stream(self, input_frame, states):
        B, _, N = input_frame.shape

        def empty_seg_states():
            shp = (1, B, self.hidden_size)
            return (
                torch.zeros(*shp, device=input_frame.device, dtype=input_frame.dtype),
                torch.zeros(*shp, device=input_frame.device, dtype=input_frame.dtype),
            )

        if not states:
            states = {
                "current_step": 0,
                "seg_state": [empty_seg_states() for _ in range(self.num_blocks)],
                "mem_state": [[None, None] for _ in range(self.num_blocks - 1)],
            }

        output = input_frame
        if states["current_step"] and states["current_step"] % self.segment_size == 0:
            tmp_states = [empty_seg_states() for _ in range(self.num_blocks)]
            for i in range(self.num_blocks - 1):
                tmp_states[i + 1], states["mem_state"][i] = self.mem_lstms[
                    i
                ].forward_one_step(
                    states["seg_state"][i], states["mem_state"][i], causal=True
                )
            states["seg_state"] = tmp_states

        for i in range(self.num_blocks):
            output, states["seg_state"][i] = self.seg_lstms[i](
                output, states["seg_state"][i]
            )

        states["current_step"] += 1
        output = self.output_fc(output.transpose(1, 2)).transpose(1, 2)
        return output, states
