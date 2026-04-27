"""ConvEncoder variant that uses abs() instead of relu().

relu() discards negative filter activations, causing the encoder-decoder chain
to have a non-flat frequency response (perceptual bass boost in separated audio).
abs() preserves both polarities, giving a flatter spectral response.

Drop-in replacement for espnet2.enh.encoder.conv_encoder.ConvEncoder.
"""

import torch
from espnet2.enh.encoder.conv_encoder import ConvEncoder


class ConvEncoderAbs(ConvEncoder):
    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        assert input.dim() == 2, "Currently only support single channel input"
        input = torch.unsqueeze(input, 1)
        feature = self.conv1d(input)
        feature = torch.abs(feature)          # abs instead of relu
        feature = feature.transpose(1, 2)
        flens = (ilens - self.kernel_size) // self.stride + 1
        return feature, flens
