from typing import Literal, Optional
import numpy as np
import torch
import torchaudio
import torch.nn as nn
from einops import rearrange
from music2latent.audio import to_representation_encoder
from music2latent.models import Encoder as m2l_encoder
from music2latent.utils import download_model
from music2latent.hparams_inference import load_path_inference_default
from mockmt3.model.m2l_helper import m2l_hparams


class ResampleLayer(nn.Module):

    def __init__(self, orig_freq: int = 16000, new_freq: int = 44100):
        super(ResampleLayer, self).__init__()
        self.resample = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)

    def forward(self, x):
        return self.resample(x)


class M2LLayer(nn.Module):

    def __init__(self, pretrained: bool = True, extract_features: bool = True, apply_1x1conv_to_output: bool = False):
        super(M2LLayer, self).__init__()
        self.extract_features = extract_features
        self.apply_1x1conv_to_output = apply_1x1conv_to_output
        self.resampler = ResampleLayer(orig_freq=16000, new_freq=44100)
        self.pad_size = self.calculate_pad_size(input_length=32767)
        self.encoder = m2l_encoder()
        if pretrained:
            download_model()
            if torch.cuda.is_available():
                checkpoint = torch.load(load_path_inference_default)
            else:
                checkpoint = torch.load(load_path_inference_default, map_location=torch.device('cpu'))
            encoder_state_dict = {
                k.replace('encoder.', ''): v for k, v in checkpoint['gen_state_dict'].items() if 'encoder' in k
            }
            self.encoder.load_state_dict(encoder_state_dict)

    def calculate_pad_size(self, input_length: int) -> int:
        # calculate only once, the pad size to avoid losing frames
        resampled_length = int(np.ceil(input_length * 44100 / 16000))
        hop = m2l_hparams["hop"]
        downscaling_factor = 2**m2l_hparams["freq_downsample_list"].count(0)
        multiple = hop * downscaling_factor

        target_length = (resampled_length // multiple) * multiple + 3 * hop
        return max(0, target_length - resampled_length)

    def apply_padding(self, x):
        # right-padding
        return nn.functional.pad(x, (0, self.pad_size))

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x = self.resampler(x)  # (B, 1, 90315) for 2.xx s audio
        x = self.apply_padding(x)  # (B, 1, 91648)
        x = to_representation_encoder(x[:, 0, :])  # (B, 2, 1024, 176)
        x = self.encoder(x, extract_features=self.extract_features)  # (B, D=64, T=22) or (B, 8192, 22)
        if self.apply_1x1conv_to_output:
            x = self.encoder.bottleneck_layers[0](x)
        return rearrange(x, 'b d t -> b t d')  # (B, T=22, D=64) or (B, 8192, 22) or (B, 512, 22)


def test():
    resample_layer = ResampleLayer()
    audio16 = torch.randn(3, 1, 32767)  # (B, 1, T), 2.048 s at 16k
    audio44 = resample_layer(audio16)  # (3, 1, 90315)\

    audio = audio44
    hop = m2l_hparams["hop"]
    freq_downsample_list = m2l_hparams["freq_downsample_list"]
    downscaling_factor = 2**freq_downsample_list.count(0)
    cropped_length = ((((audio.shape[-1] - 3 * hop) // hop) // downscaling_factor) * hop * downscaling_factor) + 3 * hop
