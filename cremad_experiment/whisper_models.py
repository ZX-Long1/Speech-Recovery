import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper

class WhisperEncoder(nn.Module):
    def __init__(self, split_point=4, freeze=True):
        super().__init__()
        self.model = whisper.load_model("small")
        self.encoder = self.model.encoder
        self.split_point = split_point
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, mel):
        x = F.gelu(self.encoder.conv1(mel))
        x = F.gelu(self.encoder.conv2(x))
        x = x.permute(0, 2, 1)
        x = x + self.encoder.positional_embedding[:x.shape[1]]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i == self.split_point - 1:
                return x
        return x

def _up_block(in_c, out_c, bn=True):
    layers = [nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
              nn.Conv1d(in_c, out_c, kernel_size=3, padding=1)]
    if bn:
        layers.append(nn.BatchNorm1d(out_c))
    layers.append(nn.GELU())
    return layers

def _conv_block(in_c, out_c, bn=True):
    layers = [nn.Conv1d(in_c, out_c, kernel_size=3, padding=1)]
    if bn:
        layers.append(nn.BatchNorm1d(out_c))
    layers.append(nn.GELU())
    return layers

class Decoder_whitebox(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.net = nn.Sequential(
            *_up_block(d_model, 512, bn=True),
            *_conv_block(512, 384, bn=True),
            *_conv_block(384, 256, bn=True),
            *_conv_block(256, 192, bn=True),
            *_conv_block(192, 128, bn=True),
            *_conv_block(128, 80, bn=True),
            nn.Conv1d(80, 80, kernel_size=1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.net(x)

class Decoder_blackbox(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.net = nn.Sequential(
            *_up_block(d_model, 384, bn=True),
            *_conv_block(384, 256, bn=True),
            *_conv_block(256, 128, bn=True),
            nn.Conv1d(128, 80, kernel_size=1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.net(x)

def tvd_loss(recon, target):
    diff = recon - target
    return F.l1_loss(diff[:, :, :-1], diff[:, :, 1:])
