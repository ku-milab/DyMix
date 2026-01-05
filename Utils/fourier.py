import warnings

from PIL import Image
import random
import numpy as np
from math import sqrt
import torch
import math

def mixup(src, trg, alpha):
    #lam = np.random.uniform(0, alpha)
    lam = alpha

    mix_img = lam * src + (1 - lam) * trg

    return mix_img

# abs : amplitude, angle : phase
def fft_amp_mix(src, int_src):
    fft_src = torch.fft.fftn(src)
    abs_src, angle_src = torch.abs(fft_src), torch.angle(fft_src)

    fft_int_src = torch.fft.fftn(int_src)
    abs_int_src, angle_int_src = torch.abs(fft_int_src), torch.angle(fft_int_src)

    fft_src = abs_int_src * torch.exp((1j) * angle_src)

    mixed_img = torch.abs(torch.fft.ifftn(fft_src))

    return mixed_img

def apr(source, target, ratio=1.0):
    b, c, h, w, d = source.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    d_crop = int(d * sqrt(ratio))

    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2
    d_start = d // 2 - d_crop // 2

    src_fft = torch.fft.fftn(source)
    src_shift = torch.fft.fftshift(src_fft)
    src_abs, src_pha = torch.abs(src_shift), torch.angle(src_shift)
    trg_fft = torch.fft.fftn(target)
    trg_shift = torch.fft.fftshift(trg_fft)
    trg_abs, trg_pha = torch.abs(trg_shift), torch.angle(trg_shift)

    src_abs[h_start:h_start+h_crop, w_start:w_start+w_crop, d_start:d_start+d_crop] = trg_abs[h_start:h_start+h_crop, w_start:w_start+w_crop, d_start:d_start+d_crop]

    fft_src = src_abs * torch.exp((1j) * src_pha)
    fft_src = torch.fft.ifftshift(fft_src)

    new_img = torch.abs(torch.fft.ifftn(fft_src))

    return new_img


def fft_mixup_block(source, target, ratio=1.0):
    lam = np.random.uniform(0, 1.0)
    #lam = 0.5

    b, c, h, w, d = source.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    d_crop = int(d * sqrt(ratio))

    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2
    d_start = d // 2 - d_crop // 2

    src_fft = torch.fft.fftn(source)
    src_amp, src_pha = torch.abs(src_fft), torch.angle(src_fft)
    amp_shift_src = torch.fft.fftshift(src_amp)

    trg_fft = torch.fft.fftn(target)
    trg_amp, trg_pha = torch.abs(trg_fft), torch.angle(trg_fft)
    amp_shift_trg = torch.fft.fftshift(trg_amp)
    
    amp_shift_trg[h_start:h_start + h_crop, w_start:w_start + w_crop, d_start:d_start + d_crop] = lam * amp_shift_src[h_start:h_start+h_crop, w_start:w_start+w_crop, d_start:d_start+d_crop] + (1 - lam) * amp_shift_trg[h_start:h_start+h_crop, w_start:w_start+w_crop, d_start:d_start+d_crop]
    mix_amp = torch.fft.ifftshift(amp_shift_trg)
    mix_img = mix_amp * torch.exp((1j) * trg_pha)
    mixed_img = torch.abs(torch.fft.ifftn(mix_img))

    return mixed_img

def hpf(source, ratio=1.0):
    b, c, h, w, d = source.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    d_crop = int(d * sqrt(ratio))

    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2
    d_start = d // 2 - d_crop // 2

    src_fft = torch.fft.fftn(source)
    src_fshift = torch.fft.fftshift(src_fft)

    src_fshift[h_start:h_start+h_crop, w_start:w_start+w_crop, d_start:d_start+d_crop] = 0

    new_img = torch.abs(torch.fft.ifftn(torch.fft.ifftshift(src_fshift)))

    return new_img