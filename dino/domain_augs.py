"""Paxos2020-style train augmentations for Paxos2025 frames (cross-domain gap, 2026-11-02).

Ranges calibrated from summary statistics of the Paxos2020 clips (2020 vs 2025, fundus pixels at 512 px): less red/yellow
(LAB a 0.8 vs 7.7, b 6.8 vs 16.1), chroma x0.64, contrast (L std) x0.8, high-pass noise x2.2, plus glare /
streak / bottom-edge light artifacts seen in the frames. Works on float tensors in [0, 1], [C, H, W] or [T, C, H, W];
all parameters except the noise are drawn once per call, so a bag stage (--bag_consistent_augs) gets one look per video.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF

P20_GROUPS = ("color", "quality", "light")


def _u(lo, hi):
    return lo + (hi - lo) * torch.rand(1).item()


def _p(p):
    return torch.rand(1).item() < p


def _blend(x, alpha, color):
    """x [T, 3, H, W], alpha [H, W] in [0, 1], color (r, g, b) or [3, H, W]."""
    color = torch.as_tensor(color, dtype=x.dtype, device=x.device)
    if color.dim() == 1:
        color = color.view(3, 1, 1)
    return x * (1 - alpha) + color * alpha


class Paxos2020Aug(nn.Module):
    """groups: subset of P20_GROUPS; each group is applied with probability p (per call), in the order color -> light -> quality."""

    def __init__(self, groups=P20_GROUPS, p=0.7, mask_thresh=0.06):
        super().__init__()
        self.groups, self.p, self.mask_thresh = tuple(groups), p, mask_thresh

    def forward(self, x):
        single = x.dim() == 3
        x = x.unsqueeze(0) if single else x
        x = x.float()
        mask = (x.amax(1, keepdim=True) > self.mask_thresh).to(x.dtype)  # [T, 1, H, W], fundus before the aug
        if "color" in self.groups and _p(self.p):
            x = self._color(x, mask)
        if "light" in self.groups and _p(self.p):
            x = self._light(x, mask)
        if "quality" in self.groups and _p(self.p):
            x = self._quality(x)
        x = x.clamp(0, 1) * mask
        return x[0] if single else x

    @staticmethod
    def _color(x, mask):
        # grey-blue / olive tint: less red, more blue, lower saturation and contrast
        gains = torch.tensor([_u(0.8, 0.97), _u(0.9, 1.05), _u(1.0, 1.35)], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        x = (x * gains).clamp(0, 1)
        lum = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        x = lum + _u(0.65, 1.0) * (x - lum)
        if _p(0.7):
            mean = (lum * mask).sum((2, 3), keepdim=True) / mask.sum((2, 3), keepdim=True).clamp(min=1)
            x = mean + _u(0.6, 0.95) * (x - mean)
        if _p(0.3):
            x = x.clamp(0, 1) ** _u(0.8, 1.25)
        return x.clamp(0, 1)

    @staticmethod
    def _geometry(mask, device):
        """Fundus centre (cy, cx) and radius in pixels from the union mask, plus the pixel grid."""
        m = mask.amax(0)[0]
        H, W = m.shape
        yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                                torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
        area = m.sum()
        if area < 100:
            return H / 2, W / 2, min(H, W) / 2, yy, xx
        cy, cx = (yy * m).sum() / area, (xx * m).sum() / area
        return cy.item(), cx.item(), math.sqrt(area.item() / math.pi), yy, xx

    def _light(self, x, mask):
        cy, cx, R, yy, xx = self._geometry(mask, x.device)
        if _p(0.5):  # glare / haze crescent at the rim, long axis tangential
            th = _u(0, 2 * math.pi)
            off = R * _u(0.6, 0.95)
            gy, gx = cy + off * math.sin(th), cx + off * math.cos(th)
            ra = R * _u(0.4, 0.8); rb = ra * _u(0.4, 0.9)
            dy, dx = yy - gy, xx - gx
            t = -dx * math.sin(th) + dy * math.cos(th)  # tangential
            r = dx * math.cos(th) + dy * math.sin(th)   # radial
            d2 = (t / ra) ** 2 + (r / rb) ** 2
            x = _blend(x, _u(0.4, 0.95) * torch.exp(-d2 ** 1.5), (_u(0.75, 0.95), _u(0.9, 1.0), 1.0))
        if _p(0.3):  # thin bright scratch / dust streaks
            for _ in range(int(_u(1, 5))):
                th, rho = _u(0, 2 * math.pi), R * math.sqrt(_u(0, 0.8))
                ay, ax = cy + rho * math.sin(th), cx + rho * math.cos(th)
                phi, L = _u(0, math.pi), R * _u(0.1, 0.4)
                by, bx = ay + L * math.sin(phi), ax + L * math.cos(phi)
                vy, vx = by - ay, bx - ax
                s = (((yy - ay) * vy + (xx - ax) * vx) / (vy * vy + vx * vx)).clamp(0, 1)
                d = torch.sqrt((yy - ay - s * vy) ** 2 + (xx - ax - s * vx) ** 2)
                x = _blend(x, _u(0.4, 0.8) * torch.exp(-0.5 * (d / _u(0.7, 1.5)) ** 2), (_u(0.7, 0.9), 0.95, 1.0))
        if _p(0.15):  # bright "tongue" at the bottom edge
            ty, tx = cy + R, cx + R * _u(-0.2, 0.2)
            a, b = R * _u(0.15, 0.3), R * _u(0.06, 0.12)
            d2 = ((xx - tx) / a) ** 2 + ((yy - ty) / b) ** 2
            x = _blend(x, _u(0.6, 0.95) * torch.sigmoid(8 * (1 - d2)), (1.0, _u(0.85, 0.95), _u(0.75, 0.9)))
        return x

    @staticmethod
    def _quality(x):
        H, W = x.shape[-2:]
        # blur / noise x1.5 of the calibrated values, resolution loss x0.3-0.8 (2026-11-02, after visual inspection)
        if _p(0.5):  # resolution loss
            f = _u(0.3, 0.8)
            x = F.interpolate(x, size=(max(8, int(H * f)), max(8, int(W * f))), mode="bilinear", antialias=True, align_corners=False)
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        if _p(0.6):
            sigma = 1.5 * _u(0.5, 1.2)
            k = 2 * math.ceil(3 * sigma) + 1
            x = TF.gaussian_blur(x, [k, k], [sigma, sigma])
        if _p(0.8):  # sensor noise: luminance + weaker per-channel chroma part, new per frame
            sigma = 1.5 * _u(0.003, 0.01)
            x = x + sigma * torch.randn_like(x[:, :1]) + 0.5 * sigma * torch.randn_like(x)
        return x
