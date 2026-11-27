"""Shape-prior loss for the segmentation stage.

The fundus crop target is always an (axis-aligned) ellipse: the annotation CSV
stores center_x/center_y/radius_x/radius_y and the masks are rendered from them.
So besides the pixel-wise DiceCE loss we can add a differentiable ellipse
prior: match the soft first/second moments of the predicted probability map to
those of the GT mask. For a filled ellipse with semi-axes a, b the second
central moments are exactly a^2/4 and b^2/4, so matching moments is equivalent
to matching (center_x, center_y, radius_x, radius_y) of the equivalent
axis-aligned ellipse. With seg_shape_loss_weight=0 the training loss is exactly
the previous DiceCELoss, so old and new loss can be compared in one sweep.
"""

import torch


def ellipse_moments(prob, eps=1e-6):
    """Soft ellipse parameters of a map prob (B, 1, H, W) on a normalized [0, 1]
    coordinate grid. Returns (B, 4) rows (cx, cy, a, b): the centroid and the
    equivalent axis-aligned ellipse semi-axes a = 2*sqrt(var_x), b = 2*sqrt(var_y)."""
    b, _, h, w = prob.shape
    ys = torch.linspace(0.0, 1.0, h, device=prob.device, dtype=prob.dtype).view(1, h, 1)
    xs = torch.linspace(0.0, 1.0, w, device=prob.device, dtype=prob.dtype).view(1, 1, w)
    p = prob[:, 0]
    m00 = p.sum(dim=(1, 2)) + eps
    cx = (p * xs).sum(dim=(1, 2)) / m00
    cy = (p * ys).sum(dim=(1, 2)) / m00
    var_x = (p * (xs - cx.view(b, 1, 1)) ** 2).sum(dim=(1, 2)) / m00
    var_y = (p * (ys - cy.view(b, 1, 1)) ** 2).sum(dim=(1, 2)) / m00
    a = 2.0 * torch.sqrt(var_x.clamp_min(eps))
    bb = 2.0 * torch.sqrt(var_y.clamp_min(eps))
    return torch.stack([cx, cy, a, bb], dim=1)


def ellipse_shape_loss(logits, mask, eps=1e-6):
    """MSE between the soft ellipse parameters of sigmoid(logits) and the hard
    ellipse parameters of the GT mask, in normalized [0, 1] coordinates."""
    pred = ellipse_moments(torch.sigmoid(logits), eps)
    gt = ellipse_moments(mask.float(), eps)
    return ((pred - gt) ** 2).mean()
