"""Deterministic NumPy reference synthesis. Flow channels are (dx, dy), in pixels.

Consistency samples the reverse flow at the forward-mapped coordinate. Images
are forward splatted; endpoint flow is never mistaken for target-to-source flow.
"""

import numpy as np


def _field(a, channels, name):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 3 or a.shape[2] != channels or min(a.shape[:2]) < 1:
        raise ValueError(f"{name} must have shape (H,W,{channels})")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains nonfinite values")
    return a


def grid(height, width):
    y, x = np.mgrid[:height, :width].astype(np.float32)
    return np.stack((x, y), axis=-1)


def endpoint_valid(flow):
    """Geometric endpoint visibility only, without testing flow consistency."""
    flow = _field(flow, 2, "flow")
    h, w = flow.shape[:2]
    xy = grid(h, w) + flow
    return ((xy[...,0] >= 0) & (xy[...,0] <= w-1)
            & (xy[...,1] >= 0) & (xy[...,1] <= h-1)).astype(np.float32)


def bilinear_sample(field, xy):
    """Sample HWC field. Return samples and mask of coordinates inside the image."""
    h, w = field.shape[:2]
    x, y = xy[..., 0], xy[..., 1]
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    ax, ay = (x - x0)[..., None], (y - y0)[..., None]
    value = ((1-ax)*(1-ay)*field[y0, x0] + ax*(1-ay)*field[y0, x1]
             + (1-ax)*ay*field[y1, x0] + ax*ay*field[y1, x1])
    return value.astype(np.float32), valid


def consistency(flow, reverse, alpha=0.01, beta=0.5):
    """UnFlow-style squared consistency residual and threshold (beta in px²)."""
    flow = _field(flow, 2, "flow")
    reverse = _field(reverse, 2, "reverse")
    if flow.shape != reverse.shape:
        raise ValueError("Flow shapes differ")
    if not np.isfinite([alpha, beta]).all() or alpha < 0 or beta <= 0:
        raise ValueError("Require finite alpha >= 0 and beta > 0")
    other, valid = bilinear_sample(reverse, grid(*flow.shape[:2]) + flow)
    residual2 = np.sum((flow + other)**2, axis=-1)
    threshold = alpha*(np.sum(flow**2, axis=-1) + np.sum(other**2, axis=-1)) + beta
    hard = (valid & (residual2 <= threshold)).astype(np.float32)
    soft = np.where(valid, np.exp(-residual2 / threshold), 0).astype(np.float32)
    return {"hard": hard, "soft": soft, "valid": valid,
            "residual2": residual2, "threshold": threshold}


def forward_splat(image, displacement, weights):
    """Accumulate weighted RGB and weights with a separable bilinear kernel.

    Contributions outside the target image are discarded, never wrapped/clipped.
    Normalization is deferred until the contributions from both images are added.
    """
    image = _field(image, 3, "image")
    displacement = _field(displacement, 2, "displacement")
    h, w = image.shape[:2]
    weights = np.asarray(weights, dtype=np.float32)
    if displacement.shape[:2] != (h, w) or weights.shape != (h, w):
        raise ValueError("Image, displacement and weight shapes differ")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Weights must be finite and nonnegative")
    target = grid(h, w) + displacement
    x, y = target[..., 0], target[..., 1]
    # Filter extreme finite coordinates before conversion to integer indices.
    possible = (x > -1) & (x < w) & (y > -1) & (y < h)
    x, y = np.where(possible, x, 0), np.where(possible, y, 0)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    ax, ay = x-ix, y-iy
    numerator = np.zeros((h*w, 3), dtype=np.float32)
    denominator = np.zeros(h*w, dtype=np.float32)
    for ox, oy, kernel in ((0,0,(1-ax)*(1-ay)), (1,0,ax*(1-ay)),
                           (0,1,(1-ax)*ay), (1,1,ax*ay)):
        xx, yy = ix+ox, iy+oy
        ok = possible & (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        ww = (kernel[ok]*weights[ok]).astype(np.float32)
        index = yy[ok]*w+xx[ok]
        np.add.at(numerator, index, image[ok]*ww[:, None])
        np.add.at(denominator, index, ww)
    return numerator.reshape(h, w, 3), denominator.reshape(h, w)


def interpolate(image0, image1, flow01, flow10, weights0=None, weights1=None, t=0.5):
    """Linear-trajectory interpolation; identical temporal-blend fallback for holes."""
    if not np.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("t must be in [0,1]")
    image0 = _field(image0, 3, "image0")
    image1 = _field(image1, 3, "image1")
    if image0.shape != image1.shape:
        raise ValueError("Image shapes differ")
    h, w = image0.shape[:2]
    if t in (0, 1):
        return (image0 if t == 0 else image1).copy(), np.zeros((h,w), dtype=bool)
    if weights0 is None:
        weights0 = np.ones((h,w), dtype=np.float32)
    if weights1 is None:
        weights1 = np.ones((h,w), dtype=np.float32)
    n0, d0 = forward_splat(image0, t*flow01, (1-t)*weights0)
    n1, d1 = forward_splat(image1, (1-t)*flow10, t*weights1)
    denominator = d0+d1
    holes = denominator <= 1e-8
    result = (n0+n1)/np.maximum(denominator[...,None], 1e-8)
    result[holes] = ((1-t)*image0+t*image1)[holes]
    return np.clip(result, 0, 1), holes
