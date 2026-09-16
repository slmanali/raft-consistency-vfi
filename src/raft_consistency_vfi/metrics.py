"""Explicit RGB metrics. No resizing, border cropping (except SSIM support), or Y conversion."""

import math
import numpy as np
from scipy.ndimage import gaussian_filter


def quality(prediction, target):
    a, b = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError("Metrics require equal HWC RGB images")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite metric input")
    if min(a.shape[:2]) < 11:
        raise ValueError("SSIM requires images at least 11 by 11")
    if min(a.min(), b.min()) < 0 or max(a.max(), b.max()) > 1:
        raise ValueError("Expected image values in [0,1]")
    mse = float(np.mean((a-b)**2))
    # Wang et al. parameters; 11x11 Gaussian, population covariance, RGB mean.
    def smooth(x):
        return gaussian_filter(x, sigma=(1.5,1.5,0), truncate=3.5, mode="reflect")
    ma, mb = smooth(a), smooth(b)
    va, vb = smooth(a*a)-ma*ma, smooth(b*b)-mb*mb
    cov = smooth(a*b)-ma*mb
    ssim = ((2*ma*mb+0.01**2)*(2*cov+0.03**2)
            / ((ma*ma+mb*mb+0.01**2)*(va+vb+0.03**2)))
    return {"psnr": math.inf if mse == 0 else -10*math.log10(mse),
            "ssim": float(ssim[5:-5,5:-5].mean())}


class LPIPSMetric:
    def __init__(self, device):
        try:
            import torch
            import lpips
        except ImportError as exc:
            raise RuntimeError("Install the models extra to calculate LPIPS") from exc
        self.torch, self.device = torch, torch.device(device)
        self.model = lpips.LPIPS(net="alex", version="0.1").to(self.device).eval()

    def __call__(self, a, b):
        def tensor(x):
            return self.torch.from_numpy(x.transpose(2,0,1).copy()).unsqueeze(0).to(self.device)*2-1
        with self.torch.inference_mode():
            return float(self.model(tensor(a),tensor(b)).item())
