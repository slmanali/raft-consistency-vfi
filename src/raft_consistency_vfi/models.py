"""Optional pretrained RAFT adapter; no ground-truth argument is accepted."""

import hashlib
from pathlib import Path


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


class RAFT:
    def __init__(self, device="cpu", weights="C_T_V2", iterations=20):
        try:
            import torch
            import torchvision
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("Install a compatible torch/torchvision pair; see README") from exc
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if weights == "DEFAULT" or weights not in Raft_Large_Weights.__members__:
            raise ValueError("Select an explicit RAFT weight enum, such as C_T_V2")
        enum = Raft_Large_Weights[weights]
        self.torch, self.device, self.iterations = torch, torch.device(device), iterations
        self.model = raft_large(weights=enum, progress=True).to(self.device).eval()
        weight_path = Path(torch.hub.get_dir())/"checkpoints"/enum.url.rsplit("/",1)[-1]
        self.metadata = {"raft_weights": weights, "raft_weight_url": enum.url,
            "raft_weight_sha256": file_sha256(weight_path), "iterations": iterations,
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "device": str(self.device), "precision": "float32", "synthesis": "numpy_cpu"}
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            self.metadata["gpu"] = torch.cuda.get_device_name(self.device)

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def __call__(self, image0, image1):
        torch = self.torch
        if image0.shape != image1.shape:
            raise ValueError("RAFT inputs must have equal shapes")
        h, w = image0.shape[:2]
        hh, ww = max(128,(h+7)//8*8), max(128,(w+7)//8*8)
        def prepare(image):
            x = torch.from_numpy(image.transpose(2,0,1).copy()).unsqueeze(0).to(self.device)
            x = torch.nn.functional.pad(x, (0,ww-w,0,hh-h), mode="replicate")
            return (2*x-1).contiguous()
        a, b = prepare(image0), prepare(image1)
        with torch.inference_mode():
            forward = self.model(a,b,num_flow_updates=self.iterations)[-1]
            backward = self.model(b,a,num_flow_updates=self.iterations)[-1]
        # CPU transfer is intentional and is included in wall-clock flow timing.
        return tuple(x[0,:,:h,:w].permute(1,2,0).cpu().numpy().copy() for x in (forward,backward))
