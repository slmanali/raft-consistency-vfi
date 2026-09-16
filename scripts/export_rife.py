"""Export floating-point predictions from the official ECCV2022-RIFE checkout.

No RIFE code or weights are redistributed. Read README before using this adapter.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("repo","weights","manifest","output"):
        p.add_argument(f"--{n}",required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--warmup",type=int,default=2)
    args=p.parse_args()
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if args.device=="cpu":
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    import numpy as np
    import torch
    from raft_consistency_vfi.data import read_image, read_manifest
    from raft_consistency_vfi.models import file_sha256
    from raft_consistency_vfi.cli import environment, new_output, write_csv, write_json
    repo=Path(args.repo).resolve()
    commit=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True).strip():
        raise RuntimeError("RIFE checkout must be clean; put weights outside it or gitignore them")
    sys.path.insert(0,str(repo))
    from model.RIFE import Model
    samples=read_manifest(args.manifest)
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    model=Model()
    state=torch.load(args.weights,map_location="cpu",weights_only=True)
    state={k.removeprefix("module."):v for k,v in state.items()}
    model.flownet.load_state_dict(state,strict=True)
    model.flownet.to(device).eval()
    out=new_output(args.output)
    pred_dir=out/"predictions"; pred_dir.mkdir()
    meta={**environment(),"model":"RIFE_ECCV2022","status":"running","code_commit":commit,
        "weights_sha256":file_sha256(args.weights),"device":str(device),
        "manifest_sha256":file_sha256(args.manifest),"precision":"float32",
        "TTA":False,"scale":1,"timestep":0.5,"channel_order":"BGR at model boundary",
        "timing_protocol":"wall-clock tensor preparation, H2D, inference, D2H and crop; excludes file I/O and warmup"}
    write_json(out/"provenance.json",meta)
    rows=[]; warmed=set()
    def synchronize():
        if device.type=="cuda": torch.cuda.synchronize()
    def predict(a,b):
        h,w=a.shape[:2]
        def prepare(x):
            x=torch.from_numpy(x[:,:,::-1].transpose(2,0,1).copy()).unsqueeze(0).to(device)
            return torch.nn.functional.pad(x,(0,(-w)%32,0,(-h)%32))
        with torch.inference_mode():
            result=model.inference(prepare(a),prepare(b),scale=1,TTA=False,timestep=0.5)
        return np.clip(result[0,:,:h,:w].permute(1,2,0).cpu().numpy()[:,:,::-1].copy(),0,1)
    for sample in samples:
        a,b=read_image(sample.input0),read_image(sample.input1)
        if a.shape!=b.shape:
            raise ValueError(f"Input dimensions differ: {sample.id}")
        if a.shape not in warmed:
            for _ in range(args.warmup): predict(a,b)
            warmed.add(a.shape)
        synchronize(); start=time.perf_counter()
        pred=predict(a,b)
        synchronize(); duration=(time.perf_counter()-start)*1000
        np.save(pred_dir/f"{sample.id}.npy",pred,allow_pickle=False)
        rows.append({"id":sample.id,"time_ms":duration})
        write_csv(out/"timings.csv",rows)
        print(sample.id,flush=True)
    meta["status"]="completed"; write_json(out/"provenance.json",meta)


if __name__=="__main__":
    main()
