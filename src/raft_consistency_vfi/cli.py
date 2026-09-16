"""Command-line experiment runner. Existing output directories are never overwritten."""

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np

from .core import consistency, endpoint_valid, interpolate
from .data import load_triplet, make_snu_manifest, read_image, read_manifest, save_image
from .metrics import LPIPSMetric, quality
from .models import RAFT, file_sha256


def environment():
    result = {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "platform": platform.platform(), "numpy": np.__version__}
    for name in ("scipy", "Pillow", "torch", "torchvision", "lpips"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    try:
        root = Path(__file__).resolve().parents[2]
        result["git_commit"] = subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
        result["git_dirty"] = bool(subprocess.check_output(["git","-C",str(root),"status","--porcelain"],text=True))
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",encoding="utf-8")


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def select_samples(args):
    samples = read_manifest(args.manifest)
    if args.split:
        samples = [s for s in samples if s.split in args.split.split(",")]
    if args.limit:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        samples = samples[:args.limit]
    if not samples:
        raise ValueError("No samples selected")
    return samples


def summarize(rows):
    """Per-split unweighted mean over triplets; no mixing of SNU-FILM modes."""
    result=[]
    for split, variant in sorted({(r["split"],r["variant"]) for r in rows}):
        group=[r for r in rows if r["split"]==split and r["variant"]==variant]
        record={"split":split,"variant":variant,"n":len(group)}
        for key in ("psnr","ssim","lpips","time_ms","hole_fraction"):
            values=[float(r[key]) for r in group if r.get(key) not in ("",None)]
            record[key] = float(np.mean(values)) if values else ""
        result.append(record)
    return result


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to write")
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args):
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    samples=select_samples(args)
    save_ids=set(Path(args.save_ids).read_text().split()) if args.save_ids else set()
    if not save_ids.issubset({s.id for s in samples}):
        raise ValueError("Every saved id must occur in the selected evaluation manifest")
    out=new_output(args.output)
    metadata=environment()
    metadata.update({"status":"running","arguments":vars(args),
        "manifest_sha256":file_sha256(args.manifest),"n":len(samples),
        "metric_protocol":"RGB [0,1], per-triplet PSNR; Gaussian SSIM 11x11 sigma 1.5 population covariance",
        "timing_protocol":"wall clock including CPU synthesis and device transfers; excludes image I/O, weights and metrics",
        "fallback":"unwarped temporal average where splat denominator <= 1e-8"})
    write_json(out/"run.json",metadata)
    model=RAFT(args.device,args.weights,args.iterations) if args.method=="raft" else None
    lpips=LPIPSMetric(args.device) if args.lpips else None
    if model:
        metadata.update(model.metadata)
    write_json(out/"run.json",metadata)
    rows=[]
    warmed=set()
    for sample in samples:
        a,gt,b=load_triplet(sample)
        if model and a.shape not in warmed:
            for _ in range(args.warmup):
                model(a,b)
            warmed.add(a.shape)
        variants=[]
        start=time.perf_counter()
        average=(a+b)/2
        variants.append(("average",average,(time.perf_counter()-start)*1000,0.0))
        if model:
            model.synchronize()
            start=time.perf_counter()
            f,g=model(a,b)
            model.synchronize()
            flow_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter()
            valid0,valid1=endpoint_valid(f),endpoint_valid(g)
            valid_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter()
            c0,c1=consistency(f,g,args.alpha,args.beta),consistency(g,f,args.alpha,args.beta)
            mask_ms=(time.perf_counter()-start)*1000
            # All three variants use the exact same flow arrays and fallback rule.
            for name in ("uniform","hard","soft"):
                start=time.perf_counter()
                w0,w1=(valid0,valid1) if name=="uniform" else (c0[name],c1[name])
                prediction,holes=interpolate(a,b,f,g,w0,w1)
                ms=flow_ms+(valid_ms if name=="uniform" else mask_ms)+(time.perf_counter()-start)*1000
                variants.append((f"raft_{name}",prediction,ms,float(holes.mean())))
            if sample.id in save_ids:
                for name in ("hard","soft"):
                    save_image(out/"images"/sample.id/f"weight0_{name}.png",c0[name])
        for name,prediction,ms,holes in variants:
            metrics=quality(prediction,gt)
            row={"id":sample.id,"split":sample.split,"group":sample.group,"variant":name,
                "height":a.shape[0],"width":a.shape[1],**metrics,
                "lpips":lpips(prediction,gt) if lpips else "",
                "time_ms":ms,"hole_fraction":holes}
            rows.append(row)
            if sample.id in save_ids:
                save_image(out/"images"/sample.id/f"{name}.png",prediction)
        if sample.id in save_ids:
            for name,im in (("input0",a),("target",gt),("input1",b)):
                save_image(out/"images"/sample.id/f"{name}.png",im)
        # Incremental metrics survive interruption; status remains 'running' until done.
        write_csv(out/"per_image.csv",rows)
        print(f"{sample.id}: evaluated",flush=True)
    write_csv(out/"summary.csv",summarize(rows))
    if model:
        baseline={r["id"]:r for r in rows if r["variant"]=="raft_uniform"}
        deltas=[]
        for row in rows:
            if row["variant"] not in ("raft_hard","raft_soft"):
                continue
            delta={k:row[k] for k in ("id","split","group","variant")}
            for key in ("psnr","ssim","lpips","time_ms","hole_fraction"):
                a,b=row[key],baseline[row["id"]][key]
                delta[f"delta_{key}"]=float(a)-float(b) if a!="" and b!="" and np.isfinite([float(a),float(b)]).all() else ""
            deltas.append(delta)
        write_csv(out/"paired_deltas.csv",deltas)
    metadata["status"]="completed"
    write_json(out/"run.json",metadata)


def score(args):
    samples=select_samples(args)
    predictions=Path(args.predictions)
    source=json.loads(Path(args.provenance).read_text())
    if not source.get("model") or not source.get("weights_sha256") or not source.get("code_commit"):
        raise ValueError("Provenance must identify model, weights_sha256 and code_commit")
    if source.get("status", "completed") != "completed":
        raise ValueError("External model export is not marked completed")
    if source.get("manifest_sha256", file_sha256(args.manifest)) != file_sha256(args.manifest):
        raise ValueError("External export and evaluation manifests differ")
    timings={}
    if args.timings:
        with Path(args.timings).open(newline="") as f:
            for row in csv.DictReader(f):
                if row["id"] in timings:
                    raise ValueError("Duplicate timing id")
                value=float(row["time_ms"])
                if not np.isfinite(value) or value < 0:
                    raise ValueError("Invalid timing")
                timings[row["id"]]=value
        if set(timings)!={s.id for s in samples}:
            raise ValueError("Timing ids differ from evaluation ids")
    out=new_output(args.output)
    metadata={**environment(),"status":"running","arguments":vars(args),
        "manifest_sha256":file_sha256(args.manifest),"external_model":source}
    write_json(out/"run.json",metadata)
    lpips=LPIPSMetric(args.device) if args.lpips else None
    rows=[]
    for s in samples:
        npy=predictions/f"{s.id}.npy"
        png=predictions/f"{s.id}.png"
        if npy.is_file() and png.is_file():
            raise ValueError(f"Ambiguous prediction format for {s.id}")
        pred=np.load(npy,allow_pickle=False) if npy.is_file() else read_image(png)
        gt=read_image(s.target)
        metric=quality(pred,gt)
        rows.append({"id":s.id,"split":s.split,"group":s.group,"variant":source["model"],
            "height":gt.shape[0],"width":gt.shape[1],**metric,
            "lpips":lpips(pred,gt) if lpips else "","time_ms":timings.get(s.id,""),"hole_fraction":""})
        write_csv(out/"per_image.csv",rows)
    write_csv(out/"summary.csv",summarize(rows))
    metadata["status"]="completed"
    metadata["png_quantization"] = any((predictions/f"{s.id}.png").is_file() for s in samples)
    write_json(out/"run.json",metadata)


def demo(args):
    out=new_output(args.output)
    h,w=64,96
    a=np.zeros((h,w,3),np.float32)
    b=a.copy()
    a[16:48,20:40]=[0.2,0.6,0.9]
    b[16:48,28:48]=[0.2,0.6,0.9]
    f=np.zeros((h,w,2),np.float32); f[...,0]=8
    g=-f
    weights0=consistency(f,g)["hard"]
    weights1=consistency(g,f)["hard"]
    pred,holes=interpolate(a,b,f,g,weights0,weights1)
    for name,im in (("input0",a),("input1",b),("average",(a+b)/2),("interpolated",pred)):
        save_image(out/f"{name}.png",im)
    write_json(out/"about.json",{"purpose":"Synthetic geometry smoke check with supplied translation flow; not a RAFT or SNU-FILM result", "hole_fraction":float(holes.mean())})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    m=sub.add_parser("make-snu-manifest")
    for name in ("data-root","list-root","output"):
        m.add_argument(f"--{name}",required=True)
    e=sub.add_parser("evaluate")
    e.add_argument("--method",choices=("average","raft"),default="raft")
    e.add_argument("--weights",default="C_T_V2")
    e.add_argument("--iterations",type=int,default=20)
    e.add_argument("--alpha",type=float,default=0.01)
    e.add_argument("--beta",type=float,default=0.5)
    e.add_argument("--warmup",type=int,default=2)
    e.add_argument("--save-ids",help="Text file of preselected sample ids; otherwise no images are saved")
    s=sub.add_parser("score")
    s.add_argument("--predictions",required=True)
    s.add_argument("--provenance",required=True)
    s.add_argument("--timings")
    for p in (e,s):
        p.add_argument("--manifest",required=True)
        p.add_argument("--output",required=True)
        p.add_argument("--split",help="Comma-separated manifest split names")
        p.add_argument("--limit",type=int)
        p.add_argument("--device",default="cpu")
        p.add_argument("--lpips",action="store_true")
    d=sub.add_parser("demo")
    d.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    if args.command=="make-snu-manifest":
        print(f"Validated {len(make_snu_manifest(args.data_root,args.list_root,args.output))} triplets")
    else:
        {"evaluate":evaluate,"score":score,"demo":demo}[args.command](args)


if __name__=="__main__":
    main()
