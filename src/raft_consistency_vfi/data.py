"""Strict manifest handling; ground truth is passed only to scoring code."""

import csv
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Sample:
    id: str
    input0: Path
    target: Path
    input1: Path
    split: str
    group: str


def read_manifest(path):
    path = Path(path).resolve()
    samples, ids, triplets = [], set(), set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "input0", "target", "input1", "split", "group"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns {sorted(required)}")
        for row in reader:
            sid = row["id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", sid) or sid in ids:
                raise ValueError(f"Invalid or repeated sample id: {sid}")
            if not row["split"] or not row["group"]:
                raise ValueError(f"Missing split or sequence group: {sid}")
            files = tuple((path.parent / row[k]).resolve() for k in ("input0", "target", "input1"))
            for file in files:
                if not file.is_file():
                    raise FileNotFoundError(file)
            if len(set(files)) != 3:
                raise ValueError(f"Inputs and target must be distinct: {sid}")
            key = (row["split"], files)
            if key in triplets:
                raise ValueError(f"Repeated triplet within split: {sid}")
            triplets.add(key)
            ids.add(sid)
            samples.append(Sample(sid, *files, row["split"], row["group"]))
    if not samples:
        raise ValueError("Empty manifest")
    return samples


def read_image(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32)/255.0


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(image, 0, 1)*255).astype(np.uint8)).save(path)


def load_triplet(sample):
    images = tuple(read_image(p) for p in (sample.input0, sample.target, sample.input1))
    if len({a.shape for a in images}) != 1:
        raise ValueError(f"Unequal image dimensions: {sample.id}")
    return images


def make_snu_manifest(data_root, list_root, output):
    """Read all four official test-*.txt lists without resizing or resampling."""
    root, lists, output = Path(data_root).resolve(), Path(list_root), Path(output)
    records = []
    for split in ("easy", "medium", "hard", "extreme"):
        list_path = lists/f"test-{split}.txt"
        for number, line in enumerate(list_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            raw = line.split()
            if len(raw) != 3:
                raise ValueError(f"Expected input0 target input1 at {list_path}:{number}")
            resolved = []
            for item in raw:
                p = Path(item)
                candidates = {root/p}
                # The official lists often contain data/SNU-FILM/... prefixes.
                for i, part in enumerate(p.parts):
                    if part == "SNU-FILM":
                        candidates.add(root/Path(*p.parts[i+1:]))
                found = {p.resolve() for p in candidates if p.is_file()}
                if len(found) != 1:
                    raise ValueError(f"Resolve exactly one file for {item}; got {len(found)}")
                resolved.append(found.pop())
            for p in resolved:
                if not p.is_relative_to(root):
                    raise ValueError(f"Path lies outside data root: {p}")
            group = resolved[0].parent.relative_to(root).as_posix()
            if len({p.parent for p in resolved}) != 1:
                raise ValueError(f"Triplet crosses sequence directories: {raw}")
            records.append(dict(zip(("id","input0","target","input1","split","group"),
                (f"{split}_{number:05d}", *map(str,resolved), split, group))))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id","input0","target","input1","split","group"])
        writer.writeheader()
        writer.writerows(records)
    return read_manifest(output)
