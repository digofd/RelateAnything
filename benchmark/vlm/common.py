"""Pack and JSONL helpers shared by the portable free-text baseline tools."""
from __future__ import annotations

import ast
import glob
import hashlib
import json
from pathlib import Path

import numpy as np


def norm(text: str) -> str:
    return " ".join(str(text).lower().strip().replace("_", " ").split())


def read_meta(pack: str | Path) -> dict:
    meta = json.loads((Path(pack) / "meta.json").read_text())
    # Some original packs stored a Python repr rather than a JSON list.
    if isinstance(meta["predicates"], str):
        meta["predicates"] = ast.literal_eval(meta["predicates"])
    return meta


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(patterns: list[str]) -> tuple[list[str], list[dict]]:
    files = sorted({name for pattern in patterns for name in glob.glob(pattern)})
    if not files:
        raise ValueError(f"No prediction files matched {patterns}")
    rows, seen = [], set()
    for name in files:
        with open(name) as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                image_id = int(row["image_id"])
                if image_id in seen:
                    raise ValueError(f"Duplicate image_id {image_id} in {name}:{line_no}")
                if row.get("error"):
                    raise ValueError(f"Generation error for image {image_id}: {row['error']}")
                if not isinstance(row.get("relations"), list):
                    raise ValueError(f"Image {image_id}: relations must be a list")
                row["image_id"] = image_id
                rows.append(row)
                seen.add(image_id)
    if not rows:
        raise ValueError("Prediction files contain no image records")
    return files, rows


def validate_rows(rows: list[dict], img_meta: np.ndarray, allow_subset: bool = False) -> None:
    by_id = {int(row[0]): row for row in img_meta}
    if len(by_id) != len(img_meta):
        raise ValueError("Pack contains duplicate image IDs")
    seen = set()
    for row in rows:
        image_id = row["image_id"]
        if image_id not in by_id:
            raise ValueError(f"Image {image_id} is not in the pack")
        if image_id in seen:
            raise ValueError(f"Duplicate image_id {image_id}")
        seen.add(image_id)
        n = row["n_regions"]
        if not isinstance(n, int) or not 0 <= n <= int(by_id[image_id][4]):
            raise ValueError(f"Image {image_id}: invalid n_regions {n}")
        if "pack_row" in row and (
            not 0 <= row["pack_row"] < len(img_meta)
            or int(img_meta[row["pack_row"]][0]) != image_id
        ):
            raise ValueError(f"Image {image_id}: pack_row does not match this pack")
        for rel in row["relations"]:
            if not isinstance(rel, list) or len(rel) != 3:
                raise ValueError(f"Image {image_id}: expected [subject, object, predicate]")
            s, o, text = rel
            if not isinstance(s, int) or not isinstance(o, int) or not isinstance(text, str):
                raise ValueError(f"Image {image_id}: invalid relation {rel!r}")
    missing = set(by_id) - seen
    if missing and not allow_subset:
        raise ValueError(
            f"{len(missing)} pack images have no prediction record; incomplete shards? "
            "Use --allow_subset only for an intentional subset, not a Table 5 result."
        )


def cxcywh_to_xyxy_abs(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    cx, cy = boxes[:, 0] * width, boxes[:, 1] * height
    bw, bh = boxes[:, 2] * width, boxes[:, 3] * height
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
