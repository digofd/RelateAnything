"""Score Table 5 JSONL generations on CPU, preserving the original arithmetic.

Generation order is the only ranking. Predicate matching precedes selection of
the first accepted phrase per ordered pair. A match set occupies one top-K slot.
Empty predictions count as recall zero on images with annotated relations.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__:
    from .common import load_rows, norm, read_meta, sha256, validate_rows
else:
    from common import load_rows, norm, read_meta, sha256, validate_rows


def score_rows(rows: list[dict], img_meta: np.ndarray, rels: np.ndarray,
               predicates: list[str], match_map: dict[str, list[int]],
               topk: list[int] = (20, 50, 100)) -> dict:
    """The Table 5 loop from score_robin.py, with file I/O separated for testing.

    R@K is the mean of per-image recall; mR@K averages pooled per-class
    recall over classes present in the evaluated ground truth.
    """
    if not topk or any(k <= 0 for k in topk) or len(set(topk)) != len(topk):
        raise ValueError("topk must contain distinct positive integers")
    gt_by_img = {}
    for row in img_meta:
        image_id, _, _, _, _, roff, nr = (int(v) for v in row)
        gt_by_img[image_id] = [(int(s), int(o), int(p))
                               for s, o, p, *_ in rels[roff:roff + nr]]
    pmap = {k: v for k, v in match_map.items() if v}
    K = max(topk)
    img_rec = {k: [] for k in topk}
    cls_tp = {k: defaultdict(int) for k in topk}
    cls_gt = defaultdict(int)
    n_pred_tot = n_hit_tot = 0
    n_gt_scored = n_gt_dropped = 0
    pair_cov_num = pair_cov_den = 0
    n_unmapped = 0
    npred_per_img = []

    for r in rows:
        gt = gt_by_img[r["image_id"]]
        n_reg = r["n_regions"]
        gt_in = [(s, o, p) for s, o, p in gt if s < n_reg and o < n_reg]
        n_gt_dropped += len(gt) - len(gt_in)
        if not gt_in:
            continue
        n_gt_scored += len(gt_in)
        pred, seen, spoken_pairs = [], set(), set()
        for s, o, text in r["relations"]:
            if not (0 <= s < n_reg and 0 <= o < n_reg):
                continue
            ent = pmap.get(norm(text))
            if ent is None:
                n_unmapped += 1
                continue
            pids = frozenset(int(x) for x in ent)
            if (s, o) in spoken_pairs:
                continue
            key = (s, o, pids)
            if key in seen:
                continue
            seen.add(key)
            spoken_pairs.add((s, o))
            pred.append(key)
        npred_per_img.append(len(pred))
        n_pred_tot += len(pred)
        gt_pairs = {(s, o) for s, o, _ in gt_in}
        pred_pairs = {(s, o) for s, o, _ in pred}
        pair_cov_num += len(gt_pairs & pred_pairs)
        pair_cov_den += len(gt_pairs)
        for s, o, p in gt_in:
            cls_gt[p] += 1
        hit_any = set()
        for k in topk:
            acc = {}
            for ps, po, pids in pred[:k]:
                acc.setdefault((ps, po), set()).update(pids)
            hits = 0
            for s, o, p in gt_in:
                if p in acc.get((s, o), ()):
                    hits += 1
                    cls_tp[k][p] += 1
                    if k == K:
                        hit_any.add((s, o, p))
            img_rec[k].append(hits / len(gt_in))
        n_hit_tot += len(hit_any)

    metrics = {}
    for k in topk:
        metrics[f"R@{k}"] = float(np.mean(img_rec[k])) if img_rec[k] else 0.0
        per = [cls_tp[k][p] / cls_gt[p] for p in sorted(cls_gt)]
        metrics[f"mR@{k}"] = float(np.mean(per)) if per else 0.0
    prec = n_hit_tot / max(n_pred_tot, 1)
    rec_micro = n_hit_tot / max(n_gt_scored, 1)
    metrics.update(
        precision=prec, recall_micro=rec_micro,
        F1=2 * prec * rec_micro / max(prec + rec_micro, 1e-9),
        pair_coverage=pair_cov_num / max(pair_cov_den, 1),
        mean_preds_per_img=float(np.mean(npred_per_img)) if npred_per_img else 0.0,
        n_images=len(rows), n_gt_scored=n_gt_scored,
        n_gt_dropped_beyond_regions=n_gt_dropped, n_unmapped_strings=n_unmapped,
        n_classes_in_gt=len(cls_gt),
    )
    return {
        "metrics": metrics,
        "per_class_recall": {
            str(k): {predicates[p]: cls_tp[k][p] / cls_gt[p] for p in sorted(cls_gt)}
            for k in topk
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--pred", required=True, nargs="+", help="JSONL paths or quoted globs")
    parser.add_argument("--map", required=True, help="A3 match-set map built from your predictions by build_a3_map.py")
    parser.add_argument("--topk", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--allow_subset", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    pack = Path(args.pack)
    predicates = read_meta(pack)["predicates"]
    img_meta = np.load(pack / "img_meta.npy", allow_pickle=False)
    rels = np.load(pack / "rels.npy", mmap_mode="r", allow_pickle=False)
    files, rows = load_rows(args.pred)
    validate_rows(rows, img_meta, args.allow_subset)
    mapping = json.loads(Path(args.map).read_text())
    if mapping["predicates"] != predicates:
        raise ValueError("Map predicate order differs from the pack; rebuild the map")
    match_map = mapping["map"]
    for phrase, ids in match_map.items():
        if not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < len(predicates) for i in ids):
            raise ValueError(f"Invalid match set for {phrase!r}")
    unknown = {norm(t) for r in rows for _, _, t in r["relations"]} - set(match_map)
    if unknown:
        raise ValueError(f"{len(unknown)} strings are absent from the map (e.g. {sorted(unknown)[:3]}); "
                         "run build_a3_map.py on these generations first")
    result = score_rows(rows, img_meta, rels, predicates, match_map, args.topk)
    result.update(
        pack=str(pack), map=args.map, min_sim=0.0, graph_constraint=True,
        exact_only=False, match_sets=True, tau=mapping["tau"],
        subset=len(rows) != len(img_meta), ranking="generation_order_after_matching",
        inputs={name: sha256(name) for name in [*files, args.map, str(pack / "meta.json"),
                                               str(pack / "img_meta.npy"), str(pack / "rels.npy")]},
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
