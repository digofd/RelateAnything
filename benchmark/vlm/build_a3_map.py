"""Build A3 match sets for new generations with the original distilled student.

The Table 5 threshold is 0.60, not the default used in some other A3 tables.
This retains the original template ensemble and exact-or-cosine matching rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__:
    from .common import load_rows, norm, read_meta, sha256
else:
    from common import load_rows, norm, read_meta, sha256

TRAIN_TEMPLATES = ["{p}", "one object is {p} another object", "a photo of something {p} something"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--pred", required=True, nargs="+")
    parser.add_argument("--student", required=True, help="Original text_student_v2_512/student.pt")
    parser.add_argument("--tau", type=float, default=0.60)
    parser.add_argument("--device", default=None, help="e.g. cpu or cuda; default auto-selects")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not -1 <= args.tau <= 1:
        parser.error("tau must be a cosine threshold between -1 and 1")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from relsgg.eval.evaluator import build_cross_match_matrix
    from relsgg.text.student import encode_texts_student

    predicates = [norm(p) for p in read_meta(args.pack)["predicates"]]
    files, rows = load_rows(args.pred)
    strings = sorted({norm(text) for r in rows for _, _, text in r["relations"] if norm(text)})
    if strings:
        predicted = encode_texts_student(strings, args.student, templates=TRAIN_TEMPLATES, device=args.device)
        target = encode_texts_student(predicates, args.student, templates=TRAIN_TEMPLATES, device=args.device)
        matrix = build_cross_match_matrix(strings, predicates, predicted.numpy(), target.numpy(),
                                          tau_eval=args.tau).bool().numpy()
        mapping = {s: np.flatnonzero(matrix[i]).tolist() for i, s in enumerate(strings)}
    else:
        mapping = {}
    result = {
        "tau": args.tau, "map": mapping, "predicates": read_meta(args.pack)["predicates"],
        "templates": TRAIN_TEMPLATES, "student_sha256": sha256(args.student),
        "inputs": {name: sha256(name) for name in files},
        "matching": "normalized_exact_or_student_cosine; no inverse_mask supplied (original Table 5)",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(mapping)} phrases at tau={args.tau} to {out}")


if __name__ == "__main__":
    main()
