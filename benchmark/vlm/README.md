# Qwen free-text evaluation (Table 5)

Code for generating and evaluating the Qwen3-VL ground-truth-region,
vocabulary-withheld baselines on PSG test. Run all commands from the repository
root. Predictions, predicate maps and scores are generated locally.

Get the PSG test pack, source images and `text_student_v2_512/student.pt` from
the existing [data setup](../../docs/installation.md#get-the-data).

## Run your predictions and evaluation

Use a separate vLLM environment from the main relation-model environment:

```bash
python3.12 -m venv .venv-vlm
source .venv-vlm/bin/activate
pip install -r benchmark/vlm/requirements.txt

# 1. Generate predictions with the original prompt and numbered boxes.
python benchmark/vlm/run_vlm_pack.py \
  --pack runs/ovsgg_bench/packs/psg/test \
  --image_root /path/to/PSG/test/images \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --out runs/vlm/psg_open_s0.jsonl

# 2. Build predicate match sets for the phrases your model generated.
python benchmark/vlm/build_a3_map.py \
  --pack runs/ovsgg_bench/packs/psg/test \
  --pred 'runs/vlm/psg_open_s*.jsonl' \
  --student runs/ovsgg_bench/text_student_v2_512/student.pt \
  --tau 0.60 --device cpu \
  --out runs/vlm/qwen_a3_map.json

# 3. Evaluate your predictions with that map.
python benchmark/vlm/score_vlm.py \
  --pack runs/ovsgg_bench/packs/psg/test \
  --pred 'runs/vlm/psg_open_s*.jsonl' \
  --map runs/vlm/qwen_a3_map.json \
  --out runs/vlm/qwen_scores.json
```

`--image_root` contains the files listed in the pack's `file_names.json`.
For the 32B baseline, use `Qwen/Qwen3-VL-32B-Instruct` and `--tp 2` when needed
for GPU memory. To shard, set `--shard S --num_shards N` with distinct output
paths and otherwise identical arguments. The evaluator requires every pack
image exactly once; use `--allow_subset` only for intentional smoke tests.

The runner defaults to the original launcher's settings: numbered boxes
without class labels, describe-then-relate prompting, at most 60 regions,
bf16, temperature 0, seed 0, 2,048 output tokens, 16,384 context tokens and
128 requests per chunk. `protocol.py` contains the original prompt and parser.
Install DejaVu Sans Bold (`fonts-dejavu-core` on Debian/Ubuntu) for the original
rendering; otherwise Pillow's default font is used.

The requirements pin the repository's recorded vLLM version. Historical model
commit IDs and per-run lockfiles were not recovered. Use `--revision` to pin
new runs; their metadata sidecars record the resolved revision and versions.
Map building uses the existing student encoder and its cached or colocated
CLIP tokenizer. Scoring alone requires only NumPy and the pack's `meta.json`,
`img_meta.npy` and `rels.npy`, without images or model weights.

## Scoring rules

- Parse JSON in generation order, convert 1-based region IDs to 0-based indices,
  discard invalid IDs, self-relations, empty phrases and duplicate triplets,
  and recover complete relation objects from truncated output when possible.
- Normalize text by lowercasing, replacing underscores and collapsing whitespace.
  A phrase matches each benchmark predicate with identical normalized text or
  student cosine at least **0.60**, using the original three-template ensemble.
  The Table 5 matcher supplies no inverse mask.
- Drop phrases with no match **before** retaining the first surviving phrase
  per ordered pair. The reverse pair is separate. Take the first K retained
  predictions in generation order; similarity does not rank them. A phrase
  matching several predicates occupies one slot.
- `R@K` averages per-image recall; `mR@K` averages pooled per-class recall.
  Empty predictions count as zero on annotated images. GT relations beyond
  the region cap are excluded and counted in the output.

Pair coverage uses all retained predictions after matching. Legacy
`precision`/`F1` diagnostics retain the original arithmetic: a phrase can
credit several GT labels, so `precision` may exceed one. The Table 5 metrics
are R/mR. Synthetic regression tests check the original scoring conventions.
