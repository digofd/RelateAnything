"""Free-text evaluation cases where plausible changes would alter Table 5."""
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
from PIL import Image
import pytest

from benchmark.vlm.common import load_rows, validate_rows
from benchmark.vlm.protocol import annotate_image, build_prompt, parse_relations
from benchmark.vlm.run_vlm_pack import select_indices
from benchmark.vlm.score_vlm import score_rows

ROOT = Path(__file__).resolve().parents[1]
VLM = ROOT / "benchmark/vlm"


@pytest.fixture
def scored_pack(tmp_path):
    predicates = ["holding", "beside", "on"]
    meta = np.array([[11, 64, 64, 0, 3, 0, 2], [12, 64, 64, 3, 3, 2, 3],
                     [13, 64, 64, 6, 3, 5, 1]])
    rels = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0],
                     [0, 1, 1, 0], [1, 2, 2, 0], [0, 1, 2, 0]])
    mapping = {"unmatched": [], "holding": [0], "beside": [1], "broad phrase": [0, 1], "on": [2]}
    rows = [
        {"image_id": 11, "n_regions": 3, "relations": [
            [0, 1, "unmatched"], [0, 1, "holding"], [0, 1, "beside"], [1, 0, "beside"]]},
        {"image_id": 12, "n_regions": 3, "relations": [[0, 1, "broad phrase"], [1, 2, "on"]]},
        {"image_id": 13, "n_regions": 3, "relations": []},
    ]
    (tmp_path / "meta.json").write_text(json.dumps({"predicates": predicates}))
    np.save(tmp_path / "img_meta.npy", meta)
    np.save(tmp_path / "rels.npy", rels)
    (tmp_path / "map.json").write_text(json.dumps({"predicates": predicates, "tau": 0.6, "map": mapping}))
    (tmp_path / "pred.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path, meta, rels, predicates, mapping, rows


def test_first_mappable_phrase_reverse_pair_and_empty_image(scored_pack):
    _, meta, rels, predicates, mapping, rows = scored_pack
    result = score_rows(rows, meta, rels, predicates, mapping, [1, 2])
    m = result["metrics"]
    # Unmatched text does not reserve (0,1); the second accepted phrase does
    # not get another guess. The reverse pair keeps its own slot.
    assert m["R@1"] == pytest.approx(7 / 18)
    assert m["R@2"] == pytest.approx(2 / 3)  # the empty image remains in the mean
    assert m["mR@1"] == pytest.approx(0.5)
    assert m["mR@2"] == pytest.approx(5 / 6)
    assert m["pair_coverage"] == pytest.approx(4 / 5)
    assert m["mean_preds_per_img"] == pytest.approx(4 / 3)
    assert m["n_unmapped_strings"] == 1
    # Original diagnostic: one phrase can credit two GT annotations. This is
    # retained for parity, not interpreted as a bounded precision estimate.
    assert m["precision"] == 1.25


def test_generation_order_changes_topk_without_similarity_sorting(scored_pack):
    _, meta, rels, predicates, mapping, rows = scored_pack
    rows[0]["relations"] = [[0, 1, "beside"], [0, 1, "holding"], [1, 0, "beside"]]
    result = score_rows(rows[:1], meta[:1], rels, predicates, mapping, [1, 2])
    assert result["metrics"]["R@1"] == 0.0
    assert result["metrics"]["R@2"] == 0.5


def test_match_set_uses_one_topk_slot(scored_pack):
    _, meta, rels, predicates, mapping, rows = scored_pack
    result = score_rows(rows[1:2], meta[1:2], rels, predicates, mapping, [1, 2])
    assert result["metrics"]["R@1"] == pytest.approx(2 / 3)
    assert result["metrics"]["R@2"] == 1.0


def test_parser_recovers_complete_prefix_and_keeps_distinct_phrases():
    text = '{"relations": [{"s":1,"o":2,"p":"holding"}, {"s":2'
    assert parse_relations(text, 2) == [[0, 1, "holding"]]
    text = json.dumps({"relations": [
        {"s": 1, "o": 2, "p": " holding "}, {"s": 1, "o": 2, "p": "HOLDING"},
        {"subject": 1, "object": 2, "predicate": "beside"},
        {"s": 1, "o": 1, "p": "on"}, {"s": 3, "o": 1, "p": "on"},
        {"s": 2, "o": 1, "p": ""}, {"s": "bad", "o": 1, "p": "on"},
    ]})
    assert parse_relations("```json\n" + text + "\n```", 2) == [[0, 1, "holding"], [0, 1, "beside"]]


def test_rendering_and_prompt_withhold_object_labels_and_vocabulary():
    image = Image.new("RGB", (96, 96), "white")
    one = annotate_image(image, [{"bbox": [10, 10, 70, 70], "label": "secret category"}])
    two = annotate_image(image, [{"bbox": [10, 10, 70, 70], "label": "different category"}])
    assert one.tobytes() == two.tobytes()
    assert one.tobytes() != image.tobytes()
    assert "Use ONLY relationships" not in build_prompt(2)


def test_missing_duplicate_and_foreign_images_fail(scored_pack):
    path, meta, _, _, _, rows = scored_pack
    with pytest.raises(ValueError, match="no prediction record"):
        validate_rows(rows[:1], meta)
    validate_rows(rows[:1], meta, allow_subset=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_rows([rows[0], rows[0]], meta, allow_subset=True)
    bad = dict(rows[0], image_id=999)
    with pytest.raises(ValueError, match="not in the pack"):
        validate_rows([bad], meta, allow_subset=True)
    (path / "duplicate.jsonl").write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_rows([str(path / "*.jsonl")])
    with pytest.raises(ValueError, match="No prediction files"):
        load_rows([str(path / "missing*.jsonl")])


def test_subsample_is_fixed_before_sharding():
    whole = select_indices(100, 17, 0, 0, 1, 0)
    shards = [select_indices(100, 17, 0, i, 4, 0) for i in range(4)]
    assert sorted(np.concatenate(shards)) == list(whole)
    assert sum(len(s) for s in shards) == len(set(np.concatenate(shards)))
    with pytest.raises(ValueError, match="shard"):
        select_indices(10, 0, 0, 2, 2, 0)


def test_cli_evaluates_own_predictions_and_checks_map_order(scored_pack):
    path, meta, rels, predicates, mapping, rows = scored_pack
    expected = score_rows(rows, meta, rels, predicates, mapping)
    args = [sys.executable, str(VLM / "score_vlm.py"), "--pack", str(path), "--pred",
            str(path / "pred.jsonl"), "--map", str(path / "map.json"), "--out", str(path / "result.json")]
    result = subprocess.run(args, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    scored = json.loads((path / "result.json").read_text())
    assert scored["metrics"] == expected["metrics"]
    assert scored["per_class_recall"] == expected["per_class_recall"]
    (path / "map.json").write_text(json.dumps({"predicates": list(reversed(predicates)),
                                               "tau": 0.6, "map": mapping}))
    result = subprocess.run(args, text=True, capture_output=True)
    assert result.returncode != 0 and "predicate order" in result.stderr


def test_generation_round_trip_without_model_download(scored_pack, monkeypatch):
    from benchmark.vlm import run_vlm_pack

    path, _, _, predicates, mapping, _ = scored_pack
    np.save(path / "boxes.npy", np.tile([0.5, 0.5, 0.5, 0.5], (9, 1)))
    files = [f"{i}.jpg" for i in range(3)]
    (path / "file_names.json").write_text(json.dumps(files))
    for name in files:
        Image.new("RGB", (64, 64), "white").save(path / name)
    captured = {}

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"][1]["text"]

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def generate(self, requests, sampling):
            assert sampling.temperature == 0.0 and sampling.max_tokens == 2048
            assert "Use ONLY relationships" not in requests[0]["prompt"]
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(
                text='{"relations":[{"s":1,"o":2,"p":"holding"}]}', token_ids=[1, 2])])
                for _ in requests]

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: types.SimpleNamespace(_commit_hash="commit")),
        AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: Processor())))
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM, SamplingParams=types.SimpleNamespace))
    out = path / "generated.jsonl"
    monkeypatch.setattr(sys, "argv", ["run_vlm_pack.py", "--pack", str(path), "--image_root", str(path),
                                      "--out", str(out)])
    run_vlm_pack.main()
    _, rows = load_rows([str(out)])
    assert len(rows) == 3 and rows[0]["relations"] == [[0, 1, "holding"]]
    manifest = json.loads(Path(str(out) + ".meta.json").read_text())
    assert manifest["complete"] and manifest["resolved_revision"] == "commit"
    assert captured["revision"] == "commit" and captured["dtype"] == "bfloat16"
    with pytest.raises(FileExistsError):
        run_vlm_pack.main()
