"""Original Table 5 prompt, rendering and parsing, without datagen dependencies.

The helpers retain the original ordering, duplicate handling and
truncated-JSON recovery. They require no model weights or torch import.
"""
from __future__ import annotations

import json
import re
from PIL import Image

_HEX = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _get_fonts():
    from PIL import ImageFont
    try:
        big   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        big = small = ImageFont.load_default()
    return big, small


def annotate_image(pil: Image.Image, objects: list) -> Image.Image:
    """Original bbox_no_label rendering: colored boxes and 1-based region IDs."""
    from PIL import ImageDraw
    font_big, _ = _get_fonts()
    img = pil.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    for i, obj in enumerate(objects):
        x1, y1, x2, y2 = obj["bbox"]
        color = _HEX[i % len(_HEX)]
        idx = str(i + 1)
        for t in range(3):
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=color)
        draw.rectangle([x1, y1, x2, y2], fill=color + "30")
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bb = draw.textbbox((0, 0), idx, font=font_big)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((cx - tw // 2, cy - th // 2), idx,
                  font=font_big, fill="white", stroke_width=2, stroke_fill="black")
    return img


def _extract_json(text: str) -> dict | None:
    for pat in [r"```json\s*\n(.*?)\n?\s*```", r"```\s*\n(\{.*?\})\s*```"]:
        m = re.search(pat, text, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                candidate = m.group(1)
                break
    else:
        m = re.search(r"(\{.*\})", text, re.S)
        candidate = m.group(1) if m else text
        if m:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    rel_pat = re.compile(
        r'\{\s*"subject_id"\s*:\s*(\d+).*?"predicate"\s*:\s*"([^"]+)".*?"object_id"\s*:\s*(\d+).*?\}',
        re.S,
    )
    rels = []
    for rm in rel_pat.finditer(candidate):
        try:
            rels.append(json.loads(candidate[rm.start(): rm.end()]))
        except json.JSONDecodeError:
            rels.append({
                "subject_id":    int(rm.group(1)), "subject_label": "",
                "predicate":     rm.group(2),
                "object_id":     int(rm.group(3)), "object_label": "",
            })
    if rels:
        return {"scene_description": "", "relations": rels, "_truncated": True}
    return None


PROMPT_DESCRIBE = """This image contains {n} numbered regions.
{regions}
First identify what each region is, then list the relationships between them.

Answer with ONLY a JSON object, no other text:
{{"objects": {{"1": "<what region 1 is>", "2": "<what region 2 is>", ...}},
 "relations": [{{"s": <subject region number>, "o": <object region number>, "p": "<relationship>"}}, ...]}}

Describe every region. Then report every relationship you can see between pairs
of regions, in the direction subject -> object, using short phrases such as
"on", "holding", "next to", "wearing", "behind", "riding", "part of"."""


REGIONS_SOM = ("Each region is marked on the image with a coloured overlay and "
               "its number.")


def _objects_of(text: str) -> list:
    """Region descriptions, when the prompt asked for them (grounding evidence)."""
    obj = _extract_json(text)
    if isinstance(obj, dict) and isinstance(obj.get("objects"), dict):
        return [v for _, v in sorted(obj["objects"].items(),
                                     key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)]
    return []


def parse_relations(text: str, n_regions: int) -> list[list]:
    """JSON array of {s,o,p} -> 0-indexed triplets, tolerant of stray prose."""
    obj = _extract_json(text)
    if obj is None:
        m = re.search(r"\[.*\]", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:                                  # noqa: BLE001
                obj = None
    if obj is None:
        # Hitting max_new_tokens leaves a VALID prefix inside a broken array.
        # Discarding it would penalise exactly the modes that say the most, so
        # salvage every complete {...} object instead.
        obj = []
        for frag in re.finditer(r"\{[^{}]*\}", text):
            try:
                obj.append(json.loads(frag.group(0)))
            except Exception:                                  # noqa: BLE001
                continue
    if isinstance(obj, dict):
        obj = obj.get("relations") or obj.get("triplets") or []
    out, seen = [], set()
    for it in obj if isinstance(obj, list) else []:
        if not isinstance(it, dict):
            continue
        try:
            s = int(it.get("s", it.get("subject", -1)))
            o = int(it.get("o", it.get("object", -1)))
        except (TypeError, ValueError):
            continue
        p = it.get("p", it.get("predicate", it.get("relation", "")))
        if not isinstance(p, str) or not p.strip():
            continue
        s, o = s - 1, o - 1                                    # prompt is 1-based
        if not (0 <= s < n_regions and 0 <= o < n_regions) or s == o:
            continue
        key = (s, o, p.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append([s, o, p.strip()])
    return out


def build_prompt(n_regions: int) -> str:
    """Table 5 prompt with the predicate vocabulary withheld."""
    return PROMPT_DESCRIBE.format(n=n_regions, regions=REGIONS_SOM)
