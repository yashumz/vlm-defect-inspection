"""Run the fine-tuned Qwen2-VL-2B inspector on one image.

Heavy libraries (torch, unsloth) are imported only when an Inspector is created,
so the API and its tests can be imported on a laptop without a GPU.
"""
from __future__ import annotations

import json
import math

from PIL import Image

IMG_SIZE = 448  # must match training

DEFECT_TYPES = {
    "bottle": ["broken_large", "broken_small", "contamination"],
    "cable": ["bent_wire", "cable_swap", "combined", "cut_inner_insulation", "cut_outer_insulation",
              "missing_cable", "missing_wire", "poke_insulation"],
    "capsule": ["crack", "faulty_imprint", "poke", "scratch", "squeeze"],
    "carpet": ["color", "cut", "hole", "metal_contamination", "thread"],
    "grid": ["bent", "broken", "glue", "metal_contamination", "thread"],
    "hazelnut": ["crack", "cut", "hole", "print"],
    "leather": ["color", "cut", "fold", "glue", "poke"],
    "metal_nut": ["bent", "color", "flip", "scratch"],
    "pill": ["color", "combined", "contamination", "crack", "faulty_imprint", "pill_type", "scratch"],
    "screw": ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"],
    "tile": ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
    "toothbrush": ["defective"],
    "transistor": ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
    "wood": ["color", "combined", "hole", "liquid", "scratch"],
    "zipper": ["broken_teeth", "combined", "fabric_border", "fabric_interior", "rough",
               "split_teeth", "squeezed_teeth"],
}
LOCATIONS = ["top-left", "top", "top-right", "left", "center", "right",
             "bottom-left", "bottom", "bottom-right", "none"]


def make_prompt(category: str) -> str:
    name = category.replace("_", " ")
    types = ", ".join(sorted(DEFECT_TYPES[category]))
    return (f"You are inspecting a {name} on a factory line. Is it defective? "
            f"Possible defect types: {types}. "
            'Answer only with JSON: {"defective": true/false, "defect_type": "<type or none>", '
            '"location": "<top-left|top|top-right|left|center|right|bottom-left|bottom|bottom-right|none>"}')


def prepare_image(image: Image.Image) -> Image.Image:
    return image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)


def parse_json(text: str):
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        return json.loads(text[start:end])
    except ValueError:
        return None


class Inspector:
    """Loads base model + LoRA adapters once; call .inspect(image, category) per request."""

    MODES = ("4bit", "8bit", "16bit")

    def __init__(self, adapter_dir: str, mode: str = "4bit"):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        import torch
        from unsloth import FastVisionModel

        self.torch = torch
        kwargs = {"4bit": dict(load_in_4bit=True),
                  "8bit": dict(load_in_4bit=False, load_in_8bit=True),
                  "16bit": dict(load_in_4bit=False, dtype=torch.float16)}[mode]
        self.mode = mode
        self.model, self.processor = FastVisionModel.from_pretrained(adapter_dir, **kwargs)
        FastVisionModel.for_inference(self.model)
        self.tok = getattr(self.processor, "tokenizer", self.processor)
        self.true_id = self.tok.encode(" true", add_special_tokens=False)[0]
        self.false_id = self.tok.encode(" false", add_special_tokens=False)[0]

    def _inputs(self, image, category, prefix=""):
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": make_prompt(category)}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True) + prefix
        return self.processor(prepare_image(image), text, add_special_tokens=False,
                              return_tensors="pt").to("cuda")

    def _prob_true(self, logits):
        pair = logits[[self.true_id, self.false_id]].float()
        return self.torch.softmax(pair, 0)[0].item()

    def inspect(self, image: Image.Image, category: str) -> dict:
        if category not in DEFECT_TYPES:
            raise ValueError(f"unknown category: {category}")
        torch = self.torch
        with torch.no_grad():
            inputs = self._inputs(image, category)
            out = self.model.generate(**inputs, max_new_tokens=48, do_sample=False, use_cache=True,
                                      return_dict_in_generate=True, output_scores=True)
            gen = out.sequences[0, inputs["input_ids"].shape[1]:]
            score = float("nan")
            if out.scores is not None:
                for step, tid in enumerate(gen.tolist()):
                    if tid in (self.true_id, self.false_id):
                        score = self._prob_true(out.scores[step][0])
                        break
            if math.isnan(score):  # backup: read the probability directly
                inputs = self._inputs(image, category, prefix='{"defective":')
                score = self._prob_true(self.model(**inputs).logits[0, -1])
        raw = self.tok.decode(gen, skip_special_tokens=True).strip()
        pred = parse_json(raw) or {}
        return {"defective": pred.get("defective"), "defect_type": pred.get("defect_type"),
                "location": pred.get("location"), "score": round(score, 4),
                "valid_json": bool(pred), "raw": raw}
