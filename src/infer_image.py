#!/usr/bin/env python3
"""Run object detection on an image with the Edge TPU and save an annotated copy.

    ./.venv/bin/python src/infer_image.py results/cat.bmp

Unlike benchmark.py, this shows something you can actually look at, which makes
it obvious whether the TPU is detecting anything or quietly falling back to the
CPU.
"""
import argparse
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tflite_runtime.interpreter as tflite

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
DEFAULT_LABELS = ROOT / "models/coco_labels.txt"
THRESHOLD = 0.4

# Distinct-ish colours so overlapping boxes stay readable.
COLOURS = [
    (255, 59, 48), (52, 199, 89), (0, 122, 255), (255, 149, 0),
    (175, 82, 222), (255, 204, 0), (0, 199, 190), (255, 45, 85),
]


def load_labels(path):
    """Handle both 'index name' and bare 'name per line' label maps."""
    labels = {}
    for i, line in enumerate(pathlib.Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        head, _, rest = line.partition(" ")
        if head.isdigit():
            labels[int(head)] = rest.strip()
        else:
            labels[i] = line.strip()
    return labels


def detect(interpreter, image, threshold=THRESHOLD):
    _, height, width, _ = interpreter.get_input_details()[0]["shape"]
    resized = image.resize((width, height))
    tensor = np.expand_dims(np.asarray(resized, dtype=np.uint8), axis=0)

    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], tensor)
    interpreter.invoke()

    boxes = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]
    classes = interpreter.get_tensor(interpreter.get_output_details()[1]["index"])[0]
    scores = interpreter.get_tensor(interpreter.get_output_details()[2]["index"])[0]
    count = int(interpreter.get_tensor(interpreter.get_output_details()[3]["index"])[0])

    results = []
    for i in range(count):
        if scores[i] < threshold:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        results.append(
            {
                "label": int(classes[i]),
                "score": float(scores[i]),
                "box": (
                    xmin * image.width,
                    ymin * image.height,
                    xmax * image.width,
                    ymax * image.height,
                ),
            }
        )
    return results


def annotate(image, results, labels):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for det in results:
        colour = COLOURS[det["label"] % len(COLOURS)]
        draw.rectangle(det["box"], outline=colour, width=3)
        text = f"{labels.get(det['label'], det['label'])} {det['score']:.0%}"
        tx, ty = det["box"][0], max(0, det["box"][1] - 22)
        bbox = draw.textbbox((tx, ty), text, font=font)
        draw.rectangle(bbox, fill=colour)
        draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to an input image")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--out", help="output path (default: <image>_detected.png)")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    interpreter = tflite.Interpreter(
        model_path=args.model,
        experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")],
    )
    interpreter.allocate_tensors()

    image = Image.open(args.image).convert("RGB")
    results = detect(interpreter, image)
    annotate(image, results, labels)

    out = pathlib.Path(args.out) if args.out else pathlib.Path(args.image).with_name(
        pathlib.Path(args.image).stem + "_detected.png"
    )
    image.save(out)
    print(f"{len(results)} detection(s) at >= {THRESHOLD:.0%} -> {out}")
    for det in sorted(results, key=lambda d: -d["score"]):
        print(f"  {labels.get(det['label'], det['label']):12} {det['score']:.0%}")


if __name__ == "__main__":
    main()