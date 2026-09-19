#!/usr/bin/env python3
"""Benchmark the Coral Edge TPU against the CPU on the same network.

Downloads SSD MobileNet-V2 in two forms: one compiled for the Edge TPU, one
not. Running both tells you whether the delegate is actually doing work or
silently falling back to the CPU.

    ./.venv/bin/python benchmark.py
"""
import pathlib
import time
import urllib.request

import numpy as np
import tflite_runtime.interpreter as tflite

MODELS = {
    "tpu": "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
    "cpu": "ssd_mobilenet_v2_coco_quant_postprocess.tflite",
}
BASE = "https://github.com/google-coral/test_data/raw/master"
N = 100
HERE = pathlib.Path(__file__).parent


def ensure_models():
    cache = HERE / "models"
    cache.mkdir(exist_ok=True)
    for name in MODELS.values():
        path = cache / name
        if not path.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(f"{BASE}/{name}", path)
    return {k: str(HERE / "models" / v) for k, v in MODELS.items()}


def bench(interpreter, n=N):
    interpreter.allocate_tensors()
    detail = interpreter.get_input_details()[0]
    rng = np.random.default_rng(0)
    sample = rng.integers(0, 256, size=tuple(detail["shape"]), dtype=np.uint8)

    for _ in range(5):  # warm up
        interpreter.set_tensor(detail["index"], sample)
        interpreter.invoke()

    start = time.perf_counter()
    for _ in range(n):
        interpreter.set_tensor(detail["index"], sample)
        interpreter.invoke()
    per_call = (time.perf_counter() - start) / n
    return detail["shape"], per_call, 1.0 / per_call


def main():
    paths = ensure_models()

    delegate = tflite.load_delegate("libedgetpu.so.1")
    tpu = tflite.Interpreter(
        model_path=paths["tpu"], experimental_delegates=[delegate]
    )
    shape, tpu_ms, tpu_fps = bench(tpu)
    print(f"Edge TPU : {tpu_ms * 1000:8.2f} ms   {tpu_fps:6.1f} FPS")

    cpu = tflite.Interpreter(model_path=paths["cpu"])
    _, cpu_ms, cpu_fps = bench(cpu)
    print(f"CPU      : {cpu_ms * 1000:8.2f} ms   {cpu_fps:6.1f} FPS")

    print(f"\ninput {tuple(shape)} (uint8)")
    print(f"speedup  : {tpu_fps / cpu_fps:.2f}x   ({cpu_ms / tpu_ms:.2f}x lower latency)")


if __name__ == "__main__":
    main()