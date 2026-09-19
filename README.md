# coral-edge-tpu-setup

Getting a Google Coral USB Accelerator working on Arch Linux, with the specific version combination that actually runs.

## Why this exists

I had a Coral USB Accelerator in a drawer and a ThinkPad on my desk. Plug it in, install the runtime, run a model. That's supposed to be a two-minute job.

It ate an evening.

The docs assume Debian. I run Arch. There's an AUR package for the runtime, but it falls over building against the current flatbuffers, and that build error isn't even about the USB stick. It's about the M.2 card. Nobody tells you that up front, so you follow it for an hour before working out it can't affect you.

The Python side is a different kind of annoying. `pip install pycoral` installs the wrong thing. PyCoral shares its name with a reef mapping library, and pip picks that one. No error. The real PyCoral wheels stop at Python 3.9. On top of that, the LiteRT/tflite-runtime version has to match libedgetpu, or loading the delegate segfaults with a trace that points into TensorFlow instead of telling you the versions are mismatched.

Six combinations later, one worked.

## What actually works

| Piece | Version | Where it comes from |
| --- | --- | --- |
| Python | 3.9 | any venv |
| tflite-runtime | 2.5.0.post1 | Google's Coral apt repo, as a `.deb` |
| pycoral | 2.0.0 | same repo, optional |
| libedgetpu | 16.0 (std) | same repo |
| numpy | <2 | pip |

The key detail is that Coral's own repo pairs `python3-tflite-runtime` 2.5.0 with libedgetpu 16.0. Newer runtimes do not work. More on that below.

If you're on the USB Accelerator, you do not need `gasket-dkms`. That's for the M.2 and Mini PCIe cards. USB also never creates a `/dev/apex*` node, so don't go looking for one when things fail.

## Install

```bash
./install.sh
```

It downloads Google's `.deb` files, extracts the runtime library to `/usr/lib`, drops a udev rule in place, and builds a Python 3.9 environment with the matching runtime. Two of those steps need sudo.

Then plug the Accelerator in and run the benchmark.

## Benchmark

```bash
./.venv/bin/python benchmark.py
```

It runs the same network twice: once compiled for the Edge TPU, once not. The difference tells you whether the delegate is really doing anything. If the TPU numbers look like the CPU numbers, you've silently fallen back to software.

## Numbers

On a ThinkPad T480, over a USB 2.0 cable, SSD MobileNet-V2 int8 at 300x300:

| Device | Latency | Throughput |
| --- | --- | --- |
| Edge TPU | ~28.6 ms | ~35 FPS |
| CPU (XNNPACK) | 75.2 ms | 13.3 FPS |

Roughly 2.7x. That's a floor, not a ceiling. The link was USB 2.0 (480 Mbit), and the Coral is a USB 3 device, so a USB 3 port and cable should push both the speed and the stability up. The measured runs ranged from 28.0 to 31.2 ms, and the device reset a few times during testing, which is what a USB 2 link tends to do with this stick.

## Versions that do not work

These are the ones I tried and the ways they failed, so you can skip them.

| Combination | Result |
| --- | --- |
| Python 3.13 + `ai-edge-litert` 2.2.0 | segfault building the interpreter |
| Python 3.11 + `tflite-runtime` 2.14.0 | segfault building the interpreter |
| Python 3.10 + `tflite-runtime` 2.13.0 | interpreter builds, `invoke()` fails with "unresolved custom op" |
| Python 3.9 + `tflite-runtime` 2.11.0 | interpreter builds, `invoke()` fails the same way |
| Python 3.9 + Coral's `tflite-runtime` 2.5.0 | works |

The segfaults happen inside `Subgraph::OpInit`, called from libedgetpu while it modifies the graph. If you're ever staring at one, a gdb backtrace tells you quickly whether the crash is a version mismatch (crash on the graph) or a missing device (the delegate fails to load at all).

## Other things that cost time

The device presents two different USB IDs. `1a6e:089a` when it's freshly plugged in and uninitialized, and `18d1:9302` once the runtime has loaded firmware onto it. Both need a udev rule, and both are normal.

A charge-only USB cable looks identical to a working one until you check. The LED lights up because power flows, but the kernel logs nothing at all. If the LED is on and `lsusb` shows no new device, it's the cable or the port, not the software.

Arch has no `plugdev` group, which the stock udev rule expects. This repo's rule sets a permissive mode instead.

## License

MIT. See LICENSE.