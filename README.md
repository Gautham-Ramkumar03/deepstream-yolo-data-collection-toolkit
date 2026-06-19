<div align="center">

# 🎥 DeepStream YOLO Data Collection Toolkit

**Turn live cameras into clean, organized datasets — a production-grade NVIDIA DeepStream pipeline for multi-camera YOLO detection and automatic data collection.**

Runs real-time YOLO inference on multiple cameras, captures *clean* frames (before any overlay), and saves them into session-rotated folders as raw detections or ready-to-train YOLO datasets.

[![DeepStream](https://img.shields.io/badge/NVIDIA-DeepStream%207.0%2B-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/deepstream-sdk)
[![Jetson](https://img.shields.io/badge/Platform-Jetson%20%7C%20dGPU-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/embedded-computing)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8%20%7C%20YOLO11-purple.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Why this exists

Collecting real-world training data from deployed cameras is tedious and error-prone: you
have to record video, run inference offline, crop/clean frames, label them, and keep
everything organized per camera and per session. Do it live on an edge device and you also
fight memory copies, overlay artifacts baked into your images, and long-running stability.

This toolkit does it all **in one real-time pipeline**. It runs YOLO on every camera,
grabs **clean frames before the bounding-box overlay is drawn**, deduplicates near-identical
detections, and writes everything into **time-rotated session folders** — either as raw
per-class detections or as a directly-trainable YOLO dataset (`images/` + `labels/`).

## ✨ Features

- **Multi-camera** — process several USB/CSI cameras at once via `nvstreammux` + `nvstreamdemux`.
- **Clean frame capture** — a `tee` branch taps the stream *before* `nvosd`, so saved images have no overlay burned in.
- **Two output modes** — raw per-camera/per-class detections, or YOLO-format dataset (`--dataset`).
- **Session rotation** — a dedicated timer thread rotates output into fresh timestamped folders and safely closes/reopens all files mid-run.
- **Similarity filtering** — IoU-based detection memory skips redundant near-duplicate saves.
- **System telemetry** — logs CPU/GPU/RAM/power/thermal metrics on Jetson (via `jtop`).
- **Model-agnostic** — swap in any YOLOv8/YOLO11 model with a config change; bundled DeepStream YOLO parser included.
- **Optional bounding boxes** — `--bbox` overlays boxes on saved images when you want them.

## 🏗️ Pipeline architecture

```
 cam0 ─┐
 cam1 ─┤  v4l2src → videoconvert → nvvideoconvert
 ...  ─┘            │
                    ▼
              nvstreammux ─► nvinfer (PGIE / YOLO) ─► nvstreamdemux
                                                          │  (per stream)
                          ┌───────────────────────────────┴───────────────┐
                          ▼                                                 ▼
                    nvvideoconvert ─► tee ───────────────┐                 ...
                                       │                 │
                  ┌────────────────────┘                 └──────────────────────┐
                  ▼  (display branch)                       ▼  (clean branch)
            nvdsosd ─► sink   ◄── detection probe      nvvideoconvert(RGBA)
            (overlay)         (reads metadata,          ─► capsfilter ─► fakesink
                               saves detections)            ▲
                                                      FrameGrabber probe
                                                   (captures CLEAN frames)
```

The detection probe on the OSD sink pad reads YOLO metadata, while the **FrameGrabber**
pulls the matching *clean* RGBA frame from the parallel branch — so detection data and
overlay-free images stay in sync.

## 📂 Repository structure

```text
.
├── scripts/
│   ├── deepstream_demux.py        # main multi-camera pipeline + detection probe
│   ├── frame_grabber.py           # thread-safe clean-frame capture from NVMM
│   ├── structured_saver.py        # session rotation, IoU dedup, CSV/YOLO saving, telemetry
│   └── ds_demux_pgie_config.txt   # nvinfer (PGIE) configuration
├── nvdsinfer_custom_impl_Yolo/    # DeepStream YOLO output parser (C++/CUDA) + Makefile
├── utils/
│   ├── export_yoloV8.py           # export a YOLOv8/YOLO11 .pt model to DeepStream ONNX
│   └── labels.txt                 # class labels (COCO 80 by default)
├── common/                        # DeepStream helpers (bus_call, FPS, platform_info, utils)
├── models/                        # place your .pt / .onnx / .engine here (see models/README.md)
└── README.md
```

## 🔧 Requirements

**Hardware:** NVIDIA Jetson (Nano / Xavier / Orin) or a dGPU system, plus USB/CSI cameras.

**Software:**
- NVIDIA DeepStream **7.0+** (with `pyds` Python bindings and GStreamer)
- Python **3.8+**, OpenCV, NumPy
- For model export: PyTorch, Ultralytics, ONNX
- Optional (Jetson telemetry): `jetson-stats` (`jtop`)

```bash
pip install opencv-python numpy ultralytics torch torchvision onnx
pip install jetson-stats   # optional, Jetson only
```

## 🚀 Quick start

### 1. Clone

```bash
git clone https://github.com/Gautham-Ramkumar03/deepstream-yolo-data-collection-toolkit.git
cd deepstream-yolo-data-collection-toolkit
```

### 2. Build the YOLO output parser

The pipeline loads a custom DeepStream parser (`custom-lib-path` in the config). Build it
once for your platform:

```bash
cd nvdsinfer_custom_impl_Yolo
# Point CUDA_VER at your install (e.g. 12.2 on Orin / DeepStream 7.0)
CUDA_VER=12.2 make
cd ..
```

This produces `nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`.

### 3. Add a model

See **[models/README.md](models/README.md)** for the full walkthrough. In short:

```bash
cd models
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
cd ../utils
python3 export_yoloV8.py -w ../models/yolo11m.pt --opset 11 --dynamic
```

The PGIE config (`scripts/ds_demux_pgie_config.txt`) already points at
`../models/yolo11m.pt.onnx` with 80 COCO classes — adjust if you use a different model.

### 4. Run

```bash
cd scripts

# Single camera, raw data collection
python3 deepstream_demux.py -i /dev/video0

# Multiple cameras
python3 deepstream_demux.py -i /dev/video0 /dev/video1

# YOLO dataset mode (images/ + labels/)
python3 deepstream_demux.py -i /dev/video0 --dataset

# Draw bounding boxes on saved images
python3 deepstream_demux.py -i /dev/video0 --bbox
```

> **First run builds the TensorRT engine** (can take a few minutes). Build it with the
> **maximum** number of cameras you intend to use — an engine built for 5 cameras runs with
> fewer, but not vice-versa. After it's generated, uncomment `model-engine-file` in the
> config to skip rebuilding.

## 🏷️ Flags

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input` | Camera device path(s), e.g. `/dev/video0` (one or more) | *required* |
| `--dataset` | Output a YOLO-format dataset (`images/` + `labels/`) instead of per-class folders | off |
| `--bbox` | Draw bounding boxes on saved images | off |

## ⚙️ Configuration

A few behaviors are set in code (tune them to your needs):

**Session rotation interval** — in `scripts/structured_saver.py`, `SessionManager`:

```python
class SessionManager:
    def __init__(self, base_dir="data", rotation_interval=15):  # seconds; raise for production (e.g. 300)
```

A dedicated timer thread rotates to a new timestamped session at this interval and safely
closes/reopens all CSV and metrics files.

**Similarity (IoU) filtering** — where `InferenceSaver` is created in
`scripts/deepstream_demux.py`:

```python
inference_saver = InferenceSaver(
    base_dir="data",
    enable_bbox_similarity_filtering=True,
    iou_threshold=0.5,     # higher = stricter "duplicate" definition
    memory_seconds=30,     # how long a detection is remembered
)
```

**Custom classes** — if your model isn't COCO-80, update `CLASS_NAMES` (and `SAVE_CLASSES`)
in `scripts/structured_saver.py`, set `num-detected-classes` in the PGIE config, and replace
`utils/labels.txt`.

## 📁 Output structure

### Default mode (raw collection)
```
data/
└── MM-DD-YY-HH-MM-SS-fff/        # session timestamp
    ├── cam0/
    │   ├── person/
    │   │   ├── images/
    │   │   └── metadata.csv
    │   └── detections/           # combined multi-object frames
    │       ├── images/
    │       └── all_detections.csv
    ├── cam1/
    └── system_metrics.csv        # Jetson telemetry (if jtop available)
```

### Dataset mode (`--dataset`)
```
data/
└── MM-DD-YY-HH-MM-SS-fff/
    ├── images/                   # clean frames
    │   └── <ts>_s0_f1234.jpg
    ├── labels/                   # YOLO labels (one per image)
    │   └── <ts>_s0_f1234.txt
    └── system_metrics.csv
```

### Detection CSV fields
```
timestamp, date, time, stream_id, frame_num, obj_id, class_id, class_name,
left, top, width, height, confidence
```

## 🛠️ Troubleshooting

- **`libnvdsinfer_custom_impl_Yolo.so` not found** — build the parser (step 2) with the right `CUDA_VER`.
- **Engine generation fails** — make sure the ONNX path in the config is correct and there's enough GPU memory; first build is slow.
- **Black / missing frames** — check camera permissions and device paths; the toolkit falls back to a black frame and logs a warning.
- **No system metrics** — `jtop` is Jetson-only and optional; metrics are skipped if unavailable.
- **Performance** — sinks run with `sync=0` for throughput; use FP16 on Jetson and pick a model size that fits your device (see models/README.md).

## 🧩 Development

- **Custom post-processing** — extend `pgie_src_pad_buffer_probe` in `deepstream_demux.py`.
- **New output formats** — extend `InferenceSaver` in `structured_saver.py` (CSV and YOLO are built in).
- **Frame handling** — clean-frame capture lives in `frame_grabber.py` (thread-safe, per-stream).

## 📄 License

Released under the [MIT License](LICENSE).

## 📚 Citation

```bibtex
@software{deepstream_yolo_data_collection_toolkit,
  title  = {DeepStream YOLO Data Collection Toolkit},
  author = {Gautham Ramkumar},
  year   = {2025},
  url    = {https://github.com/Gautham-Ramkumar03/deepstream-yolo-data-collection-toolkit}
}
```

## 🙏 Acknowledgements

The `nvdsinfer_custom_impl_Yolo` parser builds on the community DeepStream-YOLO work for
running Ultralytics YOLO models inside the DeepStream SDK.
