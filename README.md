# DeepStream Data Collection Pipeline

A production-ready, real-time data collection pipeline built on NVIDIA DeepStream 7.0 for multi-camera YOLO-based object detection. This pipeline is designed for easy model replacement and supports both raw data collection and YOLO dataset generation.

## ✨ Features

- **Multi-Camera Support**: Process multiple USB/CSI cameras simultaneously
- **Model Flexibility**: Easy replacement of YOLOv8 models with minimal configuration changes
- **Dual Output Modes**: Raw frame collection or YOLO-format dataset generation
- **Session-Based Organization**: Time-based folder rotation for organized data management
- **Clean Frame Extraction**: Captures frames before DeepStream overlay processing
- **System Monitoring**: Automatic system metrics logging (with `jtop` on Jetson)
- **Similarity Filtering**: IoU-based detection filtering to reduce redundant saves
- **Bounding Box Visualization**: Optional bounding box overlay on saved images

## 🔧 Requirements

### Hardware
- NVIDIA Jetson (Nano, Xavier, Orin) or dGPU-enabled system
- USB or CSI cameras
- Sufficient storage for data collection

### Software
- NVIDIA DeepStream 7.0+
- Python 3.8+
- OpenCV
- PyTorch (for model conversion)
- Ultralytics YOLOv8

### Python Dependencies
```bash
pip install opencv-python numpy ultralytics torch torchvision onnx
# For Jetson system monitoring (optional)
pip install jetson-stats
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-username/deepstream-data-collection-pipeline.git
cd deepstream-data-collection-pipeline
```

### 2. Run with Default YOLOv8m Model
```bash
cd scripts
python3 deepstream_demux.py -i /dev/video0 /dev/video1
```

### 3. Enable Dataset Mode (YOLO Format)
```bash
python3 deepstream_demux.py -i /dev/video0 --dataset
```

### 4. Enable Bounding Box Visualization
```bash
python3 deepstream_demux.py -i /dev/video0 --bbox
```

## 🔄 Using Your Own YOLOv8 Model

### Step 1: Convert Model to ONNX
```bash
cd utils
python3 export_yoloV8.py -w /path/to/your/model.pt --opset 11 --dynamic
```

### Step 2: Move ONNX File
```bash
mv your-model.pt.onnx ../models/
```

### Step 3: Update Configuration
Edit ds_demux_pgie_config.txt:

**Initial run (before engine generation):**
```ini
[property]
onnx-file=/home/user/path/to/deepstream-data-collection-pipeline/models/your-model.pt.onnx
# Comment out model-engine-file for first run
# model-engine-file=...
num-detected-classes=80  # Update if different
```

**After first run (engine file generated):**
```ini
[property]
onnx-file=/home/user/path/to/deepstream-data-collection-pipeline/models/your-model.pt.onnx
model-engine-file=/home/user/path/to/deepstream-data-collection-pipeline/models/your-model.pt.onnx_b3_gpu0_fp16.engine
num-detected-classes=80  # Update if different
```

### Step 4: Update Class Configuration (if needed)

If your model has different number of classes than COCO (80), update:

**In structured_saver_1.py:**
```python
# Update CLASS_NAMES dictionary with your classes
CLASS_NAMES = {
    0: "your_class_1",
    1: "your_class_2",
    # ... add all your classes
}

# Update SAVE_CLASSES if you want to filter which classes to save
SAVE_CLASSES = CLASS_NAMES.copy()  # Save all classes
```

**In ds_demux_pgie_config.txt:**
```ini
num-detected-classes=YOUR_CLASS_COUNT
```

## ⚙️ Configuration

### Multi-Camera Engine Building
**Important**: Build your TensorRT engine with the maximum number of cameras you intend to use:

```bash
# For up to 5 cameras, run first with 5 camera inputs
python3 deepstream_demux.py -i /dev/video0 /dev/video1 /dev/video2 /dev/video3 /dev/video4
```

A model built for 5 cameras can run with fewer cameras, but not vice versa.

### Session Rotation
Modify session rotation interval in structured_saver_1.py:
```python
class SessionManager:
    def __init__(self, base_dir="data", rotation_interval=300):  # 300 seconds = 5 minutes
```

### Detection Filtering
Configure IoU-based similarity filtering:
```python
inference_saver = InferenceSaver(
    base_dir=output_dir,
    enable_bbox_similarity_filtering=True,
    iou_threshold=0.5,      # Similarity threshold
    memory_seconds=30,      # Memory duration
)
```

## 🏷️ Flags & Options

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input` | Camera device paths (required) | None |
| `--bbox` | Enable bounding box drawing on saved images | Disabled |
| `--dataset` | Enable YOLO dataset format output | Disabled |

### Usage Examples
```bash
# Single camera, raw data collection
python3 deepstream_demux.py -i /dev/video0

# Multiple cameras with bounding boxes
python3 deepstream_demux.py -i /dev/video0 /dev/video1 --bbox

# Dataset generation mode
python3 deepstream_demux.py -i /dev/video0 --dataset

# All features enabled
python3 deepstream_demux.py -i /dev/video0 /dev/video1 --bbox --dataset
```

## 📁 Output Structure

### Default Mode (Raw Data Collection)
```
data/
├── MM-DD-YY-HH-MM-SS-fff/          # Session timestamp
│   ├── cam0/                        # Camera 0 data
│   │   ├── person/                  # Per-class organization
│   │   │   ├── images/              # Detection images
│   │   │   └── metadata.csv         # Detection metadata
│   │   └── detections/              # Combined detections
│   │       ├── images/              # Multi-object frames
│   │       └── all_detections.csv   # Combined metadata
│   ├── cam1/                        # Camera 1 data
│   └── system_metrics.csv           # System performance metrics
```

### Dataset Mode (YOLO Format)
```
data/
├── MM-DD-YY-HH-MM-SS-fff/          # Session timestamp
│   ├── images/                      # All detection images
│   │   ├── timestamp_s0_f1234.jpg   # Frame images
│   │   └── ...
│   ├── labels/                      # YOLO format labels
│   │   ├── timestamp_s0_f1234.txt   # Corresponding annotations
│   │   └── ...
│   └── system_metrics.csv           # System performance metrics
```

### CSV Metadata Fields
```csv
timestamp,date,time,stream_id,frame_num,obj_id,class_id,class_name,left,top,width,height,confidence
```

## 🔧 Troubleshooting & Notes

### Common Issues

**Engine File Generation Failed:**
- Ensure ONNX model path is absolute in config file
- Check that model is compatible with your DeepStream version
- Verify sufficient GPU memory for model loading

**Frame Extraction Issues:**
- Confirmed working on Jetson platforms with DeepStream 7.0
- Frame extraction uses NVIDIA's recommended NVMM memory approach
- If frames appear black, check camera permissions and device paths

**Performance Optimization:**
- Disable sync on sinks for maximum throughput: `sink.set_property("sync", 0)`
- Adjust batch timeout in config: `MUXER_BATCH_TIMEOUT_USEC = 33000`
- Use appropriate TensorRT precision (FP16 recommended for Jetson)

### Multi-Camera Notes
- Each camera gets mapped to `cam{N}` format (cam0, cam1, etc.)
- Camera device order determines mapping
- Build TensorRT engine with maximum intended camera count

### Memory Management
- Pipeline uses NVMM memory for efficient GPU-CPU transfers
- Frame extraction follows NVIDIA's recommended practices
- Automatic cleanup on script termination

## 🛠️ Development

### Adding Custom Post-Processing
Modify the `pgie_src_pad_buffer_probe` function in deepstream_demux.py to add custom detection logic.

### Extending Class Support
1. Update `CLASS_NAMES` dictionary in structured_saver_1.py
2. Modify `num-detected-classes` in config file
3. Update any class-specific filtering logic

### Custom Output Formats
Extend `InferenceSaver` class to support additional output formats beyond CSV and YOLO.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{deepstream_data_collection_pipeline,
  title={DeepStream Data Collection Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/deepstream-data-collection-pipeline}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Support

For issues and questions:
- Open a GitHub issue for bugs and feature requests
- Check NVIDIA DeepStream documentation for platform-specific issues
- Verify hardware compatibility with DeepStream 7.0+