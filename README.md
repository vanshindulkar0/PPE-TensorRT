# PPE Detection using TensorRT

Real-time Personal Protective Equipment (PPE) detection using YOLO, TensorRT, CUDA, and OpenCV. Built for GPU-accelerated inference on live webcam feeds.

> **Status:** Working prototype — v1.0. Core pipeline (ONNX → TensorRT FP16 → CUDA inference → real-time webcam detection) is functional at ~19–21 FPS. See [Roadmap](#roadmap) for planned improvements.

## Pipeline

```
YOLO (PyTorch)
      ↓
   ONNX export
      ↓
TensorRT FP16 Engine
      ↓
TensorRT Runtime + CUDA GPU inference
      ↓
YOLO post-processing (NMS, box decoding)
      ↓
OpenCV visualization (webcam)
```

## Classes

The model detects 14 classes related to construction-site PPE compliance:

| ID | Class          | ID | Class          |
|----|----------------|----|----------------|
| 0  | Fall-Detected  | 7  | NO-Goggles     |
| 1  | Gloves         | 8  | NO-Hardhat     |
| 2  | Goggles        | 9  | NO-Mask        |
| 3  | Hardhat        | 10 | NO-Safety Vest |
| 4  | Ladder         | 11 | Person         |
| 5  | Mask           | 12 | Safety Cone    |
| 6  | NO-Gloves      | 13 | Safety Vest    |

## Project Structure

```
PPE-TensorRT/
│
├── models/              # Place your .engine / .onnx / .pt files here (not tracked in git)
│   └── README.md
│
├── src/
│   ├── build_engine.py   # Converts ONNX model to a TensorRT engine
│   ├── inference.py      # Runs inference on a single image via TensorRT
│   ├── preprocessing.py  # Image resizing/normalization for model input
│   ├── postprocessing.py # Decodes raw model output into bounding boxes
│   └── webcam.py         # Real-time webcam detection loop
│
├── images/               # Sample input images
├── test_images/          # Example output/prediction images
├── requirements.txt
├── .gitignore
└── README.md
```

## Hardware & Environment

- **GPU:** NVIDIA RTX 3050
- **CUDA:** 12.6
- **TensorRT:** 10.12
- **Python:** 3.10
- **OS:** Windows 11

## Performance

Current webcam baseline:

**~19–21 FPS**

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/vanshindulkar0/PPE-TensorRT.git
   cd PPE-TensorRT
   ```

2. Create and activate a conda environment (Python 3.10 recommended):
   ```bash
   conda create -n tensorrt python=3.10
   conda activate tensorrt
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install TensorRT (NVIDIA Windows ZIP release — not pip-only). Download from the [NVIDIA TensorRT page](https://developer.nvidia.com/tensorrt), extract it, and add its `lib` folder to your system `PATH`.

5. Place your trained model files (`.pt`, `.onnx`, or `.engine`) in the `models/` folder. See `models/README.md` for details.

6. Build a TensorRT engine from your ONNX model (if you don't already have a `.engine` file):
   ```bash
   python src/build_engine.py
   ```

7. Run inference on a single image:
   ```bash
   python src/inference.py
   ```

8. Run real-time webcam detection:
   ```bash
   python src/webcam.py
   ```

## Roadmap

- [x] **v1.0** — TensorRT inference, webcam support, ~19–21 FPS
- [ ] **v1.1** — Accuracy validation (compare PyTorch YOLO output vs TensorRT output)
- [ ] **v1.2** — FPS profiling (breakdown of preprocessing / inference / postprocessing / memory transfer time)
- [ ] **v1.3** — Performance optimization
- [ ] **v2.0** — C++ TensorRT implementation
- [ ] **v3.0** — NVIDIA DeepStream integration

## License

See [LICENSE](LICENSE) for details.
