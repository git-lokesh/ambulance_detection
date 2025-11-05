# SmartLife — Emergency Vehicle Detection Dashboard (Streamlit)

Detects emergency vehicles (EV) in a video file or webcam using YOLOv8 + OpenCV, with a simple Streamlit UI.

> **Heads up:** In your code you set `MODEL_PATH = 'yolov8s.pt'` and comment “Using the Medium model.”  
> YOLO naming: `n`=nano, `s`=small, `m`=medium, `l`=large, `x`=xlarge.  
> If you truly want **medium**, set `MODEL_PATH = 'yolov8m.pt'`.

---

## Features

- Streamlit dashboard (web UI)  
- Works with **uploaded video** or **webcam**  
- Uses **Ultralytics YOLOv8** for detection  
- Per-frame drawing of detections and **status banner**  
- Confidence threshold + EV class filter (currently `bus` and `truck` as proxies)

---

## Requirements

- Windows 10/11
- Conda (Anaconda/Miniconda)
- Python 3.10–3.11

---

## Quick Start (Conda)

```bat
:: 1) Create and activate env
conda create -n smartlife python=3.11 -y
conda activate smartlife

:: 2) Upgrade pip basics
python -m pip install --upgrade pip wheel setuptools

:: 3) Install CPU-only PyTorch (recommended for Windows if you don't have CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

:: 4) Install app deps
pip install ultralytics opencv-python streamlit numpy

:: 5) (Optional) If you’ll use EasyOCR or other OCR, install it later; your current code does NOT require it.
```

> If you run into video codec issues on Windows, install **FFmpeg** and ensure it’s on your PATH, or try `opencv-contrib-python`.

---

## Project Files

- `app.py` — your Streamlit app
- `README.txt` — this file (markdown formatted)

---

## Run the App

```bat
conda activate smartlife
streamlit run app.py
```

Streamlit will print a local URL (e.g. `http://localhost:8501`). Open it in your browser.

---

## Using the Dashboard

1. **Model:** The first run will auto-download `yolov8s.pt` (or whichever model you set in `MODEL_PATH`).
2. **Select Source:**
   - **Upload a video file**: pick `.mp4/.avi/.mov/.mkv`, click **Start Processing**
   - **Use Webcam**: click **Start Webcam**, press **Stop** to end
3. **Status Panel:**
   - ✅ **AMBULANCE DETECTED!**
   - ⚪ **No Emergency Vehicle Detected**

---

## Configuration in Code

```python
MODEL_PATH = 'yolov8s.pt'   # change to 'yolov8m.pt' if you want medium
CONFIDENCE_THRESHOLD = 0.6  # 0.0–1.0
EV_CLASSES = [5, 7]         # COCO IDs for 'bus' and 'truck'
```

- **Model Types:**
  - `yolov8n.pt` → smallest / fastest  
  - `yolov8s.pt` → small  
  - `yolov8m.pt` → medium  
  - `yolov8l.pt`, `yolov8x.pt` → large models

---

## Troubleshooting

- **Could not open video source**
  - Try another video or change webcam index (0 / 1 / 2)
- **Slow performance**
  - Use smaller model (`yolov8n.pt`)
  - Reduce video resolution
- **Torch install issues**
  - Use the CPU‑only install command given above

---

## Optional: `environment.yml`

```yaml
name: smartlife
channels:
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
      - torch
      - torchvision
      - ultralytics
      - opencv-python
      - streamlit
      - numpy
```

Create it:

```bat
conda env create -f environment.yml
conda activate smartlife
```

---

## Notes / Next Steps

- Train a **custom YOLO model** with an `ambulance` class for higher accuracy
- Add **traffic signal integration** with API/webhook when detection trigger occurs

---

## License

YOLOv8 weights and Ultralytics tools are subject to their respective licenses.
