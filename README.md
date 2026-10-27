# Bolt Detection AI

AI-powered bolt detection system built using YOLO, OpenCV, and Flask.

## Features

- Real-time bolt detection
- Live webcam feed
- AI confidence display
- Missing bolt detection
- Industrial dashboard UI
- Flask web interface
- **Inspection History** (`/history`): every missing-fastener alert and reference capture is stored in SQLite
  with a snapshot of the exact frame; filter by type, click thumbnails, **Export CSV**
- Automatic model discovery (`best.pt` under `runs/**/weights/`, or env var `BOLT_MODEL`)
- Health endpoint `/api/health`

---

## Prerequisites

Install the following before running:

- Python 3.11+
- Git

---

## Step 1: Clone Repository

```bash
git clone https://github.com/m22854784-creator/BoltDetectionAI.git
```

Enter the project folder:

```bash
cd BoltDetectionAI
```

---

## Step 2: Create Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4: Run Application

```bash
python app.py
```

---

## Step 5: Open Dashboard

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## Project Structure

```text
BoltDetectionAI
│
├── app.py
├── inspections.db   (auto-created)
├── snapshots/       (auto-created alert images)
├── requirements.txt
├── README.md
│
├── static
│   ├── css
│   ├── js
│   └── assets
│
├── templates
│
├── videos
│
└── runs
    └── detect
        └── GodrejVision
            └── Fastener_AI
                └── weights
                    └── best.pt
```

---

## Technologies Used

- Python
- Flask
- OpenCV
- YOLO
- HTML
- CSS
- JavaScript

---

## Notes

- Ensure webcam permissions are enabled.
- The trained YOLO model file (`best.pt`) must be present in the weights folder.
- Recommended resolution: 1280x720 or higher.

---

## Author

Devashish Niphade

GitHub:
https://github.com/m22854784-creator

## Timed Inspection Cycle (new)
Switch **Inspection Cycle → Timed** on the dashboard. The system waits for a part, inspects it for a set time
(default **10 s**, change it in the panel; stored in `settings.json`), then shows **PASS / FAIL**.
- Verdict uses the most frequent fastener count over the whole window, so one missed frame cannot cause a false FAIL.
- PASS/FAIL is saved to History with a snapshot, part name and stability %.
- **Part Library:** capture a reference, click *Save as part*, then select parts from the list later.
- Daily yield (Inspected / Pass / Fail / Yield %) and a sound on every result.
- Advanced (API): `POST /control/cycle_config {"duration":10,"min_present":1,"grace":2}`
  (`min_present` = fasteners needed to count as a part, `grace` = seconds without part before the cycle resets).
Live mode (default) behaves exactly as before.

## Improving accuracy
See **ACCURACY_GUIDE.md**. Tools: `train_improved.py` (train / fine-tune / measure real accuracy),
`tools/tune_inference.py` (find best conf/imgsz), `settings.json -> detection` (live tuning + hard-sample collector).

## Smooth video, fewer false detections (v5)
* Video and AI now run in separate threads: the picture streams at camera speed (up to 30 FPS) while YOLO updates the
  boxes/counts as fast as the PC allows. Hover the FPS pill to see both numbers (video / AI detection).
* The video panel automatically matches the camera shape (no side bars).
* **Set Zone** (under the video): draw the area where bolts are. Everything outside is ignored (faces, phones, background).
* `settings.json -> detection`: `conf`, `count_smooth` (stable count), `min_box_area_pct` / `max_box_area_pct`
  (ignore tiny/huge false boxes), `device` (GPU is used automatically if available).
* CPU-only PC? `python tools/export_fast.py <best.pt>` makes detection 2-4x faster.

## Camera & Image panel (v6)
Sidebar -> **Camera & Image** (shows "Too bright / Too dark / Exposure OK" live).
* **Auto-fix exposure**: one click fixes washed-out / overexposed images (software, works on any camera).
* **Highlight recovery / Brightness / Contrast / Enhance bolt edges**: applied to the video AND to the AI input.
* **Resolution + FPS**: 640x480, HD 1280x720 (default), Full HD 1920x1080, 15/30/60 FPS (webcam, MJPG mode).
* Camera exposure, autofocus and focus: used only if your webcam driver supports them.
* All values are saved in `settings.json -> camera`.
