"""
Godrej Fastener Vision - Flask backend

Architecture (fixes the original per-request generator design):
  * ONE background worker thread owns the camera/video + YOLO tracker.
  * Each processed frame and its counts are published together, so the number
    on the video overlay and the dashboard KPI are always the same value.
  * /video_feed only streams the latest JPEG (any number of browser tabs is safe).
  * YOLO model, model.track() call and box drawing are unchanged.
"""
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
import time
import sqlite3
import csv
import io
import glob
import json
from collections import Counter
from flask import Flask, render_template, Response, jsonify, request, send_from_directory, abort
from ultralytics import YOLO
from collections import deque
import cv2
import numpy as np
import os
import threading
import time

app = Flask(__name__)

console = Console()
last_log_time = 0
live_table = None

# Hide Flask request spam
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(BASE_DIR, "snapshots")
DB_PATH = os.path.join(BASE_DIR, "inspections.db")
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "demo.mp4")
WEBCAM_INDEX = 1 # <--- for laptop camera 0 , for enternal webcam 1 , 2 , 3... 


def resolve_model_path():
    """Find best.pt automatically (env var > known locations > any runs/**/weights/best.pt)."""
    candidates = [
        os.environ.get("BOLT_MODEL", ""),
        os.path.join(BASE_DIR, "runs", "detect", "GodrejVision", "Fastener_AI", "weights", "best.pt"),
        os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt"),
        os.path.join(BASE_DIR, "best.pt"),
    ]
    candidates += glob.glob(os.path.join(BASE_DIR, "runs", "**", "weights", "best.pt"), recursive=True)
    for c in candidates:
        if c and os.path.exists(c):
            return c
    console.print("[bold red]ERROR:[/bold red] trained model best.pt not found.")
    console.print("Place it at: runs/detect/GodrejVision/Fastener_AI/weights/best.pt "
                  "(or set env var BOLT_MODEL=<path>)")
    raise SystemExit(1)


MODEL_PATH = resolve_model_path()
model = YOLO(MODEL_PATH)

# ---------------------------------------------------------------- shared state
lock = threading.RLock()                 # protects dashboard_data / ctrl / events
frame_cond = threading.Condition()       # signals a new JPEG is available
latest = {"jpg": None, "seq": 0}

dashboard_data = {
    "current_bolts": 0,
    "reference_bolts": 0,
    "missing_bolts": 0,
    "status": "OK",
    "active_ids": [],
    "fps": 0,
    "frame_seq": 0,
    "accuracy": 0,
    "det_fps": 0,
    "frame_w": 16,
    "frame_h": 9,
}
ctrl = {
    "running": True,
    "source": "video",     # "video" | "webcam"
    "ref": None,           # None = auto-follow, int = locked reference
    "switch": True,        # worker (re)opens the capture when True
}
events = deque(maxlen=40)
_event_id = 0


# ------------------------------------------------- settings.json (persistent)
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")
settings = {
    "cycle": {"enabled": False, "duration": 10, "min_present": 1, "grace": 2.0},
    # --- detection tuning (edit in settings.json, restart app) ---
    "detection": {
        "conf": 0.30,            # min confidence to keep a box (lower = more bolts found, more false hits)
        "iou": 0.5,              # NMS overlap threshold
        "imgsz": None,           # e.g. 960 or 1280 helps SMALL bolts (slower); None = model default
        "count_all_boxes": True, # count every detected box (not only boxes that already have a tracker ID)
        "collect_samples": False,# save hard frames + auto-labels to dataset_candidates/ for re-training
        "count_smooth": 3,       # median of last N detections -> stable count, no flicker (1 = off)
        "min_box_area_pct": 0.0, # ignore boxes smaller than this % of the frame (tiny false hits)
        "max_box_area_pct": 100.0,# ignore boxes bigger than this % of the frame
        "roi": None,             # detection zone [x1,y1,x2,y2] (0..1). Set with "Set Zone" on the dashboard
        "device": None           # None = auto (GPU if available). Or 0 / "cpu"
    },
    # --- camera (hardware = real webcam controls, sw_* = software correction that works on ANY camera) ---
    "camera": {
        "width": 1280, "height": 720, "fps": 30,            # HD 720p @ 30 (try 1920x1080)
        "auto_exposure": True, "exposure": -6,              # manual exposure: Windows scale -13 (dark) .. 0 (bright)
        "autofocus": True, "focus": 0,
        "sw_brightness": 0,        # -100..100
        "sw_contrast": 1.0,        # 0.5..2.0
        "sw_gamma": 1.0,           # 0.4..3.0  (>1 darkens bright/washed-out images = highlight recovery)
        "sw_enhance": False        # local contrast (CLAHE) - makes bolt edges clearer
    },
    "parts": {},            # part library: {"Stator-A": 5}
    "active_part": None,
}


def load_settings():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        settings["cycle"].update(saved.get("cycle", {}))
        settings["detection"].update(saved.get("detection", {}))
        settings["camera"].update(saved.get("camera", {}))
        settings["parts"] = saved.get("parts", {})
        settings["active_part"] = saved.get("active_part")
    except Exception:
        pass
    ap = settings["active_part"]
    if ap in settings["parts"]:
        ctrl["ref"] = settings["parts"][ap]      # restore last selected part's reference
    else:
        settings["active_part"] = None


def save_settings():
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        console.print(f"[red]Settings save failed:[/red] {e}")


# ---------------------------------------------- timed inspection cycle (state machine)
# IDLE -> (part seen ~0.6 s) -> INSPECTING (N seconds, counts collected every frame)
#      -> RESULT (PASS/FAIL, snapshot + DB record) -> IDLE (part removed for `grace` s)
cyc = {
    "state": "IDLE", "t_start": 0.0, "present_since": None, "absent_since": None,
    "counts": Counter(), "frames": 0, "result": None, "result_id": 0,
    "stats": {"pass": 0, "fail": 0},
}


def cycle_update(n, now):
    """Advance the state machine with this frame's count. Returns a verdict dict when a cycle finishes."""
    c, cfg = cyc, settings["cycle"]
    if not cfg["enabled"]:
        c["state"], c["present_since"], c["absent_since"] = "IDLE", None, None
        return None
    present = n >= cfg["min_present"]

    if c["state"] == "IDLE":
        if present:
            c["present_since"] = c["present_since"] or now
            if now - c["present_since"] >= 0.6:          # debounce: ignore 1-frame noise
                c.update(state="INSPECTING", t_start=now, counts=Counter(), frames=0, absent_since=None)
                log_event("info", f"Part detected - inspecting for {cfg['duration']} s")
        else:
            c["present_since"] = None

    elif c["state"] == "INSPECTING":
        c["counts"][n] += 1
        c["frames"] += 1
        if present:
            c["absent_since"] = None
        else:
            c["absent_since"] = c["absent_since"] or now
            if now - c["absent_since"] > cfg["grace"]:
                c.update(state="IDLE", present_since=None, absent_since=None)
                log_event("alert", "Inspection aborted - part removed early")
                return None
        if now - c["t_start"] >= cfg["duration"]:
            nz = {k: v for k, v in c["counts"].items() if k > 0} or dict(c["counts"])
            best = max(nz.items(), key=lambda kv: (kv[1], kv[0]))[0]     # most frequent count
            stability = round(100 * c["counts"][best] / max(c["frames"], 1))
            ref = ctrl["ref"]
            if ref is None:
                verdict, missing = "NO_REF", 0
            else:
                missing = max(ref - best, 0)
                verdict = "PASS" if missing == 0 else "FAIL"
            c["state"], c["absent_since"] = "RESULT", None
            return {"verdict": verdict, "count": best, "ref": ref or 0, "missing": missing,
                    "stability": stability, "frames": c["frames"], "part": settings["active_part"]}

    elif c["state"] == "RESULT":
        if present:
            c["absent_since"] = None
        else:
            c["absent_since"] = c["absent_since"] or now
            if now - c["absent_since"] >= cfg["grace"]:
                c.update(state="IDLE", present_since=None, absent_since=None)
    return None


def cycle_finish(v, jpg):
    """Log + store the verdict (called with the exact frame shown to the operator)."""
    c = cyc
    c["result_id"] += 1
    c["result"] = dict(v, time=time.strftime("%H:%M:%S"), id=c["result_id"])
    part = f" [{v['part']}]" if v["part"] else ""
    if v["verdict"] == "NO_REF":
        log_event("info", f"Inspection done{part}: {v['count']} fasteners (no reference set)")
        record("ok", f"Timed inspection{part}: {v['count']} fasteners (no reference)", v["count"], 0, 0, jpg)
        return
    c["stats"]["pass" if v["verdict"] == "PASS" else "fail"] += 1
    if v["verdict"] == "PASS":
        log_event("ok", f"PASS{part}: {v['count']}/{v['ref']} fasteners ({v['stability']}% stable)")
    else:
        log_event("alert", f"FAIL{part}: {v['missing']} missing ({v['count']}/{v['ref']})")
    record(v["verdict"].lower(),
           f"{v['verdict']}{part}: {v['count']}/{v['ref']} fasteners, stability {v['stability']}%",
           v["count"], v["ref"], v["missing"], jpg)


def cycle_view(now):
    cfg, c = settings["cycle"], cyc
    elapsed = max(now - c["t_start"], 0) if c["state"] == "INSPECTING" else 0
    dur = cfg["duration"]
    return {"enabled": cfg["enabled"], "state": c["state"], "duration": dur,
            "elapsed": round(elapsed, 1), "remaining": round(max(dur - elapsed, 0), 1),
            "progress": round(min(elapsed / dur, 1), 3) if dur else 0,
            "min_present": cfg["min_present"], "grace": cfg["grace"],
            "result": c["result"], "stats": dict(c["stats"])}


def log_event(level, msg):
    global _event_id
    with lock:
        _event_id += 1
        events.appendleft({
            "id": _event_id,
            "t": time.strftime("%H:%M:%S"),
            "level": level,          # info | ok | alert
            "msg": msg,
        })


# ------------------------------------------------- inspection history (SQLite)
db_lock = threading.Lock()


def db_init():
    os.makedirs(SNAP_DIR, exist_ok=True)
    with db_lock, sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS inspections(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, type TEXT NOT NULL, source TEXT,
            current_count INTEGER, reference_count INTEGER, missing_count INTEGER,
            message TEXT, snapshot TEXT, part TEXT)""")
        cols = [r[1] for r in c.execute("PRAGMA table_info(inspections)")]
        if "part" not in cols:                       # migrate databases created by older versions
            c.execute("ALTER TABLE inspections ADD COLUMN part TEXT")
        today = time.strftime("%Y-%m-%d") + "%"
        for kind, n in c.execute("SELECT type, COUNT(*) FROM inspections WHERE ts LIKE ? "
                                 "AND type IN ('pass','fail') GROUP BY type", (today,)):
            cyc["stats"][kind] = n


def record(kind, message, current=0, reference=0, missing=0, jpg=None):
    """Store an inspection record (optionally with a snapshot image). Never raises."""
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        name = None
        if jpg is not None:
            name = time.strftime("%Y%m%d_%H%M%S") + f"_{kind}.jpg"
            with open(os.path.join(SNAP_DIR, name), "wb") as f:
                f.write(jpg)
        with db_lock, sqlite3.connect(DB_PATH) as c:
            c.execute("INSERT INTO inspections(ts,type,source,current_count,reference_count,"
                      "missing_count,message,snapshot,part) VALUES(?,?,?,?,?,?,?,?,?)",
                      (ts, kind, ctrl["source"], current, reference, missing, message, name,
                       settings["active_part"]))
    except Exception as e:
        console.print(f"[red]History write failed:[/red] {e}")


def query_history(kind=None, limit=200):
    q, args = "SELECT * FROM inspections", []
    if kind in ("alert", "reference", "ok", "pass", "fail"):
        q += " WHERE type=?"; args.append(kind)
    q += " ORDER BY id DESC LIMIT ?"; args.append(limit)
    with db_lock, sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(q, args).fetchall()]


def snapshot():
    with lock:
        d = dict(dashboard_data)
        d["active_ids"] = list(dashboard_data["active_ids"])
        d["running"] = ctrl["running"]
        d["source"] = ctrl["source"]
        d["reference_locked"] = ctrl["ref"] is not None
        d["events"] = list(events)
        d["cycle"] = cycle_view(time.time())
        d["roi"] = settings["detection"].get("roi")
        d["camera"] = dict(settings["camera"], info=dict(cam_info), mean=round(cam_stats["mean"] / 2.55, 1),
                           clip=round(cam_stats["clip_pct"], 1), webcam=ctrl["source"] == "webcam")
        d["parts"] = dict(settings["parts"])
        d["active_part"] = settings["active_part"]
        return d


# ------------------------------------------------------------------- camera controls
cam_info = {"width": 0, "height": 0, "fps": 0}
cam_stats = {"raw_mean": 128.0, "mean": 128.0, "clip_pct": 0.0}
_lut_cache = {}
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_camera_hw(cap):
    """Real webcam controls (many cheap webcams ignore some of them - the software sliders always work)."""
    c = settings["camera"]
    ae_auto, ae_manual = (0.75, 0.25) if os.name == "nt" else (3, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, ae_auto if c["auto_exposure"] else ae_manual)
    if not c["auto_exposure"]:
        cap.set(cv2.CAP_PROP_EXPOSURE, c["exposure"])
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if c["autofocus"] else 0)
    if not c["autofocus"]:
        cap.set(cv2.CAP_PROP_FOCUS, c["focus"])


def sw_adjust(frame):
    """Software brightness / contrast / highlight recovery / local contrast. Applied before display AND detection."""
    c = settings["camera"]
    b, ct, g, enh = c["sw_brightness"], c["sw_contrast"], c["sw_gamma"], c["sw_enhance"]
    if b == 0 and ct == 1.0 and g == 1.0 and not enh:
        return frame
    key = (b, ct, g)
    lut = _lut_cache.get(key)
    if lut is None:
        x = np.power(np.arange(256) / 255.0, g) * 255.0
        lut = np.clip((x - 128.0) * ct + 128.0 + b, 0, 255).astype(np.uint8)
        _lut_cache.clear(); _lut_cache[key] = lut
    out = cv2.LUT(frame, lut)
    if enh:
        l, a, bb = cv2.split(cv2.cvtColor(out, cv2.COLOR_BGR2LAB))
        out = cv2.cvtColor(cv2.merge((_clahe.apply(l), a, bb)), cv2.COLOR_LAB2BGR)
    return out


CAM_LIMITS = {"width": (320, 3840), "height": (240, 2160), "fps": (5, 120), "exposure": (-13, 0), "focus": (0, 255),
              "sw_brightness": (-100, 100), "sw_contrast": (0.5, 2.0), "sw_gamma": (0.4, 3.0)}
CAM_DEFAULTS = {"auto_exposure": True, "exposure": -6, "autofocus": True, "focus": 0,
                "sw_brightness": 0, "sw_contrast": 1.0, "sw_gamma": 1.0, "sw_enhance": False}


# ------------------------------------------------------------------- capture
def open_capture(source):
    if source == "webcam":
        cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW) if os.name == "nt" \
            else cv2.VideoCapture(WEBCAM_INDEX)
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
    if cap is not None and cap.isOpened():
        if source == "webcam":
            try:
                c = settings["camera"]
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                     # no stale frames = low latency
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # MJPG = HD at 30 fps
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
                cap.set(cv2.CAP_PROP_FPS, c["fps"])
                apply_camera_hw(cap)
                cam_info.update(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                fps=round(cap.get(cv2.CAP_PROP_FPS), 1))
                log_event("info", f"Camera: {cam_info['width']}x{cam_info['height']} @ {cam_info['fps']} fps")
            except Exception:
                pass
        return cap
    if cap is not None:
        cap.release()
    return None


# ------------------------------------------------ hard-sample collector (re-training data)
COLLECT_DIR = os.path.join(BASE_DIR, "dataset_candidates")
_last_collect = [0.0]


def collect_sample(raw, results):
    """Save the un-annotated frame + model predictions as YOLO labels (to be corrected in a labelling tool)."""
    try:
        now = time.time()
        if now - _last_collect[0] < 1.5:
            return
        img_dir, lbl_dir = os.path.join(COLLECT_DIR, "images"), os.path.join(COLLECT_DIR, "labels")
        os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)
        if len(os.listdir(img_dir)) >= 3000:
            return
        stem = time.strftime("%Y%m%d_%H%M%S") + f"_{int(now * 1000) % 1000:03d}"
        cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), raw)
        lines = []
        for r in results:
            for b in r.boxes:
                cx, cy, w, h = [float(v) for v in b.xywhn[0]]
                lines.append(f"{int(b.cls[0])} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
        _last_collect[0] = now
    except Exception as e:
        console.print(f"[red]Sample collect failed:[/red] {e}")


# ------------------------------------------------------------ display + detection
# Two independent threads so the video is SMOOTH even when YOLO is slow (CPU):
#   grabber  : reads camera/video at full speed, draws the latest detections, streams JPEG  (smooth video)
#   inferer  : runs YOLO on the newest frame only (never builds a backlog), updates counts/events/DB
DISPLAY_MAX_W = 1280
cam = {"frame": None, "seq": 0, "reset": True}
cam_lock = threading.Lock()
detst = {"boxes": [], "count": 0, "ref": None, "missing": 0, "status": "OK", "roi": None}
try:
    import torch
    USE_GPU = torch.cuda.is_available()
except Exception:
    USE_GPU = False


def draw_overlay(img, scale, d):
    """Draw boxes + status text. Same numbers as the dashboard because both come from `detst`."""
    for (x1, y1, x2, y2, label) in d["boxes"]:
        p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        cv2.putText(img, label, (p1[0], max(p1[1] - 10, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    h, w = img.shape[:2]
    if d.get("roi"):
        rx1, ry1, rx2, ry2 = d["roi"]
        cv2.rectangle(img, (int(rx1 * w), int(ry1 * h)), (int(rx2 * w), int(ry2 * h)), (255, 255, 255), 1)
        cv2.putText(img, "ZONE", (int(rx1 * w) + 6, int(ry1 * h) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"CURRENT: {d['count']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    if d["ref"] is not None:
        cv2.putText(img, f"REF: {d['ref']}  MISSING: {d['missing']}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    if settings["cycle"]["enabled"]:
        if cyc["state"] == "INSPECTING":
            left = max(settings["cycle"]["duration"] - (time.time() - cyc["t_start"]), 0)
            txt = f"INSPECTING  {left:0.1f}s"
        elif cyc["state"] == "RESULT" and cyc["result"]:
            r = cyc["result"]
            txt = f"RESULT: {r['verdict']}  ({r['count']}/{r['ref']})"
        else:
            txt = "WAITING FOR PART"
        cv2.putText(img, txt, (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)


def render_jpg(frame, quality=80):
    """Resize (if huge), draw overlay with the latest detections, encode. Never modifies `frame`."""
    H, W = frame.shape[:2]
    scale = min(1.0, DISPLAY_MAX_W / W)
    img = cv2.resize(frame, (int(W * scale), int(H * scale))) if scale < 1 else frame.copy()
    with lock:
        d = dict(detst)
    draw_overlay(img, scale, d)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return (buf.tobytes() if ok else None), img.shape[1], img.shape[0]


def grabber():
    cap, fails, period, fps_ema, last = None, 0, 0.0, 0.0, time.time()
    while True:
        # ---- (re)open source when requested ----
        if ctrl["switch"]:
            ctrl["switch"] = False
            if cap is not None:
                cap.release(); cap = None
            src = ctrl["source"]
            cap = open_capture(src)
            if cap is None and src == "webcam":
                log_event("alert", "Webcam not available - falling back to video file")
                ctrl["source"] = "video"
                cap = open_capture("video")
            if cap is None:
                log_event("alert", "Cannot open video source - check VIDEO_PATH in app.py")
                time.sleep(2); ctrl["switch"] = True
                continue
            native = cap.get(cv2.CAP_PROP_FPS) if ctrl["source"] == "video" else 0
            period = 1.0 / native if native and 1 < native < 120 else (0.04 if ctrl["source"] == "video" else 0.0)
            with cam_lock:
                cam["reset"] = True
            fails = 0
            log_event("info", "Source active: " + ctrl["source"])

        if not ctrl["running"]:                      # paused: keep last frame on screen
            time.sleep(0.03)
            continue

        if ctrl.get("cam_apply") and ctrl["source"] == "webcam":
            ctrl["cam_apply"] = False
            try:
                apply_camera_hw(cap)
            except Exception:
                pass

        t0 = time.time()
        success, frame = cap.read()
        if not success:
            fails += 1
            if ctrl["source"] == "video":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)      # loop the video
                with cam_lock:
                    cam["reset"] = True                  # fresh IDs for new loop
                if fails > 30:
                    log_event("alert", "Video file cannot be read")
                    ctrl["switch"] = True; fails = 0; time.sleep(1)
            else:
                time.sleep(0.03)
                if fails > 100:
                    log_event("alert", "Webcam stream lost - reconnecting")
                    ctrl["switch"] = True; fails = 0
            continue
        fails = 0

        n_frames = cam["seq"]
        if n_frames % 5 == 0:                                  # cheap exposure meter (every 5th frame)
            cam_stats["raw_mean"] = float(cv2.cvtColor(frame[::8, ::8], cv2.COLOR_BGR2GRAY).mean())
        frame = sw_adjust(frame)
        if n_frames % 5 == 0:
            g = cv2.cvtColor(frame[::8, ::8], cv2.COLOR_BGR2GRAY)
            cam_stats["mean"] = float(g.mean())
            cam_stats["clip_pct"] = float((g >= 250).mean() * 100)
        with cam_lock:
            cam["frame"] = frame
            cam["seq"] += 1
        jpg, w, h = render_jpg(frame)
        if jpg:
            with frame_cond:
                latest["jpg"] = jpg
                latest["seq"] += 1
                frame_cond.notify_all()
        now = time.time()
        fps_ema = 0.9 * fps_ema + 0.1 * (1 / max(now - last, 1e-3)) if fps_ema else 1 / max(now - last, 1e-3)
        last = now
        with lock:
            dashboard_data.update(fps=round(fps_ema, 1), frame_w=w, frame_h=h)
        if period:                                        # play video files at their real speed
            time.sleep(max(period - (time.time() - t0), 0))


def inferer():
    global last_log_time
    last_seq, last_t, det_ema = 0, time.time(), 0.0
    pending_status, pending_n, reported_status = "OK", 0, "OK"
    recent = deque(maxlen=15)

    while True:
        if not ctrl["running"]:
            time.sleep(0.05)
            if cyc["state"] == "INSPECTING":
                cyc["t_start"] += 0.05                  # timer does not run while paused
            continue
        with cam_lock:
            if cam["frame"] is None or cam["seq"] == last_seq:
                frame = None
            else:
                frame, last_seq, reset_tracker = cam["frame"], cam["seq"], cam["reset"]
                cam["reset"] = False
        if frame is None:
            time.sleep(0.004)
            continue
        if reset_tracker:
            recent.clear()

        # ---------------- YOLO ----------------
        det = settings["detection"]
        kw = dict(persist=not reset_tracker, verbose=False, conf=det["conf"], iou=det["iou"])
        if det.get("imgsz"):
            kw["imgsz"] = int(det["imgsz"])
        if det.get("device") is not None:
            kw["device"] = det["device"]
        elif USE_GPU:
            kw["device"], kw["half"] = 0, True
        results = model.track(frame, **kw)

        # ---------------- filter + collect boxes ----------------
        H, W = frame.shape[:2]
        roi = det.get("roi")
        amin, amax = det.get("min_box_area_pct", 0) / 100.0, det.get("max_box_area_pct", 100) / 100.0
        boxes, current_ids, confs = [], set(), []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = max(x2 - x1, 0) * max(y2 - y1, 0) / float(W * H)
                if area < amin or area > amax:
                    continue
                if roi:
                    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                    if not (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
                        continue
                if box.id is not None:
                    tid = int(box.id.item())
                    current_ids.add(tid)
                    label = f"ID {tid}"
                else:
                    label = "Bolt"
                boxes.append((x1, y1, x2, y2, label))
                confs.append(float(box.conf[0]))

        n_boxes = len(boxes)
        raw_count = n_boxes if det.get("count_all_boxes", True) else len(current_ids)
        recent.append(raw_count)
        k = max(1, int(det.get("count_smooth", 1)))
        win = sorted(list(recent)[-k:])
        current_count = win[len(win) // 2]              # median of last k detections = no flicker
        avg_conf = round(100 * sum(confs) / len(confs), 1) if confs else 0
        conf_count = len(confs)

        # ---- reference / missing logic ----
        with lock:
            ref = ctrl["ref"]
        if ref is None:
            ref_val, missing, status = current_count, 0, "OK"
        else:
            ref_val = ref
            missing = max(ref - current_count, 0)
            status = "ALERT" if missing > 0 else "OK"

        save_alert = None
        if status == pending_status:                    # event only on a stable change (5 detections)
            pending_n += 1
        else:
            pending_status, pending_n = status, 1
        if pending_n == 5 and pending_status != reported_status:
            reported_status = pending_status
            if settings["cycle"]["enabled"]:
                pass                                      # timed mode: verdicts replace live alerts
            elif status == "ALERT":
                log_event("alert", f"{missing} fastener(s) missing (reference {ref_val})")
                save_alert = (f"{missing} fastener(s) missing (reference {ref_val})", current_count, ref_val, missing)
            elif ref is not None:
                log_event("ok", "All fasteners present")
                record("ok", "All fasteners present", current_count, ref_val, 0)

        if det.get("collect_samples") and (status == "ALERT" or (0 < conf_count and avg_conf < 60)
                                          or n_boxes > len(current_ids)):
            collect_sample(frame, results)

        verdict = cycle_update(current_count, time.time())

        # ---- publish numbers + overlay data together (video & KPIs always agree) ----
        now = time.time()
        det_fps = 1 / max(now - last_t, 1e-3)
        det_ema = 0.85 * det_ema + 0.15 * det_fps if det_ema else det_fps
        last_t = now
        with lock:
            detst.update(boxes=boxes, count=current_count, ref=ref, missing=missing, status=status, roi=roi)
            dashboard_data.update(current_bolts=current_count, reference_bolts=ref_val, missing_bolts=missing,
                                  status=status, active_ids=sorted(current_ids), accuracy=avg_conf,
                                  det_fps=round(det_ema, 1))
            dashboard_data["frame_seq"] += 1

        if verdict or save_alert:                       # snapshot = the annotated frame the operator sees
            jpg, _, _ = render_jpg(frame, 90)
            if verdict:
                cycle_finish(verdict, jpg)
            if save_alert:
                record("alert", save_alert[0], save_alert[1], save_alert[2], save_alert[3], jpg)

        if time.time() - last_log_time > 0.5:
            console.print(
                f"[bold green]AI-VISION[/bold green] | [cyan]BOLTS:[/cyan] {current_count} | "
                f"[yellow]AI ACCURACY:[/yellow] {avg_conf}% | [red]MISSING:[/red] {missing} | "
                f"[blue]DETECT FPS:[/blue] {round(det_ema, 1)} | [magenta]STATUS:[/magenta] {status}"
            )
            last_log_time = time.time()


_worker_started = False


def start_worker():
    global _worker_started
    if not _worker_started:
        _worker_started = True
        threading.Thread(target=grabber, daemon=True).start()
        threading.Thread(target=inferer, daemon=True).start()


# -------------------------------------------------------------------- routes
def generate_frames():
    seq = 0
    while True:
        with frame_cond:
            frame_cond.wait_for(lambda: latest["seq"] != seq, timeout=1.0)
            if latest["seq"] == seq or latest["jpg"] is None:
                continue
            seq = latest["seq"]
            jpg = latest["jpg"]
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
        )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    resp = Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp


@app.route("/status")
def status():
    resp = jsonify(snapshot())
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/control/<action>", methods=["POST"])
def control(action):
    msg, code = None, 200
    with lock:
        if action == "start":
            if not ctrl["running"]:
                ctrl["running"] = True
                log_event("info", "Monitoring started")
        elif action == "stop":
            if ctrl["running"]:
                ctrl["running"] = False
                log_event("info", "Monitoring paused")
        elif action == "capture_reference":
            n = dashboard_data["current_bolts"]
            if n <= 0:
                msg, code = "No fasteners detected - cannot capture reference", 400
            else:
                ctrl["ref"] = n
                settings["active_part"] = None
                log_event("ok", f"Reference captured: {n} fasteners")
                record("reference", f"Reference captured: {n} fasteners", n, n, 0)
                msg = f"Reference captured: {n}"
        elif action == "clear_reference":
            ctrl["ref"] = None
            settings["active_part"] = None
            log_event("info", "Reference cleared")
        elif action == "source":
            src = (request.get_json(silent=True) or {}).get("source")
            if src not in ("video", "webcam"):
                msg, code = "Invalid source", 400
            elif src != ctrl["source"]:
                ctrl["source"] = src
                ctrl["switch"] = True
                msg = "Switching to " + src
        elif action == "cycle_mode":
            settings["cycle"]["enabled"] = bool((request.get_json(silent=True) or {}).get("enabled"))
            cyc.update(state="IDLE", present_since=None, absent_since=None)
            save_settings()
            msg = "Timed inspection ON" if settings["cycle"]["enabled"] else "Live mode"
            log_event("info", msg)
        elif action == "cycle_config":
            j = request.get_json(silent=True) or {}
            try:
                if "duration" in j:
                    settings["cycle"]["duration"] = max(3, min(int(j["duration"]), 300))
                if "min_present" in j:
                    settings["cycle"]["min_present"] = max(1, min(int(j["min_present"]), 100))
                if "grace" in j:
                    settings["cycle"]["grace"] = max(0.5, min(float(j["grace"]), 30))
                save_settings()
                msg = f"Inspection time: {settings['cycle']['duration']} s"
            except (TypeError, ValueError):
                msg, code = "Invalid value", 400
        elif action == "camera":
            j = request.get_json(silent=True) or {}
            cs = settings["camera"]
            try:
                reopen = hw = False
                for k, v in j.items():
                    if k in CAM_LIMITS:
                        lo, hi = CAM_LIMITS[k]
                        v = max(lo, min(float(v), hi))
                        v = int(v) if k in ("width", "height", "fps", "exposure", "focus", "sw_brightness") else round(v, 2)
                    elif k in ("auto_exposure", "autofocus", "sw_enhance"):
                        v = bool(v)
                    else:
                        continue
                    if cs.get(k) != v:
                        cs[k] = v
                        reopen |= k in ("width", "height", "fps")
                        hw |= k in ("auto_exposure", "exposure", "autofocus", "focus")
                if reopen and ctrl["source"] == "webcam":
                    ctrl["switch"] = True
                if hw:
                    ctrl["cam_apply"] = True
                save_settings()
                msg = "Camera updated"
            except (TypeError, ValueError):
                msg, code = "Invalid camera value", 400
        elif action == "camera_reset":
            settings["camera"].update(CAM_DEFAULTS)
            ctrl["cam_apply"] = True
            save_settings(); msg = "Camera reset"
        elif action == "camera_autotune":
            m = min(max(cam_stats["raw_mean"] / 255.0, 0.05), 0.95)
            g = round(float(min(max(np.log(0.45) / np.log(m), 0.4), 3.0)), 2)   # bring average brightness to ~45 %
            settings["camera"].update(sw_gamma=g, sw_brightness=0, sw_contrast=1.0)
            save_settings(); msg = f"Auto exposure fix applied (gamma {g})"
        elif action == "roi":
            j = request.get_json(silent=True) or {}
            if j.get("clear"):
                settings["detection"]["roi"] = None
                msg = "Detection zone cleared"
            else:
                try:
                    x1, y1, x2, y2 = [max(0.0, min(float(j[k]), 1.0)) for k in ("x1", "y1", "x2", "y2")]
                    x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
                    if x2 - x1 < 0.03 or y2 - y1 < 0.03:
                        raise ValueError
                    settings["detection"]["roi"] = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]
                    msg = "Detection zone set"
                except (KeyError, TypeError, ValueError):
                    msg, code = "Draw a larger zone", 400
            if code == 200:
                save_settings(); log_event("info", msg)
        elif action == "part_save":
            name = str((request.get_json(silent=True) or {}).get("name", "")).strip()[:40]
            n = ctrl["ref"] if ctrl["ref"] is not None else None
            if not name:
                msg, code = "Enter a part name", 400
            elif n is None:
                msg, code = "Capture a reference first, then save it as a part", 400
            else:
                settings["parts"][name] = n
                settings["active_part"] = name
                save_settings()
                log_event("ok", f"Part saved: {name} = {n} fasteners")
                msg = f"Saved part {name} ({n})"
        elif action == "part_select":
            name = (request.get_json(silent=True) or {}).get("name") or None
            if name is None:
                settings["active_part"] = None
                save_settings()
                msg = "No part selected"
            elif name in settings["parts"]:
                settings["active_part"] = name
                ctrl["ref"] = settings["parts"][name]
                save_settings()
                log_event("ok", f"Part selected: {name} (reference {ctrl['ref']})")
                msg = f"Part {name}: reference {ctrl['ref']}"
            else:
                msg, code = "Unknown part", 404
        elif action == "part_delete":
            name = (request.get_json(silent=True) or {}).get("name")
            if settings["parts"].pop(name, None) is not None:
                if settings["active_part"] == name:
                    settings["active_part"] = None
                save_settings()
                msg = f"Deleted part {name}"
            else:
                msg, code = "Unknown part", 404
        else:
            msg, code = "Unknown action", 404
    out = snapshot()
    out["ok"] = code == 200
    out["message"] = msg
    return jsonify(out), code

@app.route("/history")
def history_page():
    return render_template("history.html")


@app.route("/api/history")
def api_history():
    return jsonify(query_history(request.args.get("type"),
                                 min(int(request.args.get("limit", 200)), 1000)))


@app.route("/history/export.csv")
def history_csv():
    rows = query_history(request.args.get("type"), 100000)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["ID", "Time", "Type", "Source", "Current", "Reference", "Missing", "Message", "Snapshot", "Part"])
    for r in rows:
        w.writerow([r["id"], r["ts"], r["type"], r["source"], r["current_count"],
                    r["reference_count"], r["missing_count"], r["message"], r["snapshot"] or "", r.get("part") or ""])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=inspection_history.csv"})


@app.route("/snapshots/<path:name>")
def snapshot_file(name):
    return send_from_directory(SNAP_DIR, name)


@app.route("/api/health")
def health():
    d = snapshot()
    return jsonify(ok=True, detection=settings["detection"], running=d["running"], source=d["source"], fps=d["fps"],
                   frames=d["frame_seq"], model=os.path.relpath(MODEL_PATH, BASE_DIR))


def create_dashboard(current, missing, fps, status):

    table = Table(title="AI INSPECTION MONITOR")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Detected Bolts", str(current))
    table.add_row("Missing Bolts", str(missing))
    table.add_row("FPS", str(fps))
    table.add_row("Status", status)

    return table

if __name__ == "__main__":

    with Progress(
        SpinnerColumn(style="green"),
        TextColumn("[bold cyan]{task.description}"),
    ) as progress:

        task = progress.add_task("Loading YOLO Model...", total=None)
        time.sleep(0.5)

        progress.update(task, description="Initializing Tracking Engine...")
        time.sleep(0.5)

        progress.update(task, description="Connecting Video Source...")
        time.sleep(0.5)

        progress.update(task, description="Starting AI Inspection System...")
        time.sleep(0.5)

    console.print("[bold green]✓ AI SYSTEM READY[/bold green]")
    console.print("[bold yellow]Dashboard:[/bold yellow] http://127.0.0.1:5000")

    load_settings()
    db_init()
    start_worker()

    app.run(
        host="0.0.0.0", # <--- change it to 0 for multiple monitor or for normal usage use (0.0.0.0) or (127.0.0.0)
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )