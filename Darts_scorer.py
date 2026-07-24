#!/usr/bin/env python3
"""
dart_scorer.py — single-camera dart scorer for Raspberry Pi (Debian)
======================================================================

One self-contained script: point a USB webcam at your dartboard, run
this, and open the printed URL from any phone/laptop on your network
to calibrate the board and keep score for a game of 301/501/701.

WHAT'S IN HERE (all in one file, so it's easy to copy to a Pi):
  - DartBoard    : dartboard geometry + pixel->score mapping (homography)
  - DartDetector : webcam capture thread + frame-diff dart detection
  - Game/Player/Dart : simple X01 scoring rules
  - Flask app    : web dashboard, calibration UI, MJPEG video stream
  - INDEX_HTML / CALIBRATE_HTML : the two page templates, as strings

SETUP (Debian / Raspberry Pi OS):
    sudo apt-get update
    sudo apt-get install -y python3-opencv python3-flask python3-numpy
    python3 dart_scorer.py

  (apt's python3-opencv is used deliberately - pip has no prebuilt
  OpenCV wheel for the Pi's 32-bit ARM, so `pip install opencv-python`
  can take hours to compile from source on a Pi 3. If your Debian
  apt packages are too old to have python3-opencv/python3-flask,
  fall back to a venv:
      python3 -m venv --system-site-packages venv
      source venv/bin/activate
      pip install flask opencv-python-headless numpy
  )

RUNNING:
    python3 dart_scorer.py
    -> open http://<this machine's IP>:5000 from your phone or laptop
       (find the IP on the Pi with: hostname -I)

FIRST-TIME CALIBRATION (from the web dashboard):
  1. Click "Calibrate board".
  2. Click "Refresh snapshot".
  3. Click the outer edge of the double wire at 4 points, in order:
     the board's own top, right, bottom, left (as physically mounted -
     "top" is directly above the bullseye, regardless of which number
     is painted there).
  4. Click "Save calibration".
  Recalibrate any time the camera or board moves.

PLAYING:
  - "New game": add player names, pick 301/501/701, optional
    double-out.
  - Before the first throw, and after pulling darts out between
    turns, click "Darts pulled - reset board" so the camera treats
    the current (empty) board as the new baseline.
  - Each detected dart appears with its score - tap it to correct it
    (tap the true landing spot on its snapshot) before that turn ends.
  - "Manual entry" is there for any throw the camera misses entirely.

KNOWN LIMITATIONS - please read before relying on this:
  A single ordinary camera cannot see depth, so it cannot always tell
  a dart's tip apart from its shaft/flight, especially when two darts
  overlap from the camera's point of view, or near a ring/segment
  boundary. Lighting changes can also occasionally trigger a false
  detection. This is why every detected dart is shown with its
  snapshot for a quick correction, and why manual entry exists as a
  fallback - treat this as a scoring aid to keep an eye on, not a
  fully hands-off referee.

TUNING DETECTION:
  See the DartDetector(...) constructor call further down (in the
  "Flask app" section) for motion_threshold, motion_pixels_start/idle,
  settle_frames, min_dart_area/max_dart_area if you're getting too
  many false triggers or missed darts.

PERFORMANCE ON A PI 3:
  Defaults are kept at 640x480 to stay smooth on a Pi 3. If the video
  feels laggy, lower frame_width/frame_height in config.json (created
  next to this script after your first run), or increase the
  time.sleep() in _mjpeg_generator() below.

OPTIONAL - run automatically on boot (systemd):
  Create /etc/systemd/system/dart-scorer.service with:

    [Unit]
    Description=Dart Scorer web dashboard
    After=network.target

    [Service]
    WorkingDirectory=/home/pi/dart_scorer
    ExecStart=/usr/bin/python3 /home/pi/dart_scorer/dart_scorer.py
    Restart=on-failure
    User=pi

    [Install]
    WantedBy=multi-user.target

  Then:  sudo systemctl enable --now dart-scorer
"""

import json
import math
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


# ----------------------------------------------------------------------
# Dartboard geometry & scoring
# ----------------------------------------------------------------------



# --- Standard dartboard dimensions (mm), per WDF/steel-tip regulations ---
BULL_INNER_R = 6.35      # double bull (50)
BULL_OUTER_R = 15.9      # single bull (25)
TRIPLE_INNER_R = 99.0
TRIPLE_OUTER_R = 107.0
DOUBLE_INNER_R = 162.0
DOUBLE_OUTER_R = 170.0   # outer edge of the wire - the calibration radius

# Segment numbers, clockwise starting from the top (12 o'clock) segment.
SEGMENTS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8,
            11, 14, 9, 12, 5]

# The four physical calibration targets, in board-plane mm coordinates.
# Order matters: it must match the order the user is asked to click in.
CALIBRATION_TARGETS_MM = [
    (0.0, DOUBLE_OUTER_R),    # Top
    (DOUBLE_OUTER_R, 0.0),    # Right
    (0.0, -DOUBLE_OUTER_R),   # Bottom
    (-DOUBLE_OUTER_R, 0.0),   # Left
]



class CalibrationError(Exception):
    pass


class DartBoard:
    """Holds calibration state and converts pixel points to scores."""

    def __init__(self):
        self.homography = None  # 3x3 np.ndarray, pixel -> mm
        self.camera_index = 0
        # Kept modest by default - a Pi 3 struggles to decode/encode much
        # more than this in real time alongside Flask and OpenCV. Raise
        # it in config.json if you're running on something faster.
        self.frame_width = 640
        self.frame_height = 480
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self):
        if not os.path.exists(CONFIG_PATH):
            return
        try:
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
            if data.get("homography"):
                self.homography = np.array(data["homography"], dtype=np.float64)
            self.camera_index = data.get("camera_index", self.camera_index)
            self.frame_width = data.get("frame_width", self.frame_width)
            self.frame_height = data.get("frame_height", self.frame_height)
        except (json.JSONDecodeError, OSError):
            pass

    def save(self):
        data = {
            "homography": self.homography.tolist() if self.homography is not None else None,
            "camera_index": self.camera_index,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def is_calibrated(self):
        return self.homography is not None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(self, pixel_points):
        """
        pixel_points: list of 4 (x, y) pixel coordinates, in the order
        Top, Right, Bottom, Left (outer edge of the double wire).
        """
        if len(pixel_points) != 4:
            raise CalibrationError("Need exactly 4 calibration points.")

        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(CALIBRATION_TARGETS_MM, dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        self.homography = H
        self.save()

    def pixel_to_mm(self, px, py):
        if self.homography is None:
            raise CalibrationError("Board is not calibrated yet.")
        pt = np.array([[[px, py]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.homography.astype(np.float32))
        return float(out[0][0][0]), float(out[0][0][1])

    def mm_to_pixel(self, x_mm, y_mm):
        """Inverse mapping - useful for drawing the board outline overlay."""
        if self.homography is None:
            raise CalibrationError("Board is not calibrated yet.")
        H_inv = np.linalg.inv(self.homography.astype(np.float64))
        pt = np.array([[[x_mm, y_mm]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, H_inv.astype(np.float32))
        return float(out[0][0][0]), float(out[0][0][1])

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score_at_mm(self, x, y):
        """
        Returns (points, label, number) for a board-plane point in mm.
        label examples: "D20", "T19", "BULL", "25", "MISS"
        """
        r = math.hypot(x, y)

        if r > DOUBLE_OUTER_R:
            return 0, "MISS", None

        if r <= BULL_INNER_R:
            return 50, "BULL", None
        if r <= BULL_OUTER_R:
            return 25, "25", None

        angle = math.degrees(math.atan2(x, y)) % 360
        index = int(((angle + 9) % 360) // 18)
        number = SEGMENTS[index]

        if TRIPLE_INNER_R <= r <= TRIPLE_OUTER_R:
            return number * 3, f"T{number}", number
        if DOUBLE_INNER_R <= r <= DOUBLE_OUTER_R:
            return number * 2, f"D{number}", number
        return number, f"{number}", number

    def score_at_pixel(self, px, py):
        x, y = self.pixel_to_mm(px, py)
        return self.score_at_mm(x, y)

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------
    def outline_points_px(self, n=72):
        """Points (in pixel space) tracing the double-ring outer edge,
        for drawing a board outline overlay on the video feed."""
        if self.homography is None:
            return []
        pts = []
        for i in range(n):
            ang = 2 * math.pi * i / n
            x = DOUBLE_OUTER_R * math.sin(ang)
            y = DOUBLE_OUTER_R * math.cos(ang)
            pts.append(self.mm_to_pixel(x, y))
        return pts



# ----------------------------------------------------------------------
# Camera capture + dart detection
# ----------------------------------------------------------------------




class DartDetector:
    def __init__(self, camera_index=0, width=640, height=480,
                 motion_threshold=25, motion_pixels_start=1500,
                 motion_pixels_idle=250, settle_frames=8,
                 min_dart_area=60, max_dart_area=9000):
        self.camera_index = camera_index
        self.width = width
        self.height = height

        # Motion detection tuning
        self.motion_threshold = motion_threshold      # per-pixel diff threshold
        self.motion_pixels_start = motion_pixels_start  # changed-pixel count to declare "motion"
        self.motion_pixels_idle = motion_pixels_idle     # below this = considered still
        self.settle_frames = settle_frames               # consecutive still frames before capturing
        self.min_dart_area = min_dart_area
        self.max_dart_area = max_dart_area

        self.cap = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        self.latest_frame = None           # most recent raw BGR frame (for streaming)
        self.baseline_gray = None          # accepted "no new dart" reference frame
        self.state = "IDLE"
        self.still_count = 0
        self.recent_gray = deque(maxlen=5)

        self.on_dart_detected = None       # callback(px, py, debug_frame)
        self.paused = False                # ignore new detections while paused

    # ------------------------------------------------------------------
    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.camera_index}. "
                "Check it's plugged in and not in use by another program."
            )
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()

    def resync_baseline(self):
        """Call this after darts are pulled out of the board, or at
        startup, so the detector stops treating the current board state
        as 'new'."""
        with self.lock:
            if self.latest_frame is not None:
                gray = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (9, 9), 0)
                self.baseline_gray = gray
                self.state = "IDLE"
                self.still_count = 0

    def get_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    # ------------------------------------------------------------------
    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            with self.lock:
                self.latest_frame = frame

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (9, 9), 0)

            if self.baseline_gray is None:
                self.baseline_gray = gray
                continue

            if self.paused:
                continue

            self._process(gray, frame)

        # loop end

    def _process(self, gray, frame):
        prev = self.recent_gray[-1] if self.recent_gray else gray
        self.recent_gray.append(gray)

        frame_diff = cv2.absdiff(gray, prev)
        _, moving_mask = cv2.threshold(frame_diff, self.motion_threshold, 255,
                                        cv2.THRESH_BINARY)
        moving_pixels = int(np.count_nonzero(moving_mask))

        if self.state == "IDLE":
            if moving_pixels > self.motion_pixels_start:
                self.state = "MOTION"
                self.still_count = 0

        elif self.state == "MOTION":
            if moving_pixels < self.motion_pixels_idle:
                self.still_count += 1
                if self.still_count >= self.settle_frames:
                    self._evaluate_settled_frame(gray, frame)
                    self.state = "IDLE"
                    self.still_count = 0
            else:
                self.still_count = 0

    def _evaluate_settled_frame(self, settled_gray, settled_frame):
        baseline_diff = cv2.absdiff(settled_gray, self.baseline_gray)
        _, mask = cv2.threshold(baseline_diff, self.motion_threshold, 255,
                                 cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                 np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                 np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # Nothing dart-sized changed (e.g. lighting flicker) - resync
            # quietly so small drift doesn't accumulate.
            self.baseline_gray = settled_gray
            return

        candidate = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(candidate)

        if area < self.min_dart_area:
            # Too small - noise. Ignore, keep old baseline.
            return

        if area > self.max_dart_area:
            # Too big to be a single dart (a hand, body, lighting change).
            # Don't guess - just resync so we don't get stuck. The
            # dashboard's "Reset board" button is the normal way to do
            # this, but this keeps things self-healing.
            self.baseline_gray = settled_gray
            return

        tip = self._estimate_tip(mask, candidate)

        # New baseline includes this dart, so the next dart is detected
        # relative to a board that now has this one sticking in it.
        self.baseline_gray = settled_gray

        if self.on_dart_detected:
            self.on_dart_detected(tip[0], tip[1], settled_frame.copy())

    @staticmethod
    def _estimate_tip(mask, contour):
        """Heuristic: the dart tip end of the blob tends to be the
        narrower end (buried in the board), while the flight/shaft end
        is bushier. We find the two most distant points on the convex
        hull and pick whichever has less mask density around it."""
        hull = cv2.convexHull(contour).reshape(-1, 2)

        best_pair = None
        best_dist = -1
        for i in range(len(hull)):
            for j in range(i + 1, len(hull)):
                d = np.hypot(*(hull[i] - hull[j]))
                if d > best_dist:
                    best_dist = d
                    best_pair = (hull[i], hull[j])

        if best_pair is None:
            m = cv2.moments(contour)
            if m["m00"] == 0:
                x, y, w, h = cv2.boundingRect(contour)
                return (x + w / 2, y + h / 2)
            return (m["m10"] / m["m00"], m["m01"] / m["m00"])

        def density(p, r=10):
            x, y = int(p[0]), int(p[1])
            y0, y1 = max(0, y - r), min(mask.shape[0], y + r)
            x0, x1 = max(0, x - r), min(mask.shape[1], x + r)
            patch = mask[y0:y1, x0:x1]
            return patch.mean() if patch.size else 0

        p1, p2 = best_pair
        tip = p1 if density(p1) < density(p2) else p2
        return (float(tip[0]), float(tip[1]))



# ----------------------------------------------------------------------
# X01 game logic
# ----------------------------------------------------------------------



class Player:
    def __init__(self, name, start_score):
        self.name = name
        self.score = start_score
        self.history = []  # list of completed turns: [{"darts": [...], "busted": bool}]


class Dart:
    def __init__(self, points, label, px, py, manual=False, dart_id=None):
        self.points = points
        self.label = label
        self.px = px
        self.py = py
        self.manual = manual
        self.dart_id = dart_id
        self.timestamp = time.time()


class Game:
    def __init__(self, player_names, start_score=501, double_out=False):
        if not player_names:
            player_names = ["Player 1"]
        self.start_score = start_score
        self.double_out = double_out
        self.players = [Player(n, start_score) for n in player_names]
        self.current_player_idx = 0
        self.current_turn_darts = []  # list[Dart] for the in-progress turn
        self.finished = False
        self.winner = None
        self.log = []  # human-readable event log, most recent last

    # ------------------------------------------------------------------
    @property
    def current_player(self):
        return self.players[self.current_player_idx]

    def darts_thrown_this_turn(self):
        return len(self.current_turn_darts)

    def turn_complete(self):
        return len(self.current_turn_darts) >= 3

    # ------------------------------------------------------------------
    def add_dart(self, points, label, px=None, py=None, manual=False, dart_id=None):
        if self.finished:
            return
        if self.turn_complete():
            return
        dart = Dart(points, label, px, py, manual, dart_id)
        self.current_turn_darts.append(dart)
        self.log.append(f"{self.current_player.name}: {label} ({points})")

        remaining = self.current_player.score - self._turn_total()
        is_last_dart_of_turn = self.turn_complete()

        busted = False
        won = False
        if remaining < 0:
            busted = True
        elif remaining == 0:
            if self.double_out and not label.startswith("D") and label != "BULL":
                busted = True
            else:
                won = True
        elif remaining == 1 and self.double_out:
            # Can never finish on 1 with a double - treat as bust once the
            # turn ends, but let the player keep throwing their remaining
            # darts (matches typical league play) unless this dart was
            # already the 3rd.
            busted = is_last_dart_of_turn

        if won:
            self.current_player.score = 0
            self.finished = True
            self.winner = self.current_player
            self.log.append(f"{self.current_player.name} wins!")
            return

        if busted:
            self.log.append(f"{self.current_player.name} busts - score stays at "
                             f"{self.current_player.score}")
            self._end_turn(apply_score=False)
            return

        if is_last_dart_of_turn:
            self._end_turn(apply_score=True)

    def correct_dart(self, dart_id, points, label):
        """Update a dart that's still part of the in-progress turn (i.e.
        its turn hasn't been scored/busted yet). Returns False if the
        dart_id can't be found there any more - most likely because its
        turn already ended, which this simple scorer doesn't support
        editing after the fact."""
        for d in self.current_turn_darts:
            if d.dart_id == dart_id:
                d.points = points
                d.label = label
                self.log.append(f"Correction: dart set to {label} ({points})")

                remaining = self.current_player.score - self._turn_total()
                if remaining < 0:
                    self.log.append(f"{self.current_player.name} busts - "
                                     f"score stays at {self.current_player.score}")
                    self._end_turn(apply_score=False)
                elif remaining == 0 and (not self.double_out
                                          or label.startswith("D") or label == "BULL"):
                    self.current_player.score = 0
                    self.finished = True
                    self.winner = self.current_player
                    self.log.append(f"{self.current_player.name} wins!")
                elif self.turn_complete():
                    self._end_turn(apply_score=True)
                return True
        return False

    def _turn_total(self):
        return sum(d.points for d in self.current_turn_darts)

    def undo_last_dart(self):
        if self.current_turn_darts:
            removed = self.current_turn_darts.pop()
            self.log.append(f"Undo: removed {removed.label}")
            return True
        return False

    def _end_turn(self, apply_score):
        if apply_score:
            self.current_player.score -= self._turn_total()
        self.current_player.history.append({
            "darts": [d.label for d in self.current_turn_darts],
            "busted": not apply_score,
        })
        self.current_turn_darts = []
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)

    def force_end_turn(self):
        """Manually move to the next player without waiting for 3 darts
        (e.g. detection missed a dart and the player wants to move on)."""
        if not self.current_turn_darts:
            self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
            return
        self._end_turn(apply_score=True)

    def state(self):
        turn_total = self._turn_total()
        return {
            "players": [
                {
                    "name": p.name,
                    "score": p.score,
                    "history": p.history,
                    # Live "remaining" for the player currently throwing,
                    # so the scoreboard ticks down dart-by-dart rather than
                    # only jumping once the turn ends (score isn't
                    # committed until then, since a bust reverts it).
                    "display_score": (p.score - turn_total
                                       if i == self.current_player_idx and not self.finished
                                       else p.score),
                }
                for i, p in enumerate(self.players)
            ],
            "current_player_idx": self.current_player_idx,
            "current_turn_darts": [
                {"points": d.points, "label": d.label, "manual": d.manual}
                for d in self.current_turn_darts
            ],
            "finished": self.finished,
            "winner": self.winner.name if self.winner else None,
            "double_out": self.double_out,
            "start_score": self.start_score,
            "log": self.log[-25:],
        }



# ----------------------------------------------------------------------
# HTML templates
# ----------------------------------------------------------------------

INDEX_HTML = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>Oche &middot; Dart Scorer</title>\n<link rel="preconnect" href="https://fonts.googleapis.com">\n<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">\n<style>\n  :root{\n    --bg:#12140f;\n    --panel:#1b1f16;\n    --panel-2:#20251a;\n    --wire:#8a7a4a;\n    --brass:#c9a24b;\n    --chalk:#f2ede1;\n    --muted:#9a9d8a;\n    --red:#b3392b;\n    --red-dim:#5c2119;\n    --green:#3c7a4f;\n    --green-dim:#1f3d29;\n    --radius:10px;\n  }\n  *{box-sizing:border-box;}\n  body{\n    margin:0; background:var(--bg); color:var(--chalk);\n    font-family:\'Inter\',sans-serif;\n    background-image:\n      radial-gradient(circle at 15% 0%, rgba(60,122,79,0.08), transparent 40%),\n      radial-gradient(circle at 85% 100%, rgba(179,57,43,0.07), transparent 40%);\n  }\n  header{\n    display:flex; align-items:center; justify-content:space-between;\n    padding:18px 22px; border-bottom:1px solid #2a2f21;\n  }\n  header .brand{ display:flex; align-items:center; gap:10px; }\n  header .brand .dot{ width:10px; height:10px; border-radius:50%; background:var(--green); box-shadow:0 0 8px var(--green);}\n  header .brand .dot.off{ background:var(--red); box-shadow:0 0 8px var(--red);}\n  header h1{\n    font-family:\'Oswald\',sans-serif; font-weight:600; letter-spacing:0.06em;\n    text-transform:uppercase; font-size:1.3rem; margin:0;\n  }\n  header a{ color:var(--muted); text-decoration:none; font-size:0.85rem; border:1px solid #33392a; padding:8px 12px; border-radius:8px;}\n  header a:hover{ color:var(--chalk); border-color:var(--brass); }\n\n  main{ display:grid; grid-template-columns: 1.15fr 1fr; gap:18px; padding:18px 22px 40px; max-width:1300px; margin:0 auto; }\n  @media (max-width:880px){ main{ grid-template-columns:1fr; } }\n\n  .panel{ background:var(--panel); border:1px solid #262c1c; border-radius:var(--radius); overflow:hidden; }\n  .panel-title{\n    font-family:\'Oswald\',sans-serif; text-transform:uppercase; letter-spacing:0.08em;\n    font-size:0.78rem; color:var(--brass); padding:12px 16px; border-bottom:1px solid #262c1c;\n  }\n\n  .video-wrap{ position:relative; background:#000; }\n  .video-wrap img{ display:block; width:100%; height:auto; }\n  .video-controls{ display:flex; gap:10px; padding:12px 16px; flex-wrap:wrap; }\n  .cam-error{ padding:12px 16px; color:#e8a49c; font-size:0.85rem; }\n\n  button, .btn{\n    font-family:\'Inter\',sans-serif; font-weight:600; font-size:0.85rem;\n    background:var(--panel-2); color:var(--chalk); border:1px solid #33392a;\n    padding:9px 14px; border-radius:8px; cursor:pointer;\n  }\n  button:hover{ border-color:var(--brass); }\n  button.primary{ background:var(--green-dim); border-color:var(--green); }\n  button.danger{ background:var(--red-dim); border-color:var(--red); }\n  button:disabled{ opacity:0.4; cursor:not-allowed; }\n\n  .players{ display:flex; flex-direction:column; }\n  .player-card{\n    display:flex; align-items:center; justify-content:space-between;\n    padding:16px; border-bottom:1px solid #23281a; position:relative;\n  }\n  .player-card.active{ background:linear-gradient(90deg, rgba(201,162,75,0.10), transparent); }\n  .player-card.active::before{\n    content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--brass);\n  }\n  .player-name{ font-weight:600; font-size:0.95rem; }\n  .player-sub{ color:var(--muted); font-size:0.75rem; margin-top:2px; }\n  .player-score{\n    font-family:\'Oswald\',sans-serif; font-size:2.4rem; font-weight:700;\n    letter-spacing:0.02em; min-width:110px; text-align:right;\n  }\n  .player-card.winner .player-score{ color:var(--brass); }\n\n  .turn-strip{ display:flex; gap:10px; padding:14px 16px; border-bottom:1px solid #23281a; }\n  .dart-slot{\n    flex:1; min-height:76px; border:1px dashed #33392a; border-radius:8px;\n    display:flex; flex-direction:column; align-items:center; justify-content:center;\n    font-family:\'JetBrains Mono\',monospace; gap:4px; padding:6px; text-align:center;\n  }\n  .dart-slot.filled{ border-style:solid; border-color:var(--wire); background:var(--panel-2); cursor:pointer; }\n  .dart-slot .label{ font-size:1.05rem; font-weight:600; color:var(--chalk); }\n  .dart-slot .pts{ font-size:0.7rem; color:var(--muted); }\n  .dart-slot .manual-tag{ font-size:0.6rem; color:var(--brass); }\n\n  .controls-row{ display:flex; gap:10px; padding:12px 16px; flex-wrap:wrap; border-bottom:1px solid #23281a; }\n\n  .manual-entry{ padding:14px 16px; border-bottom:1px solid #23281a; }\n  .manual-grid{ display:grid; grid-template-columns:repeat(7,1fr); gap:6px; margin-top:10px; }\n  .manual-grid button{ padding:8px 0; font-family:\'JetBrains Mono\',monospace; }\n  .mult-row{ display:flex; gap:8px; margin-top:10px; }\n  .mult-row button.active{ background:var(--green-dim); border-color:var(--green); }\n\n  .log{ padding:14px 16px; font-family:\'JetBrains Mono\',monospace; font-size:0.78rem; color:var(--muted);\n        max-height:180px; overflow-y:auto; }\n  .log div{ padding:2px 0; }\n\n  .new-game{ padding:16px; }\n  .new-game input[type=text]{\n    width:100%; padding:9px 10px; margin:6px 0; background:var(--panel-2); border:1px solid #33392a;\n    border-radius:8px; color:var(--chalk); font-family:\'Inter\',sans-serif;\n  }\n  .new-game label{ font-size:0.8rem; color:var(--muted); display:block; margin-top:10px;}\n  .row-inline{ display:flex; align-items:center; gap:8px; margin-top:8px; }\n\n  .modal-backdrop{\n    position:fixed; inset:0; background:rgba(0,0,0,0.7); display:none;\n    align-items:center; justify-content:center; z-index:50; padding:16px;\n  }\n  .modal-backdrop.open{ display:flex; }\n  .modal{ background:var(--panel); border:1px solid var(--brass); border-radius:12px; padding:16px; max-width:520px; width:100%; }\n  .modal h3{ font-family:\'Oswald\',sans-serif; margin:0 0 10px; text-transform:uppercase; letter-spacing:0.05em; font-size:1rem;}\n  .modal img{ width:100%; border-radius:8px; cursor:crosshair; border:1px solid #33392a;}\n  .modal .hint{ color:var(--muted); font-size:0.8rem; margin-top:8px; }\n  .modal .modal-actions{ display:flex; justify-content:flex-end; gap:8px; margin-top:14px; }\n\n  .winner-banner{\n    margin:16px; padding:16px; text-align:center; border:1px solid var(--brass); border-radius:10px;\n    background:linear-gradient(180deg, rgba(201,162,75,0.1), transparent);\n    font-family:\'Oswald\',sans-serif; text-transform:uppercase; letter-spacing:0.06em;\n  }\n</style>\n</head>\n<body>\n\n<header>\n  <div class="brand">\n    <span class="dot" id="cam-dot"></span>\n    <h1>Oche</h1>\n  </div>\n  <a href="/calibrate">Calibrate board</a>\n</header>\n\n<main>\n  <section class="panel">\n    <div class="panel-title">Live view</div>\n    <div class="video-wrap">\n      <img src="/video_feed" alt="Camera feed">\n    </div>\n    <div id="cam-error" class="cam-error" style="display:none;"></div>\n    <div class="video-controls">\n      <button onclick="resetBoard()">Darts pulled &mdash; reset board</button>\n      <button onclick="endTurn()">Force end turn</button>\n      <button onclick="undoDart()">Undo last dart</button>\n    </div>\n  </section>\n\n  <section>\n    <div class="panel" style="margin-bottom:18px;">\n      <div class="panel-title">Scoreboard</div>\n      <div id="players" class="players"></div>\n      <div id="turn-strip" class="turn-strip"></div>\n      <div class="controls-row">\n        <button onclick="showNewGame()">New game</button>\n      </div>\n      <div class="manual-entry" id="manual-entry" style="display:none;">\n        <div class="panel-title" style="padding:0 0 8px; border:none;">Manual entry (if a throw isn\'t picked up)</div>\n        <div class="mult-row">\n          <button id="mult-1" class="active" onclick="setMult(1)">Single</button>\n          <button id="mult-2" onclick="setMult(2)">Double</button>\n          <button id="mult-3" onclick="setMult(3)">Triple</button>\n        </div>\n        <div class="manual-grid" id="manual-grid"></div>\n        <div class="mult-row" style="margin-top:10px;">\n          <button onclick="manualSpecial(25,\'25\')">25</button>\n          <button onclick="manualSpecial(50,\'BULL\')">Bull (50)</button>\n          <button onclick="manualSpecial(0,\'MISS\')">Miss</button>\n        </div>\n      </div>\n      <div id="log" class="log"></div>\n    </div>\n  </section>\n</main>\n\n<div class="modal-backdrop" id="new-game-modal">\n  <div class="modal">\n    <h3>New game</h3>\n    <div id="player-inputs">\n      <input type="text" class="p-name" placeholder="Player 1 name" value="Player 1">\n    </div>\n    <button onclick="addPlayerInput()">+ Add player</button>\n    <label>Starting score</label>\n    <select id="start-score" style="width:100%; padding:9px; background:var(--panel-2); color:var(--chalk); border:1px solid #33392a; border-radius:8px;">\n      <option value="501" selected>501</option>\n      <option value="301">301</option>\n      <option value="701">701</option>\n    </select>\n    <div class="row-inline">\n      <input type="checkbox" id="double-out">\n      <label style="margin:0;">Require double to finish</label>\n    </div>\n    <div class="modal-actions">\n      <button onclick="closeNewGame()">Cancel</button>\n      <button class="primary" onclick="startNewGame()">Start</button>\n    </div>\n  </div>\n</div>\n\n<div class="modal-backdrop" id="correct-modal">\n  <div class="modal">\n    <h3>Correct this dart</h3>\n    <img id="correct-img" src="" alt="Dart snapshot">\n    <div class="hint">Tap the point where the dart actually landed.</div>\n    <div class="modal-actions">\n      <button onclick="closeCorrect()">Cancel</button>\n    </div>\n  </div>\n</div>\n\n<script>\nlet currentMult = 1;\nlet correctingDartId = null;\n\nfunction buildManualGrid(){\n  const grid = document.getElementById(\'manual-grid\');\n  const numbers = [20,1,18,4,13,6,10,15,2,17,3,19,7,16,8,11,14,9,12,5];\n  numbers.sort((a,b)=>a-b);\n  grid.innerHTML = numbers.map(n => `<button onclick="manualNumber(${n})">${n}</button>`).join(\'\');\n}\nbuildManualGrid();\n\nfunction setMult(m){\n  currentMult = m;\n  [1,2,3].forEach(i => document.getElementById(\'mult-\'+i).classList.toggle(\'active\', i===m));\n}\n\nasync function manualNumber(n){\n  const points = n * currentMult;\n  const label = currentMult===1 ? `${n}` : (currentMult===2 ? `D${n}` : `T${n}`);\n  await postJSON(\'/api/manual_dart\', {points, label});\n  refresh();\n}\nasync function manualSpecial(points, label){\n  await postJSON(\'/api/manual_dart\', {points, label});\n  refresh();\n}\n\nasync function postJSON(url, body){\n  const res = await fetch(url, {method:\'POST\', headers:{\'Content-Type\':\'application/json\'}, body: JSON.stringify(body||{})});\n  return res.json();\n}\n\nasync function resetBoard(){ await postJSON(\'/api/reset_board\'); }\nasync function endTurn(){ await postJSON(\'/api/end_turn\'); refresh(); }\nasync function undoDart(){ await postJSON(\'/api/undo\'); refresh(); }\n\nfunction showNewGame(){ document.getElementById(\'new-game-modal\').classList.add(\'open\'); }\nfunction closeNewGame(){ document.getElementById(\'new-game-modal\').classList.remove(\'open\'); }\nfunction addPlayerInput(){\n  const wrap = document.getElementById(\'player-inputs\');\n  const n = wrap.children.length + 1;\n  const inp = document.createElement(\'input\');\n  inp.type=\'text\'; inp.className=\'p-name\'; inp.placeholder=`Player ${n} name`; inp.value=`Player ${n}`;\n  wrap.appendChild(inp);\n}\nasync function startNewGame(){\n  const names = Array.from(document.querySelectorAll(\'.p-name\')).map(i=>i.value.trim()).filter(Boolean);\n  const start_score = parseInt(document.getElementById(\'start-score\').value, 10);\n  const double_out = document.getElementById(\'double-out\').checked;\n  await postJSON(\'/api/new_game\', {players: names, start_score, double_out});\n  closeNewGame();\n  document.getElementById(\'manual-entry\').style.display = \'block\';\n  refresh();\n}\n\nfunction openCorrect(dartId){\n  correctingDartId = dartId;\n  const img = document.getElementById(\'correct-img\');\n  img.src = `/api/dart_image/${dartId}?t=${Date.now()}`;\n  document.getElementById(\'correct-modal\').classList.add(\'open\');\n}\nfunction closeCorrect(){\n  document.getElementById(\'correct-modal\').classList.remove(\'open\');\n  correctingDartId = null;\n}\ndocument.getElementById(\'correct-img\').addEventListener(\'click\', async (e)=>{\n  if(correctingDartId === null) return;\n  const rect = e.target.getBoundingClientRect();\n  const scaleX = e.target.naturalWidth / rect.width;\n  const scaleY = e.target.naturalHeight / rect.height;\n  const px = (e.clientX - rect.left) * scaleX;\n  const py = (e.clientY - rect.top) * scaleY;\n  await postJSON(\'/api/correct_dart\', {id: correctingDartId, px, py});\n  closeCorrect();\n  refresh();\n});\n\nfunction fmtLabel(d){\n  return d.manual ? `${d.label} *` : d.label;\n}\n\nasync function refresh(){\n  const res = await fetch(\'/api/state\');\n  const data = await res.json();\n\n  document.getElementById(\'cam-dot\').classList.toggle(\'off\', !!data.camera_error);\n  const errEl = document.getElementById(\'cam-error\');\n  if(data.camera_error){ errEl.style.display=\'block\'; errEl.textContent = \'Camera: \' + data.camera_error; }\n  else{ errEl.style.display=\'none\'; }\n\n  const g = data.game;\n  const playersEl = document.getElementById(\'players\');\n  const turnEl = document.getElementById(\'turn-strip\');\n  const logEl = document.getElementById(\'log\');\n\n  if(!g){\n    playersEl.innerHTML = \'<div style="padding:16px; color:var(--muted);">No game yet &mdash; start a new game to begin.</div>\';\n    turnEl.innerHTML=\'\'; logEl.innerHTML=\'\';\n    document.getElementById(\'manual-entry\').style.display = \'none\';\n    return;\n  }\n\n  document.getElementById(\'manual-entry\').style.display = g.finished ? \'none\' : \'block\';\n\n  playersEl.innerHTML = g.players.map((p,i)=>{\n    const active = (i===g.current_player_idx && !g.finished);\n    const won = g.finished && g.winner===p.name;\n    return `<div class="player-card ${active?\'active\':\'\'} ${won?\'winner\':\'\'}">\n      <div>\n        <div class="player-name">${p.name}${won?\' &nbsp;&#127942;\':\'\'}</div>\n        <div class="player-sub">${p.history.length} turn${p.history.length===1?\'\':\'s\'} played</div>\n      </div>\n      <div class="player-score">${p.display_score}</div>\n    </div>`;\n  }).join(\'\');\n\n  if(g.finished){\n    turnEl.innerHTML = `<div class="winner-banner" style="flex:1;">${g.winner} wins!</div>`;\n  } else {\n    const slots = [0,1,2].map(i=>{\n      const d = g.current_turn_darts[i];\n      if(!d) return `<div class="dart-slot">&mdash;</div>`;\n      const canCorrect = d.id != null;\n      return `<div class="dart-slot filled" ${canCorrect?`onclick="openCorrect(${d.id})"`:\'\'}>\n        <div class="label">${fmtLabel(d)}</div>\n        <div class="pts">${d.points} pts</div>\n        ${d.manual ? \'<div class="manual-tag">manual</div>\' : (canCorrect ? \'<div class="manual-tag">tap to correct</div>\' : \'\')}\n      </div>`;\n    }).join(\'\');\n    turnEl.innerHTML = slots;\n  }\n\n  logEl.innerHTML = g.log.slice().reverse().map(l=>`<div>${l}</div>`).join(\'\');\n}\n\nrefresh();\nsetInterval(refresh, 1200);\n</script>\n\n</body>\n</html>\n'


CALIBRATE_HTML = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>Calibrate &middot; Oche</title>\n<link rel="preconnect" href="https://fonts.googleapis.com">\n<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">\n<style>\n  :root{\n    --bg:#12140f; --panel:#1b1f16; --panel-2:#20251a; --brass:#c9a24b;\n    --chalk:#f2ede1; --muted:#9a9d8a; --red:#b3392b; --green:#3c7a4f;\n  }\n  *{box-sizing:border-box;}\n  body{ margin:0; background:var(--bg); color:var(--chalk); font-family:\'Inter\',sans-serif; }\n  header{ display:flex; align-items:center; justify-content:space-between; padding:18px 22px; border-bottom:1px solid #2a2f21;}\n  h1{ font-family:\'Oswald\',sans-serif; text-transform:uppercase; letter-spacing:0.06em; font-size:1.2rem; margin:0;}\n  a.back{ color:var(--muted); text-decoration:none; font-size:0.85rem; }\n  main{ max-width:760px; margin:0 auto; padding:20px 22px 60px; }\n  .steps{ display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap; }\n  .step{ padding:8px 12px; border-radius:20px; border:1px solid #33392a; font-size:0.82rem; color:var(--muted); }\n  .step.active{ border-color:var(--brass); color:var(--chalk); background:rgba(201,162,75,0.1); }\n  .step.done{ border-color:var(--green); color:var(--chalk); }\n  .frame-wrap{ position:relative; border-radius:10px; overflow:hidden; border:1px solid #262c1c; }\n  .frame-wrap img{ width:100%; display:block; cursor:crosshair; }\n  .marker{ position:absolute; width:16px; height:16px; margin:-8px 0 0 -8px; border-radius:50%; border:2px solid var(--brass); background:rgba(201,162,75,0.35); pointer-events:none;}\n  .marker span{ position:absolute; top:-22px; left:50%; transform:translateX(-50%); font-size:0.7rem; color:var(--brass); white-space:nowrap;}\n  p.help{ color:var(--muted); font-size:0.9rem; line-height:1.5; }\n  .actions{ display:flex; gap:10px; margin-top:16px; }\n  button{ font-family:\'Inter\',sans-serif; font-weight:600; font-size:0.9rem; background:var(--panel-2);\n          color:var(--chalk); border:1px solid #33392a; padding:10px 16px; border-radius:8px; cursor:pointer; }\n  button.primary{ background:#1f3d29; border-color:var(--green); }\n  button:disabled{ opacity:0.4; cursor:not-allowed; }\n  #status{ margin-top:14px; font-size:0.88rem; }\n  #status.ok{ color:#8fd19e; }\n  #status.err{ color:#e8a49c; }\n</style>\n</head>\n<body>\n<header>\n  <h1>Calibrate board</h1>\n  <a class="back" href="/">&larr; Back to scoreboard</a>\n</header>\n<main>\n  <p class="help">\n    Click <strong>Refresh snapshot</strong> to grab a fresh frame, then click the outer\n    edge of the double wire at four points, in this exact order:\n    the board\'s own <strong>top</strong>, <strong>right</strong>, <strong>bottom</strong> and\n    <strong>left</strong> (relative to how the board is mounted &mdash; top is directly above\n    the bullseye, not necessarily where "20" is written).\n  </p>\n\n  <div class="steps">\n    <div class="step" id="step-0">1. Top</div>\n    <div class="step" id="step-1">2. Right</div>\n    <div class="step" id="step-2">3. Bottom</div>\n    <div class="step" id="step-3">4. Left</div>\n  </div>\n\n  <div class="frame-wrap" id="frame-wrap">\n    <img id="frame" src="/snapshot.jpg" alt="Board snapshot">\n  </div>\n\n  <div class="actions">\n    <button onclick="refreshSnapshot()">Refresh snapshot</button>\n    <button onclick="resetPoints()">Start over</button>\n    <button class="primary" id="save-btn" onclick="save()" disabled>Save calibration</button>\n  </div>\n\n  <div id="status"></div>\n</main>\n\n<script>\nlet points = [];\n\nfunction refreshSnapshot(){\n  document.getElementById(\'frame\').src = \'/snapshot.jpg?t=\' + Date.now();\n  resetPoints();\n}\n\nfunction resetPoints(){\n  points = [];\n  document.querySelectorAll(\'.marker\').forEach(m => m.remove());\n  [0,1,2,3].forEach(i => document.getElementById(\'step-\'+i).className = \'step\');\n  document.getElementById(\'step-0\').className = \'step active\';\n  document.getElementById(\'save-btn\').disabled = true;\n  document.getElementById(\'status\').textContent = \'\';\n}\n\ndocument.getElementById(\'frame\').addEventListener(\'click\', (e)=>{\n  if(points.length >= 4) return;\n  const img = e.target;\n  const rect = img.getBoundingClientRect();\n  const scaleX = img.naturalWidth / rect.width;\n  const scaleY = img.naturalHeight / rect.height;\n  const x = (e.clientX - rect.left) * scaleX;\n  const y = (e.clientY - rect.top) * scaleY;\n  const labels = [\'Top\',\'Right\',\'Bottom\',\'Left\'];\n\n  const marker = document.createElement(\'div\');\n  marker.className = \'marker\';\n  marker.style.left = (e.clientX - rect.left) + \'px\';\n  marker.style.top = (e.clientY - rect.top) + \'px\';\n  marker.innerHTML = `<span>${labels[points.length]}</span>`;\n  document.getElementById(\'frame-wrap\').appendChild(marker);\n\n  points.push({x, y});\n  document.getElementById(\'step-\'+(points.length-1)).className = \'step done\';\n  if(points.length < 4){\n    document.getElementById(\'step-\'+points.length).className = \'step active\';\n  } else {\n    document.getElementById(\'save-btn\').disabled = false;\n  }\n});\n\nasync function save(){\n  const statusEl = document.getElementById(\'status\');\n  statusEl.className=\'\'; statusEl.textContent = \'Saving...\';\n  const res = await fetch(\'/api/calibrate\', {\n    method:\'POST\', headers:{\'Content-Type\':\'application/json\'},\n    body: JSON.stringify({points})\n  });\n  const data = await res.json();\n  if(data.ok){\n    statusEl.className=\'ok\'; statusEl.textContent = \'Calibration saved. You can go back to the scoreboard now.\';\n  } else {\n    statusEl.className=\'err\'; statusEl.textContent = data.error || \'Something went wrong.\';\n  }\n}\n</script>\n</body>\n</html>\n'


# ----------------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------------




app = Flask(__name__)

board = DartBoard()
detector = DartDetector(camera_index=board.camera_index,
                         width=board.frame_width, height=board.frame_height)

state_lock = threading.Lock()
game = None                # type: Game
dart_events = {}           # dart_id -> dict(px, py, points, label, jpeg, player)
next_dart_id = 1
camera_error = None


def _on_dart_detected(px, py, frame):
    global next_dart_id
    with state_lock:
        if game is None or game.finished or not board.is_calibrated:
            return
        if game.turn_complete():
            # Extra motion after 3 darts are already thrown - most likely
            # someone reaching in. Ignore; use "Reset board" after pulling.
            return

        points, label, _number = board.score_at_pixel(px, py)
        dart_id = next_dart_id
        next_dart_id += 1

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        dart_events[dart_id] = {
            "px": px, "py": py, "points": points, "label": label,
            "jpeg": jpeg.tobytes() if ok else None,
            "player": game.current_player.name,
        }
        game.add_dart(points, label, px, py, dart_id=dart_id)


detector.on_dart_detected = _on_dart_detected


def _try_start_camera():
    global camera_error
    try:
        detector.start()
        time.sleep(0.5)
        detector.resync_baseline()
        camera_error = None
    except RuntimeError as e:
        camera_error = str(e)


_try_start_camera()


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/calibrate")
def calibrate_page():
    return render_template_string(CALIBRATE_HTML)


# ----------------------------------------------------------------------
# Video
# ----------------------------------------------------------------------
def _mjpeg_generator():
    while True:
        frame = detector.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        overlay = frame.copy()

        if board.is_calibrated:
            pts = board.outline_points_px()
            if pts:
                pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(overlay, [pts_np], True, (60, 220, 60), 2)

        if game is not None:
            for d in game.current_turn_darts:
                if d.px is not None:
                    p = (int(d.px), int(d.py))
                    cv2.drawMarker(overlay, p, (40, 40, 235),
                                    markerType=cv2.MARKER_CROSS,
                                    markerSize=22, thickness=2)

        ok, jpeg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            continue
        chunk = (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                  jpeg.tobytes() + b"\r\n")
        yield chunk
        time.sleep(0.04)  # ~25fps cap, gentle on the Pi 3


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_generator(),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot.jpg")
def snapshot():
    frame = detector.get_frame()
    if frame is None:
        return "No camera frame available", 503
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(jpeg.tobytes(), mimetype="image/jpeg")


@app.route("/api/dart_image/<int:dart_id>")
def dart_image(dart_id):
    ev = dart_events.get(dart_id)
    if not ev or not ev.get("jpeg"):
        return "Not found", 404
    return Response(ev["jpeg"], mimetype="image/jpeg")


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------
@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    data = request.get_json(force=True)
    points = data.get("points")
    if not points or len(points) != 4:
        return jsonify({"error": "Provide exactly 4 points: top, right, bottom, left."}), 400
    try:
        board.calibrate([(p["x"], p["y"]) for p in points])
    except CalibrationError as e:
        return jsonify({"error": str(e)}), 400
    detector.resync_baseline()
    return jsonify({"ok": True})


@app.route("/api/camera", methods=["POST"])
def api_camera():
    data = request.get_json(force=True)
    index = data.get("index")
    if index is None:
        return jsonify({"error": "Missing 'index'."}), 400
    detector.stop()
    board.camera_index = int(index)
    board.save()
    detector.camera_index = int(index)
    _try_start_camera()
    return jsonify({"ok": camera_error is None, "error": camera_error})


# ----------------------------------------------------------------------
# Game / scoring
# ----------------------------------------------------------------------
@app.route("/api/state")
def api_state():
    with state_lock:
        g = game.state() if game else None
        # attach dart ids for the current in-progress turn so the client
        # can offer a "correct" action with the matching snapshot image
        if game is not None and g is not None:
            for i, d in enumerate(game.current_turn_darts):
                g["current_turn_darts"][i]["id"] = d.dart_id
    return jsonify({
        "game": g,
        "calibrated": board.is_calibrated,
        "camera_error": camera_error,
    })


@app.route("/api/new_game", methods=["POST"])
def api_new_game():
    global game, dart_events, next_dart_id
    data = request.get_json(force=True)
    players = data.get("players") or ["Player 1"]
    start_score = int(data.get("start_score", 501))
    double_out = bool(data.get("double_out", False))
    with state_lock:
        game = Game(players, start_score=start_score, double_out=double_out)
        dart_events = {}
        next_dart_id = 1
    detector.resync_baseline()
    return jsonify({"ok": True})


@app.route("/api/manual_dart", methods=["POST"])
def api_manual_dart():
    data = request.get_json(force=True)
    points = data.get("points")
    label = data.get("label")
    if points is None or label is None:
        return jsonify({"error": "Provide 'points' and 'label'."}), 400
    with state_lock:
        if game is None:
            return jsonify({"error": "No game in progress."}), 400
        game.add_dart(int(points), str(label), manual=True)
    return jsonify({"ok": True})


@app.route("/api/correct_dart", methods=["POST"])
def api_correct_dart():
    data = request.get_json(force=True)
    dart_id = data.get("id")
    if dart_id is None:
        return jsonify({"error": "Missing 'id'."}), 400
    dart_id = int(dart_id)

    if "px" in data and "py" in data:
        if not board.is_calibrated:
            return jsonify({"error": "Board isn't calibrated."}), 400
        points, label, _ = board.score_at_pixel(data["px"], data["py"])
    elif "points" in data and "label" in data:
        points, label = int(data["points"]), str(data["label"])
    else:
        return jsonify({"error": "Provide either px/py or points/label."}), 400

    with state_lock:
        if game is None:
            return jsonify({"error": "No game in progress."}), 400
        ok = game.correct_dart(dart_id, points, label)
        if not ok:
            return jsonify({"error": "That dart can no longer be corrected "
                                      "(its turn has already ended)."}), 400
        if dart_id in dart_events:
            dart_events[dart_id]["points"] = points
            dart_events[dart_id]["label"] = label
    return jsonify({"ok": True})


@app.route("/api/undo", methods=["POST"])
def api_undo():
    with state_lock:
        if game is None:
            return jsonify({"error": "No game in progress."}), 400
        ok = game.undo_last_dart()
    return jsonify({"ok": ok})


@app.route("/api/end_turn", methods=["POST"])
def api_end_turn():
    with state_lock:
        if game is None:
            return jsonify({"error": "No game in progress."}), 400
        game.force_end_turn()
    return jsonify({"ok": True})


@app.route("/api/reset_board", methods=["POST"])
def api_reset_board():
    detector.resync_baseline()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
