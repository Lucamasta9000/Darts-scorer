# Quick start

Get scoring in about 10 minutes: install, point the camera at the board, calibrate, play.

## 1. What you need

- A Raspberry Pi (or any Debian-based machine) — **Pi 4 or newer recommended**, 64-bit OS
- A USB webcam (UVC-compatible — most are)
- Your dartboard mounted somewhere the camera can see the whole face of it

## 2. Install

```bash
git clone https://github.com/Lucamasta9000/Darts-scorer.git
cd Darts-scorer
```

Fastest path on Raspberry Pi OS (uses prebuilt system packages instead of compiling OpenCV from source, which can take a very long time on some Pi hardware):

```bash
sudo apt-get update
sudo apt-get install -y python3-opencv python3-flask python3-numpy
```

Prefer pip / a venv instead?

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Run it

```bash
python3 dart_scorer.py
```

It'll tell you in the terminal if anything's missing. Once it's running, find your Pi's IP address:

```bash
hostname -I
```

Then open `http://<that-ip>:5000` from your phone or laptop (same Wi-Fi/network as the Pi). For example:

```
http://192.168.1.42:5000
```

## 4. Mount and aim the camera

- Point it at the whole board with a bit of margin around the edge
- As close to straight-on as you can manage — some angle is fine, steep angles hurt accuracy
- Mount it securely — if it moves, you'll need to recalibrate
- Even, steady lighting; avoid backlighting the board or anything that flickers

## 5. Calibrate (one-time, per camera position)

1. On the dashboard, click **Calibrate board**
2. Click **Refresh snapshot**
3. Click the outer edge of the double wire, in this exact order:
   **top → right → bottom → left**, relative to how the board is physically mounted (top is directly above the bullseye — not necessarily where "20" is painted)
4. Click **Save calibration**

Recalibrate any time the camera or board gets bumped.

## 6. Play

1. Click **New game**, add player names, choose 301/501/701, optionally require a double to finish
2. Before the first throw (and after pulling darts out between turns), click **Darts pulled — reset board**
3. Throw. Detected darts show up in real time with their score
4. Got one wrong? Tap it in the turn strip to correct it (tap the true landing spot on its snapshot) — only works before that turn ends
5. Camera missed a throw entirely? Use **Manual entry** to add it by hand

## Something not detecting properly?

Run `dart_scorer_debug.py` instead of `dart_scorer.py` — same app, but it prints exactly what's happening in the terminal as you throw (motion detected, contour size, accepted/rejected, and why). Point that output at an issue if you open one.

## Warning

A single camera can't see depth, so it can sometimes misjudge which dart is which, especially near ring boundaries or when darts overlap from the camera's angle. That's what the tap-to-correct and manual entry tools are for — treat this as a scoring aid to keep an eye on, not a fully hands-off referee. See the README for the full list of known limitations.
