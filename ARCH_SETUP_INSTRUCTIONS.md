# Arch Linux setup

The app itself (`dart_scorer.py`) is plain Python/Flask/OpenCV with nothing
Debian-specific in the code — only the *install steps* differ from the
main README, since Arch uses `pacman` instead of `apt`. Everything else
(calibration, playing, the debug build) works exactly the same; see
`QUICKSTART.md` for that part.

> **Note:** this project was built and tested on Raspberry Pi OS
> (Debian-based). Arch runs fine on a Pi, but it's a less common
> combination than Raspberry Pi OS, so treat this as "should work,
> please report issues" rather than as thoroughly tested as the main
> Debian instructions.

## Install

```bash
sudo pacman -Syu
sudo pacman -S python python-opencv python-numpy python-pip
```

Flask isn't in the official Arch repos under a simple package name, so
install it with pip:

```bash
pip install --break-system-packages flask
```

(Or use a venv instead, if you'd rather not touch system Python packages:)

```bash
python -m venv --system-site-packages venv
source venv/bin/activate
pip install flask
```

## Run

```bash
git clone https://github.com/Lucamasta9000/Darts-scorer.git
cd Darts-scorer
python3 dart_scorer.py
```

Find your machine's IP with:

```bash
ip addr show
```

(Arch doesn't ship `hostname -I` by default the way Debian does — `ip addr show` is the more universal equivalent. Look for the `inet` address on your active network interface.)

Then open `http://<that-ip>:5000` from your phone or laptop, same as the main quick start.

## If `python-opencv` isn't available or is out of date

Arch's official `opencv` package tracks fairly recent OpenCV releases and
usually works well, but if you hit issues, `opencv-python-headless` from
`requirements.txt` via pip is a safe fallback:

```bash
pip install --break-system-packages opencv-python-headless
```

From here, calibration and gameplay are identical to the main
[Quick Start guide](QUICKSTART.md).
