# Darts-scorer
Run the folder on your raspberry pi (or select debian distro). It will tell you what you are missing when you run the file. It will use flask to create a web application and an api where you can connect via an ip address on port 5000. e.g(192.168.1.1:5000). For now it will only use a single camera I might work on adding multi camera support but I can't test it because I only have one camera. For now it isn't brilliant but I am not too sure on how to intergrate machine learning as most of this project was create by claude.


KNOWN LIMITATIONS - please read before relying on this:
  A single ordinary camera cannot see depth, so it cannot always tell
  a dart's tip apart from its shaft/flight, especially when two darts
  overlap from the camera's point of view, or near a ring/segment
  boundary. Lighting changes can also occasionally trigger a false
  detection. This is why every detected dart is shown with its
  snapshot for a quick correction, and why manual entry exists as a
  fallback - treat this as a scoring aid to keep an eye on, not a
  fully hands-off referee.


I would recommend a 64bit raspberry pi 3 and above but anything that runs debian should be fine. Will be working on an arch version.

As of right now, each detected dart is snapshotted for the correction feature. Those snapshots are only kept in memory while that dart can still be corrected — once its turn ends (scored or busted) or the game finishes, they're automatically dropped, so RAM use stays roughly flat over a session instead of growing with every dart ever thrown. Everything is cleared when the program stops regardless. You can run it in a venv but you can just run it in root.

PERFORMANCE ON A PI 3:
  Defaults are kept at 640x480 to stay smooth on a Pi 3. If the video
  feels laggy, lower frame_width/frame_height in config.json (created
  next to this script after your first run), or increase the
  time.sleep() in _mjpeg_generator() in the .py.

This is just a passion project so any and all help will be much appreciated.

If you are experiencing issues while the program is running please use the debug version located here: https://github.com/Lucamasta9000/Dart-scorer-debug
