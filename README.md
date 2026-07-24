# Darts-scorer
Run the folder on your raspberry pi. It will tell you what you are missing when you run the file. It will use flask to create a web application and an api where you can connect via an ip address on port 5000. e.g(192.168.1.1:5000). For now it will only use a single camera I might work on adding multi camera support but I can't test it because I only have one camera. For now it isn't brilliant but I am not too sure on how to intergrate machine learning as most of this project was create by claude.


KNOWN LIMITATIONS - please read before relying on this:
  A single ordinary camera cannot see depth, so it cannot always tell
  a dart's tip apart from its shaft/flight, especially when two darts
  overlap from the camera's point of view, or near a ring/segment
  boundary. Lighting changes can also occasionally trigger a false
  detection. This is why every detected dart is shown with its
  snapshot for a quick correction, and why manual entry exists as a
  fallback - treat this as a scoring aid to keep an eye on, not a
  fully hands-off referee.
