"""UDP + JSON link to the other subsystems (no ROS needed).

Out: one measurement per frame, as JSON, to host:port.
In (any subset of keys):
    {"robot_id": 1, "distance_m": 2.1}   floor distance camera -> listener, from overhead CV
    {"camera_pitch_deg": 5.0}             camera tilt up from level, from the servo code
Messages with a different robot_id are ignored.
"""

import json
import socket
import threading


class UdpPublisher:
    def __init__(self, target):
        host, port = target.rsplit(":", 1)
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, msg):
        try:
            self.sock.sendto(json.dumps(msg).encode(), self.addr)
        except OSError:
            pass  # nobody listening yet; UDP just drops it


class UdpListener(threading.Thread):
    def __init__(self, port, tracker):
        super().__init__(daemon=True)
        self.tracker = tracker
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))

    def run(self):
        while True:
            try:
                msg = json.loads(self.sock.recv(4096))
                if int(msg.get("robot_id", self.tracker.robot_id)) != self.tracker.robot_id:
                    continue
                if "distance_m" in msg:
                    self.tracker.set_distance(float(msg["distance_m"]))
                if "camera_pitch_deg" in msg:
                    self.tracker.set_camera_pitch(float(msg["camera_pitch_deg"]))
            except (OSError, ValueError, TypeError, AttributeError):
                continue  # ignore malformed messages
