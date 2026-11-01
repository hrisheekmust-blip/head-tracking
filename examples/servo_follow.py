#!/usr/bin/env python3
"""Starting point for the servo side (Nate): make pan/tilt follow the listener.

Reads the tracker's UDP messages and nudges each servo a fraction of the error
every update, until the face is centered (= speaker aimed at the listener).
Sends the current tilt back so the tracker can compute head height.

Assumes a PCA9685 servo driver on I2C with hobby PWM servos:
    pip install adafruit-circuitpython-servokit

    python3 run_head_tracker.py --source picam --udp 127.0.0.1:5005 --listen 5006
    python3 examples/servo_follow.py --pan-ch 0 --tilt-ch 1
Without hardware:  python3 examples/servo_follow.py --dry-run
If a servo turns the wrong way, flip --pan-sign / --tilt-sign.
"""

import argparse
import json
import socket


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def step(current, error_deg, sign, kp, max_step, deadband, lo, hi):
    """One proportional correction toward zero error."""
    if abs(error_deg) <= deadband:
        return current
    return clamp(current + sign * clamp(kp * error_deg, -max_step, max_step), lo, hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5005, help="Port the tracker sends to")
    p.add_argument("--tracker", default="127.0.0.1:5006", help="Tracker --listen address")
    p.add_argument("--pan-ch", type=int, default=0)
    p.add_argument("--tilt-ch", type=int, default=1)
    p.add_argument("--pan-center", type=float, default=90.0, help="Servo angle for straight ahead")
    p.add_argument("--tilt-level", type=float, default=90.0, help="Servo angle for level")
    p.add_argument("--pan-range", type=float, nargs=2, default=[0.0, 180.0])
    p.add_argument("--tilt-range", type=float, nargs=2, default=[60.0, 120.0])
    p.add_argument("--pan-sign", type=float, default=-1.0)
    p.add_argument("--tilt-sign", type=float, default=1.0)
    p.add_argument("--kp", type=float, default=0.35, help="Fraction of the error corrected per update")
    p.add_argument("--max-step", type=float, default=4.0, help="Max degrees per update")
    p.add_argument("--deadband", type=float, default=1.5, help="Ignore errors below this (deg)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    kit = None
    if not a.dry_run:
        from adafruit_servokit import ServoKit
        kit = ServoKit(channels=16)

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", a.port))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    host, port = a.tracker.rsplit(":", 1)
    tracker_addr = (host, int(port))

    pan, tilt = a.pan_center, a.tilt_level

    def apply():
        if kit:
            kit.servo[a.pan_ch].angle = pan
            kit.servo[a.tilt_ch].angle = tilt
        else:
            print(f"pan {pan:6.1f}  tilt {tilt:6.1f}")
        # Hobby servos have no position feedback, so this is the commanded tilt.
        tilt_up = (tilt - a.tilt_level) * a.tilt_sign
        tx.sendto(json.dumps({"camera_pitch_deg": tilt_up}).encode(), tracker_addr)

    apply()
    print(f"Listening for the tracker on UDP {a.port}")
    while True:
        try:
            m = json.loads(rx.recv(4096))
        except ValueError:
            continue
        if not m.get("detected"):
            continue  # hold position while the listener is out of view
        pan = step(pan, m["yaw_error_deg"], a.pan_sign, a.kp, a.max_step, a.deadband, *a.pan_range)
        tilt = step(tilt, m["pitch_error_deg"], a.tilt_sign, a.kp, a.max_step, a.deadband, *a.tilt_range)
        # TODO: report when pan/tilt sits at a range limit, for the audio phase-delay feature.
        apply()


if __name__ == "__main__":
    main()
