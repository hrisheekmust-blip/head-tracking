"""Head tracker main loop.

Laptop:  python run_head_tracker.py --source 0 --show --distance 2.0
Robot:   python3 run_head_tracker.py --source picam --robot-id 1 --calib calib/robot1.json \
             --udp 127.0.0.1:5005 --listen 5006 --stats
"""

import argparse
import json
import os
import sys
import time

import cv2

from .camera import open_camera
from .detector import FaceFinder
from .geometry import FOV_PRESETS, Intrinsics
from .net import UdpListener, UdpPublisher
from .tracker import HeadTracker


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Listener head tracker for the speaker robots")
    p.add_argument("--source", default="picam", help="'picam', a webcam index like '0', or a video file")
    p.add_argument("--robot-id", type=int, default=1)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--fps", type=float, default=15.0, help="Max frames per second")
    p.add_argument("--lens-position", type=float, default=0.5,
                   help="Pi camera manual focus in dioptres (0.5 = 2 m). Same as calibration")
    p.add_argument("--calib", help="Calibration JSON from calibrate.py")
    p.add_argument("--fov", default="cam3", help="No --calib: 'cam3', 'cam3_wide' or degrees")
    p.add_argument("--score", type=float, default=0.5, help="Face confidence threshold")
    p.add_argument("--cam-height", type=float, default=0.95, help="Lens height above floor (m)")
    p.add_argument("--cam-pitch", type=float, default=0.0, help="Starting camera tilt (deg)")
    p.add_argument("--distance", type=float, help="Fixed floor distance (m) for testing without overhead CV")
    p.add_argument("--smoothing", type=float, default=0.5, help="0 = none, 0.9 = heavy")
    p.add_argument("--udp", help="Send measurements to host:port")
    p.add_argument("--listen", type=int, help="UDP port for distance_m / camera_pitch_deg")
    p.add_argument("--print", dest="print_json", action="store_true", help="Print JSON per frame")
    p.add_argument("--show", action="store_true", help="Debug window (laptop only)")
    p.add_argument("--save-video", help="Write an annotated .mp4")
    p.add_argument("--stats", action="store_true", help="Print fps and CPU use every 5 s")
    p.add_argument("--no-roi", action="store_true", help="Search the whole frame every time")
    p.add_argument("--threads", type=int, default=1, help="OpenCV threads (1 leaves cores free)")
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args(argv)


def load_intrinsics(args, w, h):
    if args.calib:
        return Intrinsics.load(args.calib).scaled_to(w, h)
    fov = FOV_PRESETS.get(args.fov) or float(args.fov)
    print(f"[head_tracking] no --calib, using {fov:.0f} deg FOV estimate. "
          "Calibrate before measuring height.", file=sys.stderr)
    return Intrinsics.from_fov(w, h, fov)


def draw(frame, m):
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), (90, 90, 90), 1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (90, 90, 90), 1)
    if m.bbox:
        x, y, bw, bh = map(int, m.bbox)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 220, 0) if m.detected else (0, 160, 255), 2)
        cv2.drawMarker(frame, tuple(map(int, m.head_px)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    lines = ["NO FACE"] if m.yaw_error_deg is None else [
        f"yaw {m.yaw_error_deg:+.1f}  pitch {m.pitch_error_deg:+.1f}  elev {m.elevation_deg:+.1f} deg"]
    if m.head_height_m is not None:
        lines.append(f"head height {m.head_height_m:.2f} m at {m.distance_m:.2f} m")
    for i, s in enumerate(lines):
        y = 24 + 22 * i
        cv2.putText(frame, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


class Stats:
    """Every `period` seconds: loop fps, processing time and CPU use."""

    def __init__(self, period=5.0):
        self.period = period
        self._reset()

    def _reset(self):
        self.t0, self.cpu0 = time.monotonic(), time.process_time()
        self.n, self.work, self.modes = 0, 0.0, {"roi": 0, "full": 0, "skip": 0}

    def add(self, work_s, mode):
        self.n += 1
        self.work += work_s
        self.modes[mode] += 1
        wall = time.monotonic() - self.t0
        if wall < self.period:
            return
        cpu = 100 * (time.process_time() - self.cpu0) / wall
        cores = os.cpu_count() or 1
        pct = {k: round(100 * v / self.n) for k, v in self.modes.items()}
        print(f"[stats] {self.n / wall:.1f} fps | {1000 * self.work / self.n:.1f} ms/frame | "
              f"CPU {cpu:.1f}% of one core ({cpu / cores:.1f}% of all {cores}) | "
              f"search: roi {pct['roi']}% full {pct['full']}% skipped {pct['skip']}%",
              file=sys.stderr, flush=True)
        self._reset()


def main(argv=None):
    args = parse_args(argv)
    cv2.setNumThreads(max(args.threads, 1))

    cam = open_camera(args.source, args.width, args.height, args.fps, args.lens_position)
    frame = cam.read()
    if frame is None:
        sys.exit("Camera returned no frames")
    h, w = frame.shape[:2]

    finder = FaceFinder(score_threshold=args.score, roi=not args.no_roi)
    tracker = HeadTracker(load_intrinsics(args, w, h), args.robot_id, args.cam_height,
                          args.cam_pitch, args.smoothing)
    tracker.fixed_distance_m = args.distance
    pub = UdpPublisher(args.udp) if args.udp else None
    if args.listen:
        UdpListener(args.listen, tracker).start()
    stats = Stats() if args.stats else None
    writer = None

    # Video files run as fast as possible but use the video's own clock, so the
    # timing logic behaves as if it were live. Webcams often ignore fps
    # requests, so we sleep to cap the rate. The Pi camera paces itself.
    is_file = not args.source.startswith("picam") and not args.source.isdigit()
    file_fps = (cam.fps or args.fps) if is_file else None
    period = 1.0 / args.fps if args.source.isdigit() and args.fps > 0 else 0.0

    print(f"[head_tracking] OpenCV {cv2.__version__}, {w}x{h}, robot {args.robot_id}", file=sys.stderr)
    n = 0
    try:
        while frame is not None:
            start = time.perf_counter()
            t = n / file_fps if is_file else time.monotonic()
            faces = finder.find(frame, now=t)
            m = tracker.update(faces, frame.shape, t)
            finder.focus(m.bbox)  # keep searching near the last spot until the tracker gives up

            msg = m.to_dict()
            msg["t"] = round(time.time(), 3)
            msg["mode"] = finder.mode
            if pub:
                pub.send(msg)
            if args.print_json:
                print(json.dumps(msg), flush=True)
            if stats:
                stats.add(time.perf_counter() - start, finder.mode)

            if args.show or args.save_video:
                vis = draw(frame.copy(), m)
                if args.save_video:
                    if writer is None:
                        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 file_fps or args.fps, (w, h))
                    writer.write(vis)
                if args.show:
                    cv2.imshow("head tracker", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            n += 1
            if args.max_frames and n >= args.max_frames:
                break
            spare = period - (time.perf_counter() - start)
            if spare > 0:
                time.sleep(spare)  # sleeping (not spinning) is what frees the CPU
            frame = cam.read()
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
