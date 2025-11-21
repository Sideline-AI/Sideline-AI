"""
AI Sports Cam - Phase 1 PC Prototype

This script provides a desktop prototype that:
- Supports two input modes: webcam or video file
- Uses YOLOv8n from ultralytics with .track() for consistent ball tracking
- Smooths and predicts the target position with a Kalman filter
- Crops each frame to 1280x720 around the smoothed position with proper edge handling
- Displays only the final, cropped, and smoothed image via cv2.imshow

Usage examples:
  python prototype.py --source 0               # webcam index 0
  python prototype.py --source path/to/video.mp4

Notes:
- The model uses the COCO class "sports ball" for ball detection.
- Install dependencies: pip install ultralytics opencv-python numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - runtime environment specific
    print(
        "Error: Failed to import 'ultralytics'.\n"
        "Install dependencies first, e.g.:\n"
        "  pip install ultralytics opencv-python numpy\n"
        f"Details: {exc}",
        file=sys.stderr,
    )
    # We don't exit immediately because we might be using Roboflow
    pass

try:
    from roboflow import Roboflow
except ImportError:
    pass


WINDOW_TITLE: str = "AI Sports Cam Prototype"
DEFAULT_OUTPUT_SIZE: Tuple[int, int] = (1280, 720)


@dataclass
class Detection:
    track_id: Optional[int]
    confidence: float
    center_x: float
    center_y: float
    class_id: int  # 0 for person, 1 for ball (mapped internally)
    box: Optional[Tuple[int, int, int, int]] = None


def get_class_ids(model: YOLO) -> Tuple[Optional[int], Optional[int]]:
    """Returns (person_id, sports_ball_id)"""
    names = getattr(model, "names", None)
    person_id = None
    ball_id = None
    
    if isinstance(names, dict):
        for class_id, class_name in names.items():
            name = str(class_name).lower().strip()
            if name == "person":
                person_id = int(class_id)
            elif name == "sports ball":
                ball_id = int(class_id)
    return person_id, ball_id


def calculate_hybrid_target(
    ball_detections: List[Detection],
    player_detections: List[Detection],
    last_target: Optional[Tuple[float, float]],
    frame_w: int,
    frame_h: int
) -> Tuple[float, float]:
    """
    Calculates the target camera center based on ball and players.
    Strategy:
    1. If ball is visible: Target = 0.7 * Ball + 0.3 * Player_Centroid
    2. If ball lost: Target = Player_Centroid (or last target if no players)
    """
    
    # Calculate player centroid (average position of all visible players)
    player_centroid_x, player_centroid_y = None, None
    if player_detections:
        px = sum(d.center_x for d in player_detections) / len(player_detections)
        py = sum(d.center_y for d in player_detections) / len(player_detections)
        player_centroid_x, player_centroid_y = px, py

    # Find best ball (highest confidence)
    best_ball = max(ball_detections, key=lambda d: d.confidence) if ball_detections else None
    
    target_x, target_y = None, None

    if best_ball:
        if player_centroid_x is not None:
            # HYBRID MODE: Ball is main focus, but players pull the camera slightly
            # This helps stabilize when ball moves erratically but play is in one area
            # And helps frame the context
            w_ball = 0.8
            w_players = 0.2
            target_x = best_ball.center_x * w_ball + player_centroid_x * w_players
            target_y = best_ball.center_y * w_ball + player_centroid_y * w_players
        else:
            # Only ball visible
            target_x, target_y = best_ball.center_x, best_ball.center_y
    elif player_centroid_x is not None:
        # No ball, follow the crowd
        target_x, target_y = player_centroid_x, player_centroid_y
    
    # Fallback to last known target or center screen
    if target_x is None:
        if last_target:
            target_x, target_y = last_target
        else:
            target_x, target_y = frame_w / 2, frame_h / 2
            
    return target_x, target_y


def extract_detections_from_results(
    results, 
    person_id: Optional[int], 
    ball_id: Optional[int]
) -> Tuple[List[Detection], List[Detection]]:
    """Returns (ball_detections, player_detections)"""
    balls: List[Detection] = []
    players: List[Detection] = []
    
    if not results:
        return balls, players

    res = results[0] if isinstance(results, list) else results
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return balls, players

    cls = getattr(boxes, "cls", None)
    conf = getattr(boxes, "conf", None)
    ids = getattr(boxes, "id", None)
    xyxy = getattr(boxes, "xyxy", None)
    
    if cls is None or conf is None or xyxy is None:
        return balls, players

    cls_np = cls.detach().cpu().numpy() if hasattr(cls, "detach") else np.array(cls)
    conf_np = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.array(conf)
    xyxy_np = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.array(xyxy)
    ids_np = None
    if ids is not None:
        ids_np = ids.detach().cpu().numpy() if hasattr(ids, "detach") else np.array(ids)

    for i in range(xyxy_np.shape[0]):
        c_id = int(cls_np[i])
        
        # Map to internal class ID: 0=person, 1=ball
        internal_class = -1
        if person_id is not None and c_id == person_id:
            internal_class = 0
        elif ball_id is not None and c_id == ball_id:
            internal_class = 1
        else:
            continue

        x1, y1, x2, y2 = xyxy_np[i]
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)
        confidence = float(conf_np[i])
        track_id = int(ids_np[i]) if ids_np is not None and len(ids_np) > i else None
        
        det = Detection(
            track_id=track_id,
            confidence=confidence,
            center_x=cx,
            center_y=cy,
            class_id=internal_class,
            box=(int(x1), int(y1), int(x2), int(y2))
        )
        
        if internal_class == 1:
            balls.append(det)
        else:
            players.append(det)

    return balls, players


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Sports Cam - PC Prototype")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (e.g. '0') or path to a video file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device for the model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model weights path or name (e.g., yolov8n.pt)",
    )
    parser.add_argument(
        "--use_roboflow",
        action="store_true",
        help="Use Roboflow API for inference instead of local YOLO",
    )
    parser.add_argument(
        "--roboflow_key",
        type=str,
        default="DAWQI4w1KCHH1MlWH7t4",
        help="Roboflow API Key",
    )
    parser.add_argument(
        "--roboflow_project",
        type=str,
        default="soccer-ball-kjoyy",
        help="Roboflow Project ID",
    )
    parser.add_argument(
        "--roboflow_version",
        type=int,
        default=1,
        help="Roboflow Model Version",
    )
    parser.add_argument(
        "--skip_frames",
        type=int,
        default=2,
        help="Skip N frames between inferences (for Roboflow API performance). 0=process every frame, 2=every 3rd frame, 4=every 5th frame",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (short side), e.g. 640, 960",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for NMS",
    )
    parser.add_argument(
        "--output_width",
        type=int,
        default=DEFAULT_OUTPUT_SIZE[0],
        help="Output crop width (default 1280)",
    )
    parser.add_argument(
        "--output_height",
        type=int,
        default=DEFAULT_OUTPUT_SIZE[1],
        help="Output crop height (default 720)",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Digital zoom factor (e.g. 1.5 for 150% zoom)",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker configuration used by ultralytics .track()",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug overlays (FPS, center marker, status text)",
    )
    parser.add_argument(
        "--debug_dets",
        action="store_true",
        help="Draw raw detection boxes (diagnostics)",
    )
    parser.add_argument(
        "--no_ball_filter",
        action="store_true",
        help="Do not filter to sports ball class (diagnostics)",
    )
    return parser.parse_args()


def is_webcam_source(source_arg: str) -> bool:
    if source_arg.lower() == "webcam":
        return True
    # Treat as webcam if it's a pure integer string
    try:
        int(source_arg)
        return True
    except ValueError:
        return False


def open_video_capture(source_arg: str) -> cv2.VideoCapture:
    if is_webcam_source(source_arg):
        cam_index: int = int(source_arg) if source_arg.lower() != "webcam" else 0
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        # Try to request a higher resolution for smoother cropping when available
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(source_arg)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source_arg}")
    return cap


def create_kalman_filter(initial_x: float, initial_y: float) -> cv2.KalmanFilter:
    # State vector: [x, y, vx, vy]
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    # TUNE: Lower process noise = more inertia (smoother, harder to change direction)
    # Was: 1e-2. Now: 1e-4 for position, 1e-3 for velocity
    kf.processNoiseCov = np.array(
        [
            [1e-4, 0, 0, 0],
            [0, 1e-4, 0, 0],
            [0, 0, 1e-3, 0],
            [0, 0, 0, 1e-3],
        ],
        dtype=np.float32,
    )
    # TUNE: Higher measurement noise = trust detection less (smoother, less jitter)
    # Was: 1e-1. Now: 1e-0 (1.0)
    kf.measurementNoiseCov = np.array([[1.0, 0], [0, 1.0]], dtype=np.float32)
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[initial_x], [initial_y], [0.0], [0.0]], dtype=np.float32)
    return kf


def calculate_hybrid_target(
    ball_detections: List[Detection],
    player_detections: List[Detection],
    last_target: Optional[Tuple[float, float]],
    frame_w: int,
    frame_h: int
) -> Tuple[float, float]:
    """
    Calculates the target camera center based on ball and players.
    Includes 'Virtual Rail' logic to dampen Y-axis movement.
    """
    
    # Calculate player centroid
    player_centroid_x, player_centroid_y = None, None
    if player_detections:
        px = sum(d.center_x for d in player_detections) / len(player_detections)
        py = sum(d.center_y for d in player_detections) / len(player_detections)
        player_centroid_x, player_centroid_y = px, py

    # Find best ball
    best_ball = max(ball_detections, key=lambda d: d.confidence) if ball_detections else None
    
    target_x, target_y = None, None

    if best_ball:
        if player_centroid_x is not None:
            # HYBRID MODE
            w_ball = 0.8
            w_players = 0.2
            target_x = best_ball.center_x * w_ball + player_centroid_x * w_players
            target_y = best_ball.center_y * w_ball + player_centroid_y * w_players
        else:
            target_x, target_y = best_ball.center_x, best_ball.center_y
    elif player_centroid_x is not None:
        target_x, target_y = player_centroid_x, player_centroid_y
    
    # Fallback
    if target_x is None:
        if last_target:
            target_x, target_y = last_target
        else:
            target_x, target_y = frame_w / 2, frame_h / 2
            
    # === VIRTUAL RAIL / STABILIZATION ===
    # If we have a previous target, apply damping to the Y-axis specifically
    # Sports action is mostly horizontal. We don't want the camera jumping up/down for every lob.
    if last_target:
        last_x, last_y = last_target
        
        # Deadzone: If change is small, don't move at all
        dx = target_x - last_x
        dy = target_y - last_y
        
        if abs(dx) < frame_w * 0.02: # 2% deadzone X
            target_x = last_x
        if abs(dy) < frame_h * 0.05: # 5% deadzone Y (larger for Y)
            target_y = last_y
            
        # Virtual Rail: Pull Y towards the center of the screen (or horizon)
        # This keeps the camera leveled.
        ideal_y = frame_h * 0.5
        target_y = target_y * 0.6 + ideal_y * 0.4

    return target_x, target_y


def set_kalman_dt(kf: cv2.KalmanFilter, dt: float) -> None:
    # Update transition matrix with the current dt
    F = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    kf.transitionMatrix[:] = F


def compute_crop_coordinates(
    frame_w: int,
    frame_h: int,
    center_x: float,
    center_y: float,
    crop_w: int,
    crop_h: int,
) -> Tuple[int, int, int, int]:
    # Initial desired top-left based on the center position
    x1 = int(round(center_x - crop_w / 2.0))
    y1 = int(round(center_y - crop_h / 2.0))

    # Clamp to ensure the crop rectangle is fully inside the frame
    x1 = max(0, min(x1, frame_w - crop_w))
    y1 = max(0, min(y1, frame_h - crop_h))

    x2 = x1 + crop_w
    y2 = y1 + crop_h
    return x1, y1, x2, y2


def crop_frame(
    frame: np.ndarray, center_x: float, center_y: float, out_w: int, out_h: int
) -> np.ndarray:
    h, w = frame.shape[:2]

    # If frame is smaller than target, resize the entire frame to the output size
    if w < out_w or h < out_h:
        return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    x1, y1, x2, y2 = compute_crop_coordinates(w, h, center_x, center_y, out_w, out_h)
    cropped = frame[y1:y2, x1:x2]
    if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
        # Ensure exact output shape in case of rounding issues
        cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return cropped


def main() -> None:
    args = parse_args()
    
    # Initialize Backend
    yolo_model = None
    roboflow_backend = None
    
    # Class IDs for YOLO
    person_id = None
    ball_id = None

    if args.use_roboflow:
        print(f"Initializing Roboflow model: {args.roboflow_project}/{args.roboflow_version}")
        roboflow_backend = RoboflowBackend(
            api_key=args.roboflow_key,
            project_id=args.roboflow_project,
            version=args.roboflow_version
        )
        # Roboflow model likely only has one class (ball). 
        # We assume class 0 from Roboflow is ball.
        # Hybrid tracking with Roboflow ONLY works if we also run a local detector for people,
        # OR if the Roboflow model detects people too.
        # For this prototype, if using Roboflow, we might lose player tracking unless we add a secondary local model.
        # To keep it simple: In Roboflow mode, we just map its output to 'ball'.
    else:
        print(f"Initializing Local YOLO model: {args.model}")
        yolo_model = YOLO(args.model)
        person_id, ball_id = get_class_ids(yolo_model)
        print(f"Classes found - Person: {person_id}, Ball: {ball_id}")

    # Initialize tracking state
    kalman: Optional[cv2.KalmanFilter] = None
    last_target: Optional[Tuple[float, float]] = None
    last_time = time.perf_counter()

    out_w, out_h = int(args.output_width), int(args.output_height)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, out_w, out_h)

    # Debug/diagnostics state
    debug_enabled = bool(args.debug)
    warned_resize_only = False
    warned_equal_size = False
    smoothed_fps = 0.0
    
    # Frame skipping for Roboflow
    frame_counter = 0
    skip_frames = args.skip_frames if args.use_roboflow else 0

    cap = open_video_capture(args.source)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            fh, fw = frame.shape[:2]
            now = time.perf_counter()
            dt = max(1e-3, min(0.25, now - last_time))
            last_time = now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * inst_fps if smoothed_fps > 0 else inst_fps

            # === INFERENCE ===
            ball_detections: List[Detection] = []
            player_detections: List[Detection] = []
            
            should_run_inference = True
            if args.use_roboflow and skip_frames > 0:
                should_run_inference = (frame_counter % (skip_frames + 1) == 0)
            
            frame_counter += 1
            
            if should_run_inference:
                if roboflow_backend:
                    # Roboflow Inference (Assumed to be Ball only for now)
                    # If we wanted people, we'd need a model that detects them or a second pass
                    raw_dets = roboflow_backend.predict(frame, args.conf, args.iou)
                    # Map all Roboflow detections to 'ball' (class_id=1)
                    for d in raw_dets:
                        d.class_id = 1
                        ball_detections.append(d)
                else:
                    # YOLO Inference
                    classes_to_track = []
                    if person_id is not None: classes_to_track.append(person_id)
                    if ball_id is not None: classes_to_track.append(ball_id)
                    
                    results = yolo_model.track(
                        source=frame,
                        conf=args.conf,
                        iou=args.iou,
                        classes=classes_to_track,
                        persist=True,
                        device=args.device,
                        tracker=args.tracker,
                        verbose=False,
                        imgsz=args.imgsz,
                    )
                    ball_detections, player_detections = extract_detections_from_results(results, person_id, ball_id)

            # === HYBRID TARGET CALCULATION ===
            target_x, target_y = calculate_hybrid_target(
                ball_detections, 
                player_detections, 
                last_target, 
                fw, fh
            )
            
            # === SMOOTHING (KALMAN) ===
            # Initialize Kalman if needed
            if kalman is None:
                kalman = create_kalman_filter(target_x, target_y)
                last_target = (target_x, target_y)

            # Predict
            set_kalman_dt(kalman, dt)
            kalman.predict()

            # Update
            # We always have a 'target' from calculate_hybrid_target (it falls back to last known)
            # But we should only correct Kalman if we actually had NEW data (inference ran and found something)
            # Or if we are just interpolating.
            # For simplicity, we treat the hybrid target as a measurement.
            measurement = np.array([[target_x], [target_y]], dtype=np.float32)
            kalman.correct(measurement)
            
            # Get smoothed position
            state = kalman.statePost
            smooth_x = float(state[0, 0])
            smooth_y = float(state[1, 0])
            last_target = (smooth_x, smooth_y)

            # === ZOOM & CROP ===
            zoom_factor = max(1.0, args.zoom)
            view_w = int(fw / zoom_factor)
            view_h = int(fh / zoom_factor)
            view_w = min(fw, view_w)
            view_h = min(fh, view_h)

            cropped_view = crop_frame(frame, smooth_x, smooth_y, view_w, view_h)

            # === DEBUG OVERLAYS ===
            if debug_enabled:
                # Map coordinates to crop
                x1_src, y1_src, x2_src, y2_src = compute_crop_coordinates(
                    fw, fh, smooth_x, smooth_y, view_w, view_h
                )
                
                if args.debug_dets:
                    # Draw balls (Green)
                    for d in ball_detections:
                        if d.box:
                            bx1, by1, bx2, by2 = d.box
                            # Map and clip
                            ix1 = max(bx1, x1_src); iy1 = max(by1, y1_src)
                            ix2 = min(bx2, x2_src); iy2 = min(by2, y2_src)
                            if ix2 > ix1 and iy2 > iy1:
                                cv2.rectangle(cropped_view, (int(ix1-x1_src), int(iy1-y1_src)), (int(ix2-x1_src), int(iy2-y1_src)), (0, 255, 0), 2)
                    
                    # Draw players (Blue)
                    for d in player_detections:
                        if d.box:
                            bx1, by1, bx2, by2 = d.box
                            ix1 = max(bx1, x1_src); iy1 = max(by1, y1_src)
                            ix2 = min(bx2, x2_src); iy2 = min(by2, y2_src)
                            if ix2 > ix1 and iy2 > iy1:
                                cv2.rectangle(cropped_view, (int(ix1-x1_src), int(iy1-y1_src)), (int(ix2-x1_src), int(iy2-y1_src)), (255, 0, 0), 1)

                # Draw Target Point (Red Cross) - The raw hybrid target
                tx = int(max(x1_src, min(target_x, x2_src)) - x1_src)
                ty = int(max(y1_src, min(target_y, y2_src)) - y1_src)
                cv2.drawMarker(cropped_view, (tx, ty), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)

                # Status
                mode_text = "Roboflow" if args.use_roboflow else "YOLO-Hybrid"
                cv2.putText(cropped_view, f"{mode_text} | FPS: {smoothed_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Final Resize
            final_output = cv2.resize(cropped_view, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(WINDOW_TITLE, final_output)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


