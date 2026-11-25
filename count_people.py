import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file, RTSP url or webcam index")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--axis", choices=["x", "y"], default="y")
    ap.add_argument("--line", type=float, default=0.5)
    ap.add_argument("--margin", type=int, default=15,
                    help="dead zone in pixels around the line (prevents double counts)")
    ap.add_argument("--flip", action="store_true", help="swap in / out direction")
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="class ids to keep (use 0 for COCO 'person' with default weights)")
    ap.add_argument("--out", default="counted.mp4", help="annotated output video ('' to disable)")
    ap.add_argument("--csv", default="crossings.csv")
    ap.add_argument("--show", action="store_true", help="show a live window")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.weights)

    # video properties for the writer
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    in_count = out_count = 0
    last_side = {}          
    writer = None
    events = []             

    results = model.track(
        source=source,
        conf=args.conf,
        classes=args.classes,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
    )

    for frame_no, r in enumerate(results):
        frame = r.orig_img.copy()
        h, w = frame.shape[:2]
        line_pos = int((h if args.axis == "y" else w) * args.line)

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            xyxy = r.boxes.xyxy.tolist()
            for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                pos = cy if args.axis == "y" else cx

            
                if pos < line_pos - args.margin:
                    side = -1
                elif pos > line_pos + args.margin:
                    side = 1
                else:
                    side = None

                if side is not None:
                    prev = last_side.get(tid)
                    if prev is not None and prev != side:
                        entering = (prev == -1 and side == 1)
                        if args.flip:
                            entering = not entering
                        if entering:
                            in_count += 1
                            events.append((frame_no, tid, "in"))
                        else:
                            out_count += 1
                            events.append((frame_no, tid, "out"))
                    last_side[tid] = side

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"#{tid}", (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        
        if args.axis == "y":
            cv2.line(frame, (0, line_pos), (w, line_pos), (0, 255, 255), 2)
        else:
            cv2.line(frame, (line_pos, 0), (line_pos, h), (0, 255, 255), 2)
        label = f"IN {in_count}   OUT {out_count}   INSIDE {in_count - out_count}"
        cv2.rectangle(frame, (0, 0), (420, 34), (0, 0, 0), -1)
        cv2.putText(frame, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if args.out:
            if writer is None:
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        if args.show:
            cv2.imshow("people counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer:
        writer.release()
    cv2.destroyAllWindows()

    with open(args.csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "track_id", "direction"])
        wr.writerows(events)

    print(f"IN: {in_count}  OUT: {out_count}  INSIDE: {in_count - out_count}")
    if args.out:
        print(f"Annotated video: {Path(args.out).resolve()}")
    print(f"Events log: {Path(args.csv).resolve()}")


if __name__ == "__main__":
    main()
