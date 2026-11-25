import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder with .mp4/.avi/.mov files")
    ap.add_argument("--out", default="frames", help="output folder for .jpg frames")
    ap.add_argument("--every", type=int, default=10, help="keep every Nth frame")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    saved = 0

    for video in sorted(Path(args.videos).iterdir()):
        if video.suffix.lower() not in exts:
            continue
        cap = cv2.VideoCapture(str(video))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % args.every == 0:
                cv2.imwrite(str(out / f"{video.stem}_{idx:06d}.jpg"), frame)
                saved += 1
            idx += 1
        cap.release()
        print(f"{video.name}: done")

    print(f"Saved {saved} frames to {out}/")


if __name__ == "__main__":
    main()
