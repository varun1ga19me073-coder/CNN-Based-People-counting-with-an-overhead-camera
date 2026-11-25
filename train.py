import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="overhead.yaml")
    ap.add_argument("--model", default="yolov8s.pt", help="pretrained start weights")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="overhead_people")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=20,         
        
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,   
        degrees=10.0,                         
        translate=0.1, scale=0.3,
        fliplr=0.5, flipud=0.5,                
        perspective=0.0,                       
        mosaic=1.0,
    )

    metrics = model.val()
    print(f"precision={metrics.box.mp:.3f}  recall={metrics.box.mr:.3f}  mAP50={metrics.box.map50:.3f}")


if __name__ == "__main__":
    main()
