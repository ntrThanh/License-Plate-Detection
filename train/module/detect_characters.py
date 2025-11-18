from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('checkpoint/yolo11n.pt')
    results = model.train(
        data='./train/yml/data-character-digit.yml',
        epochs=20,
        imgsz=448,
        batch=32,
        name='yolo detects characters in license plate'
    )
