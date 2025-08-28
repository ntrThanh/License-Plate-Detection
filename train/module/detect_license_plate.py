from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./yolo11n.pt')
    result = model.train(data='./train/yml/data-license-plate.yml', epochs=10, imgsz=360, batch=32,
                          name='yolo detects license plate')