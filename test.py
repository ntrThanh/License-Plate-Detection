import os

import cv2
import torch
import torchvision
from torch import nn
from ultralytics import YOLO
from model_function import license_plate_to_text
from model.model_cnn import ClassifierNumber

if __name__ == '__main__':
    path = '/home/trong-thanh/Downloads/OCR/images/train/1xemay915.jpg'
    path_check_point_CNN = './checkpoint/best_model.pth'
    path_check_point_YOLO = './runs/detect/yolo detects characters in license plate7/weights/best.pt'

    # model yolo
    model_yolo = YOLO(path_check_point_YOLO)

    # model CNN
    model = ClassifierNumber(num_classes=len(os.listdir('./dataset/CNN letter Dataset')))
    model.load_state_dict(torch.load(path_check_point_CNN, map_location=torch.device('cpu'))['state_dict'])

    image = cv2.imread(path)
    result = model_yolo(image)

    # list categories
    categories = [x for x in os.listdir('./dataset/CNN letter Dataset')]

    softmax = nn.Softmax(dim=1)

    print(f'BOX: {result[0].boxes}')
    print(f'POSITION: {result[0].boxes.xyxy}')
    print(f'CLASS: {result[0].boxes.cls}')
    print(f'VALUE: {license_plate_to_text(result[0].boxes.xyxy, result[0].boxes.cls)}')

    for box in result[0].boxes:
        # just get coordinate
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        image_after_crop = image[y1:y2, x1:x2]
        image_resized = cv2.resize(image_after_crop, (128, 128))
        image_tensor = torchvision.transforms.ToTensor()(image_resized).unsqueeze(0)

        with torch.no_grad():
            result = model(image_tensor)

            result_softmax = softmax(result)
            prediction_image = torch.argmax(result_softmax).item()
            emotion = categories[prediction_image]

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f'{emotion}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    