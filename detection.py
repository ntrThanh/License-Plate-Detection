import argparse
import os

import cv2
import torch
import torchvision
from ultralytics import YOLO

import model.model_cnn
import model_function
from model_function import categories

softmax = torch.nn.Softmax(dim=1)

def get_args():
    parser = argparse.ArgumentParser(description='Program detects license plates')
    parser.add_argument('--image', '-i', help='Use image', action='store_true')
    parser.add_argument('--path-image', '-p', help='Path to image', default='None', type=str)
    parser.add_argument('--camera', '-c', help='Use Camera', action='store_true')
    args = parser.parse_args()
    return args


def print_decor(character):
    print()
    for i in range(10):
        print(f'{character}', end='')

    print(' ', end='')
    print('Result', end=' ')

    for i in range(10):
        print(f'{character}', end='')

    print('\n')

def get_model():
    model_CNN = model.model_cnn.ClassifierNumber(num_classes=len(os.listdir('./dataset/CNN letter Dataset')))
    model_CNN.load_state_dict(torch.load('./checkpoint/best_model.pth', map_location=torch.device('cpu'))['state_dict'])
    return (YOLO('./runs/detect/yolo detects license plate/weights/best.pt'),
            YOLO('./runs/detect/yolo detects characters in license plate7/weights/best.pt'),
            model_CNN)

def resize_keep_ratio(image, target_height=64):
    h, w = image.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    resized = cv2.resize(image, (new_w, target_height))
    return resized

def detect_use_image(image_path):
    p1, p2, p3, p4 = None, None, None, None
    list_license = []
    model1, model2, model3 = get_model()

    image = cv2.imread(image_path)
    image_detected_license_plate = model1(image)

    for box in image_detected_license_plate[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        p1, p2, p3, p4 = x1, y1, x2, y2

        image_crop = image[y1:y2, x1:x2]
        image_crop = resize_keep_ratio(image_crop, 100)
        image_detect = model2(image_crop)

        labels_characters = []
        for x in image_detect[0].boxes:
            a1, b1, a2, b2 = map(int, x.xyxy[0])

            crop = image_crop[b1:b2, a1:a2]
            crop_resized = cv2.resize(crop, (75, 100))
            cv2.imshow('crop', crop_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            tensor_crop = torchvision.transforms.ToTensor()(crop_resized).unsqueeze(0)

            with torch.no_grad():
                result = model3(tensor_crop)

                result_softmax = softmax(result)
                prediction = torch.argmax(result_softmax).item()
                character = categories[prediction]
                labels_characters.append(character)
        output = model_function.license_plate_to_text(image_detect[0].boxes.xyxy, labels_characters)
        list_license.append(output)

    return list_license


# def detect_use_camera():
#     model1, model2, model3 = get_model()

if __name__ == '__main__':
    arguments = get_args()

    if arguments.image and arguments.path_image:
        list_license = detect_use_image(arguments.path_image)
        for x in list_license:
            print(x)
