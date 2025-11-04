import argparse

import cv2
from ultralytics import YOLO
from model_function import license_plate_to_text
import torch
import numpy as np


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
    return (YOLO('runs/detect/yolo detects license plate/weights/best.pt'),
            YOLO('runs/detect/yolo detects characters in license plate7/weights/best_detect_character.pt'))


def resize_keep_ratio(image, target_height=64):
    h, w = image.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    resized = cv2.resize(image, (new_w, target_height))
    return resized


def get_class_character(cls_id):
    if cls_id == 0:
        pass
    dic = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C',
           13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'K', 19: 'L', 20: 'M', 21: 'L', 22: 'P', 23: 'T', 24: 'U',
           25: 'V', 26: 'X', 27: 'R', 28: 'Z', 29: 'N', 30: '0', 31: 'P', 32: 'W', 33: 'Q', 34: 'Y', 35: 'S'}
    return dic[cls_id + 1]


def detect_use_image(image_path):
    list_license = []
    model1, model2 = get_model()

    image = cv2.imread(image_path)
    image_detected_license_plate = model1(image)

    for box in image_detected_license_plate[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        image_crop = image[y1:y2, x1:x2]
        image_crop = resize_keep_ratio(image_crop, 100)
        image_detect = model2(image_crop)

        boxes_characters = []
        labels_characters = []

        for char_box in image_detect[0].boxes:
            x1_c, y1_c, x2_c, y2_c = map(float, char_box.xyxy[0])
            cls_id = int(char_box.cls[0])
            label = get_class_character(cls_id)

            boxes_characters.append([x1_c, y1_c, x2_c, y2_c])
            labels_characters.append(label)

        boxes_tensor = torch.tensor(np.array(boxes_characters), dtype=torch.float32)

        license_text = license_plate_to_text(boxes_tensor, labels_characters)
        list_license.append(license_text)

    return list_license


if __name__ == '__main__':
    arguments = get_args()

    if arguments.image and arguments.path_image:
        list_license = detect_use_image(arguments.path_image)
        for x in list_license:
            print(x)
