import argparse

import cv2
from ultralytics import YOLO
# from model_function import license_plate_to_text
import torch
import numpy as np
from detection import DetectionLicensePlate


def get_args():
    parser = argparse.ArgumentParser(description='Program detects license plates')
    parser.add_argument('--image', '-i', help='Use image', action='store_true')
    parser.add_argument('--path-image', '-p', help='Path to image', default='None', type=str)
    parser.add_argument('--camera', '-c', help='Use Camera', action='store_true')
    args = parser.parse_args()
    return args
#
#
# def print_decor(character):
#     print()
#     for i in range(10):
#         print(f'{character}', end='')
#
#     print(' ', end='')
#     print('Result', end=' ')
#
#     for i in range(10):
#         print(f'{character}', end='')
#
#     print('\n')
#
# def get_model():
#     return (YOLO('runs/detect/yolo detects license plate/weights/best.pt'),
#             YOLO('checkpoint/best_yolo_classified/best.pt'))
#
# def up_constraint_threshold(image):
#     gray_image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_TRUNC)
#
#     return image
#
#
#
# def get_class_character(cls_id):
#     dic = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: 'A', 10: 'B', 11: 'C', 12: 'D', 13: 'E',
#      14: 'F', 15: 'G', 16: 'H', 17: 'K', 18: 'L', 19: 'M', 20: 'N', 21: 'P', 22: 'S', 23: 'T', 24: 'U', 25: 'V',
#      26: 'X', 27: 'Y', 28: 'Z', 29: '0', 30: 'J', 31: 'Q', 32: 'R', 33: 'W', 34: 'I'}
#     return dic[cls_id]
#
#
# def detect_use_image(image_path):
#     list_license = []
#     model1, model2 = get_model()
#
#     image = cv2.imread(image_path)
#     image_detected_license_plate = model1(image)
#
#     for box in image_detected_license_plate[0].boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#
#         cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         image_crop = image[y1:y2, x1:x2]
#         # image_crop = resize_keep_ratio(image_crop, 100)
#
#         image_crop = image[y1:y2, x1:x2]
#
#         if image_crop.size == 0:
#             continue  # bỏ qua nếu crop lỗi
#         image_crop = cv2.resize(image_crop, (64, 64))  # hoặc resize_keep_ratio(image_crop, 100)
#
#         image_detect = model2(image_crop)
#
#         boxes_characters = []
#         labels_characters = []
#
#         for char_box in image_detect[0].boxes:
#             x1_c, y1_c, x2_c, y2_c = map(float, char_box.xyxy[0])
#             cls_id = int(char_box.cls[0])
#             label = get_class_character(cls_id)
#
#             boxes_characters.append([x1_c, y1_c, x2_c, y2_c])
#             labels_characters.append(label)
#
#         boxes_tensor = torch.tensor(np.array(boxes_characters), dtype=torch.float32)
#
#         license_text = license_plate_to_text(boxes_tensor, labels_characters)
#         string_text = ''
#         for i in range(len(license_text)):
#             string_text += license_text[i]
#
#         cv2.putText(image, string_text, (x1 - 15, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
#         list_license.append(string_text)
#
#     cv2.imshow('image', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     return list_license


if __name__ == '__main__':
    # arguments = get_args()

    path_image = '/home/trong-thanh/Downloads/greenpack_0016.png'

    image = cv2.imread(path_image)
    model = DetectionLicensePlate(
        checkpoint_detect='runs/detect/yolo detects license plate/weights/best.pt',
        checkpoint_classify='checkpoint/best_yolo_classified/best.pt'
    )

    image_out = model.detect(image)[1]

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
