from typing import List, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO


# ---- Define class format ---- #
class Format:
    def __init__(self,
                 min_num_chars=8,
                 max_num_chars=9):
        self.min_num_chars = min_num_chars
        self.max_num_chars = max_num_chars

    def license_plate_to_text(self,
                              boxes_np: np.ndarray,
                              characters_labels: list
                              ) -> str:
        return ""


class KMeansFormat(Format):
    def __init__(self,
                 min_num_chars=8,
                 max_num_chars=9
                 ):
        super().__init__(min_num_chars, max_num_chars)

    def license_plate_to_text(self,
                              boxes_np: np.ndarray,
                              characters_labels: list
                              ) -> str:
        result_chars = []
        num_chars = boxes_np.shape[0]

        if not (self.min_num_chars <=
                num_chars <=
                self.max_num_chars):
            return "Unknown"

        x_centers = (boxes_np[:, 0] + boxes_np[:, 2]) / 2
        y_centers = (boxes_np[:, 1] + boxes_np[:, 3]) / 2
        heights = (boxes_np[:, 3] - boxes_np[:, 1])

        if heights.size == 0:
            return "Unknown"
        avg_height = heights.mean()

        kmeans = KMeans(
            n_clusters=2,
            n_init=10,
            random_state=0
        ).fit(y_centers.reshape(-1, 1))
        labels = kmeans.labels_

        y_dist = abs(
            kmeans.cluster_centers_[0][0] -
            kmeans.cluster_centers_[1][0]
        )

        if y_dist < avg_height * 0.5:
            sorted_indices = sorted(
                range(num_chars),
                key=lambda i: x_centers[i]
            )
            for idx in sorted_indices:
                result_chars.append(
                    (str(characters_labels[idx]))
                )
        else:
            cluster_info = [
                (i, kmeans.cluster_centers_[i][0]) for i in range(2)
            ]
            cluster_info.sort(key=lambda x: x[1])
            top_cluster = cluster_info[0][0]
            bottom_cluster = cluster_info[1][0]

            top_indices = [
                i for i in range(num_chars) if labels[i] == top_cluster
            ]
            top_indices.sort(key=lambda i: x_centers[i])
            for idx in top_indices:
                result_chars.append((characters_labels[idx]))

            bottom_indices = [
                i for i in range(num_chars) if labels[i] == bottom_cluster
            ]
            bottom_indices.sort(key=lambda i: x_centers[i])
            for idx in bottom_indices:
                result_chars.append((characters_labels[idx]))

        return "".join(result_chars)


class HeuristicFormat(Format):

    def __init__(self,
                 min_num_chars=8,
                 max_num_chars=9
                 ):
        super().__init__(min_num_chars, max_num_chars)

    def license_plate_to_text(self,
                              boxes_np: np.ndarray,
                              characters_labels: list
                              ) -> str:

        num_chars = boxes_np.shape[0]

        if not (self.min_num_chars
                <= num_chars
                <= self.max_num_chars):
            return "Unknown"

        x_centers = (boxes_np[:, 0] + boxes_np[:, 2]) / 2
        y_centers = (boxes_np[:, 1] + boxes_np[:, 3]) / 2
        heights = (boxes_np[:, 3] - boxes_np[:, 1])

        if heights.size == 0:
            return "Unknown"

        avg_height = heights.mean()

        chars = []

        for i in range(num_chars):
            chars.append({
                'x': x_centers[i],
                'y': y_centers[i],
                'label': str(characters_labels[i])
            })

        chars.sort(key=lambda c: c['y'])

        max_y_gap = 0
        split_index = -1

        for i in range(num_chars - 1):
            gap = chars[i + 1]['y'] - chars[i]['y']
            if gap > max_y_gap:
                max_y_gap = gap
                split_index = i + 1

        result_chars = []
        if max_y_gap < avg_height * 0.5:
            chars.sort(key=lambda c: c['x'])
            result_chars = [c['label'] for c in chars]
        else:
            top_line = chars[:split_index]
            bottom_line = chars[split_index:]

            top_line.sort(key=lambda c: c['x'])
            bottom_line.sort(key=lambda c: c['x'])

            result_chars = (
                    [c['label'] for c in top_line] +
                    [c['label'] for c in bottom_line]
            )

        return "".join(result_chars)


def get_class_character(cls_id):
    dic = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5',
           5: '6', 6: '7', 7: '8', 8: '9', 9: 'A',
           10: 'B', 11: 'C', 12: 'D', 13: 'E', 14: 'F',
           15: 'G', 16: 'H', 17: 'K', 18: 'L', 19: 'M',
           20: 'N', 21: 'P', 22: 'S', 23: 'T', 24: 'U',
           25: 'V', 26: 'X', 27: 'Y', 28: 'Z', 29: '0',
           30: 'J', 31: 'Q', 32: 'R', 33: 'W', 34: 'I'}
    return dic[cls_id]


def resize_with_aspect_ratio(
        image,
        target_size=64,
        pad_color=(0, 0, 0)):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    result = np.full(
        (target_size, target_size, 3),
        pad_color,
        dtype=np.uint8
    )

    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    result[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized
    return result


def fix_uneven_lighting_bgr(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    L2 = clahe.apply(L)

    L_float = L2.astype(np.float32) + 1.0

    L_retinex = (
            np.log(L_float) -
            cv2.GaussianBlur(np.log(L_float),
                             (0, 0),
                             sigmaX=50)
    )

    L_retinex = cv2.normalize(
        L_retinex,
        None,
        0,
        255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    lab2 = cv2.merge((L_retinex, A, B))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    out = cv2.bilateralFilter(
        out,
        d=5,
        sigmaColor=40,
        sigmaSpace=40
    )

    return out


# version YOLO
class DetectionLicensePlate:
    def __init__(
            self,
            format_license_plate: Format = HeuristicFormat(
                min_num_chars=8,
                max_num_chars=9
            ),
            checkpoint_detect: str = "",
            checkpoint_classify: str = ""
    ):
        self.license_plate = format_license_plate
        self.checkpoint_detect = checkpoint_detect
        self.checkpoint_classify = checkpoint_classify

    def detect(self,
               image: np.ndarray
               ) -> Tuple[List[str], np.ndarray]:

        # List result (license plates)
        list_license_plate = []
        model_detect = YOLO(self.checkpoint_detect)
        model_classify = YOLO(self.checkpoint_classify)
        image_detected = model_detect(image)

        for box in image_detected[0].boxes:
            # get coordinate and color rectangle
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # cut image license after detected
            if image is None:
                print('Can\'t read image!!!')
                break
            image_crop = image[y1:y2, x1:x2]
            if image_crop.size == 0:
                continue

            # resize to optimal receptive field
            # image_crop = fix_uneven_lighting_bgr(image_crop)
            image_crop = resize_with_aspect_ratio(
                image_crop,
                target_size=64
            )

            letters_image_detected = model_classify(image_crop)
            boxes_characters = []
            labels_characters = []

            for char_box in letters_image_detected[0].boxes:
                x1_c, y1_c, x2_c, y2_c = map(
                    float,
                    char_box.xyxy[0]
                )
                cls_id = int(char_box.cls[0])
                label = get_class_character(cls_id)
                boxes_characters.append([x1_c, y1_c, x2_c, y2_c])
                labels_characters.append(label)

            license_text = self.license_plate.license_plate_to_text(
                np.array(boxes_characters),
                labels_characters
            )

            list_license_plate.append(license_text)

            cv2.putText(
                image,
                license_text,
                (x1 - 15, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

        return list_license_plate, image

    def get_boxes_and_labels(self,
               image: np.ndarray
               ) -> List[str]:

        # List result (license plates)
        list_license_plate = []
        model_detect = YOLO(self.checkpoint_detect)
        model_classify = YOLO(self.checkpoint_classify)
        image_detected = model_detect(image)

        for box in image_detected[0].boxes:
            # get coordinate and color rectangle
            dic = {}

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dic['coordinates'] = (x1, y1, x2, y2)

            # cut image license after detected
            if image is None:
                print('Can\'t read image!!!')
                break
            image_crop = image[y1:y2, x1:x2]
            if image_crop.size == 0:
                continue

            # resize to optimal receptive field
            # image_crop = fix_uneven_lighting_bgr(image_crop)
            image_crop = resize_with_aspect_ratio(
                image_crop,
                target_size=64
            )

            letters_image_detected = model_classify(image_crop)
            boxes_characters = []
            labels_characters = []

            for char_box in letters_image_detected[0].boxes:
                x1_c, y1_c, x2_c, y2_c = map(
                    float,
                    char_box.xyxy[0]
                )
                cls_id = int(char_box.cls[0])
                label = get_class_character(cls_id)
                boxes_characters.append([x1_c, y1_c, x2_c, y2_c])
                labels_characters.append(label)

            license_text = self.license_plate.license_plate_to_text(
                np.array(boxes_characters),
                labels_characters
            )

            dic['labels'] = license_text
            list_license_plate.append(dic)

        return list_license_plate
