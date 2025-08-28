# Detection license plate (version Vietnam)

## Problem

In Vietnam, license plates have 8 or 9 characters, one row or two rows, and their backgrounds have many colors such as
yellow, blue, white,etc..
The problem here is how to detect this, detect character on this, how to arrange these characters in
correct position after detect characters and finally is how to recognize all background color of license plate.

## My solution

The technologies used in this project include Yolo architecture to detect license plates and characters within them,
convolutional
neuron network to classify each character. I used K-means Algorithm to split rows if there are more than
one row and then arrange characters in the correct position.

Frameworks used are:
`Pytorch`,
`OpenCV`,
`Scikit learn`,
`Yolo v11`

## Benefit

Benefit of this program is:

- Program can detect multiple license plates with detection accuracy of up to more than 80% (but characters
  classification in it is so bad).
- Realtime, it can integrate into camera.
- Easy to use.

## Pipeline

```text
Input -> Yolo -> CNN -> K-means -> Result
```

## Install

#### Download repo

```commandline
git clone https://github.com/ntrThanh/detection-license-plate.git
cd detection-license-plate
```

#### Create python environment

```commandline
python -m venv .venv
source .venv/bin/activate
```

#### Install required packages

```commandline
pip install -r requirements.txt
```

## User manual

Run program, you can run by `CLI` or use function `detect_use_image(image_path)` in `detection.py` module.

#### Use command line interface

```commandline
python detection.py -i -p <path_to_file>
```

**Arguments:**

- `-i`, `--image`       : Detect objects in a single image file.
- `-p`, `--path-image`  : Path to the input image.
- `-c`, `--camera`      : Perform real-time detection using a camera.

#### Function:

- Function `detect_use_image(image_path)` in `detection.py` must have input is image can be `.png`, `.jpg`,..., The
  return type is a 2D.
  list of license plates.

## Example

This is a example:
![Example](assets/image/example_1.png)

## Project limitations

- Dataset is not perfect to train so model has overfitting.
- I have many mistakes, It is make plane, resize image, setup data is so bad and more..
- Only detect license plate with background is white and character is black (I will fix it in the future).

## Future

- I will retrain these model if i find new perfect dataset.
- Clean this code, make it is useful, easy to use,..
- Integrate into camera for enhance security, web api,..
- If you have idea, please tell me via email: **nguyentrongthanh672@gmail.com**.

## Reference
- https://docs.ultralytics.com/vi/
- https://pytorch.org/
- https://scikit-learn.org/stable/user_guide.html
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- https://www.kaggle.com/datasets/fareselmenshawii/license-plate-dataset
- https://www.kaggle.com/datasets/aladdinss/license-plate-digits-classification-dataset