# Face Mask Detector using MobileNetV2 TensorFlow Keras Model

This project aims to detect whether a person is wearing a face mask or not using the MobileNetV2 architecture implemented in TensorFlow's Keras.

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [References](#references)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction

In the context of the COVID-19 pandemic, face mask detection has become crucial for ensuring safety and compliance with health regulations. This project provides a solution by employing the MobileNetV2 architecture, which is lightweight and efficient, for detecting whether individuals in images or videos are wearing face masks.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- MobileNetV2 pre-trained model
- OpenCV
- NumPy

## Installation

1. **Python**: If you don't have Python installed, download and install it from [python.org](https://www.python.org/).
2. **TensorFlow**: Install TensorFlow using pip:
    ```
    pip install tensorflow
    ```
3. **Keras**: TensorFlow comes with Keras integrated, but you can install it separately using pip:
    ```
    pip install keras
    ```
4. **MobileNetV2 Pre-trained Model**: Obtain the pre-trained MobileNetV2 model for face mask detection. You can find pre-trained models online or through platforms like GitHub.
5. **OpenCV**: Install OpenCV using pip:
    ```
    pip install opencv-python
    ```
6. **NumPy**: Install NumPy using pip:
    ```
    pip install numpy
    ```

## Usage

1. Clone or download this repository to your local machine.
2. Place the pre-trained MobileNetV2 model file in the project directory.
3. Run the `mask_detector_mobilenetv2.py` script:
    ```
    python classify.py
    ```
4. The script will prompt you to input the path to the image or video file you want to analyze.
5. The program will detect faces in the input image or video and classify them as wearing masks or not.


## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
