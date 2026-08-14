import cv2
import numpy as np


def preprocess_frame(frame):
    """
    Convert an OpenCV BGR frame into
    TensorRT input format: (1, 3, 640, 640)
    """

    image = cv2.resize(frame, (640, 640))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32)

    image /= 255.0

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    image = np.ascontiguousarray(image)

    return image