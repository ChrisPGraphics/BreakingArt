import logging

import cv2
import numpy as np


def load_image(path: str, grayscale: bool = False, normalize: bool = True) -> np.ndarray:
    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError("Image '{}' does not exist!".format(path))

    if grayscale:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if normalize:
        return (img / 255).astype(np.float32)

    else:
        return img
