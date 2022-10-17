import sys
import os
import cv2
import PIL.Image as Image
import numpy as np
import lama
from lama import Inpainter


def load_image(fname, mode='RGB', return_orig=False):
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img


def main():
    inpainter = Inpainter("./weights/lama-fourier")
    img_path = "./custom_feed/my_images/004.png"
    mask_path = "./custom_feed/my_images/004_mask.png"

    img = load_image(img_path, mode="RGB")
    mask = load_image(mask_path, mode="L")

    res_img = inpainter.predict([img], [mask])[0]
    res_img = Image.fromarray(res_img)
    res_img.save("./custom_feed/out.png")


if __name__ == "__main__":
    main()
