import os
import math
from skimage import io
from scipy import stats
import cv2
import argparse
import cfg
import torch
from tqdm import tqdm
import numpy as np
from model import Generator, Discriminator, Vgg19, mask_extraction_net, inpainting_net_mask, fusion_net_alone, FontClassifier
from utils import *
from datagen import datagen_srnet, example_dataset, To_tensor, manual_dataset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR, draw_ocr


height = 64


def preprocess(batch):
    img_batch = []

    w_sum = 0
    for item in batch:
        h, w = item.shape[1:]
        scale_ratio = height / h
        w_sum += int(w * scale_ratio)

    to_h = height
    to_w = w_sum
    to_w = int(round(to_w / 8)) * 8
    to_w = max(to_h, to_w)
    to_scale = (to_h, to_w)
    torch_resize = transforms.Resize(to_scale)
    cnt = 0
    for img in batch:
        img = torch_resize(img)
        img = torch.clamp(img, 0., 1.)
        img_batch.append(img)
        cnt += 1

    img_batch = torch.stack(img_batch)

    return img_batch


def expand_bbox(bbox, img_shape):
    x1 = np.min(bbox[:, 0])
    x2 = np.max(bbox[:, 0])
    y1 = np.min(bbox[:, 1])
    y2 = np.max(bbox[:, 1])

    w = x2 - x1
    h = y2 - y1
    padding_w = int(w * 0.1)
    padding_h = int(h * 0.3)

    x1 -= padding_w
    x2 += padding_w
    y1 -= padding_h
    y2 += padding_h

    x1 = int(max(0, min(x1, img_shape[1]-1)))
    x2 = int(max(0, min(x2, img_shape[1]-1)))
    y1 = int(max(0, min(y1, img_shape[0]-1)))
    y2 = int(max(0, min(y2, img_shape[0]-1)))

    return x1, y1, x2, y2

class ModelFactory:
    def __init__(self, model_dir="./weights", font_dir="./fonts"):
        self.font_dir = font_dir
        FONT_FILE = os.path.join(font_dir, "font_list.txt")
        with open(FONT_FILE, "r", encoding="utf-8") as f:
            font_list = f.readlines()
        self.font_list = [f.strip().split("|")[-1] for f in font_list]

        self.device = torch.device("cpu")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en',
                             det_model_dir=os.path.join(model_dir, "en_PP-OCRv3_det_infer"),
                             rec_model_dir=os.path.join(model_dir, "en_PP-OCRv3_rec_infer"),
                             cls_model_dir=os.path.join(model_dir, "ch_ppocr_mobile_v2.0_cls_infer"))

        self.mask_net = mask_extraction_net(in_channels = 3, get_feature_map=True).to(self.device)
        checkpoint = torch.load(os.path.join(model_dir, "mask_net.pth"), map_location=self.device)
        self.mask_net.load_state_dict(checkpoint['model'])
        self.mask_net.eval()

        self.fusion_net = fusion_net_alone(in_channels = 8).to(self.device)
        checkpoint_bg = torch.load(os.path.join(model_dir, "fusion_net.pth"), map_location=self.device)
        self.fusion_net.load_state_dict(checkpoint_bg['model'])
        self.fusion_net.eval()

        self.font_clf = FontClassifier(in_channels=1, num_classes=len(self.font_list)).to(self.device)
        checkpoint = torch.load(os.path.join(model_dir, "font_classifier.pth"), map_location=self.device)
        self.font_clf.load_state_dict(checkpoint['model'])
        self.font_clf.eval()

        self.K = torch.nn.ZeroPad2d((0, 1, 1, 0))

    def detect_text(self, img):
        '''
        img: BGR image
        '''
        return self.ocr.ocr(img, cls=True)

    def extract_mask(self, img):
        img = preprocess(img)
        mask_s = self.mask_net(img)
        mask_s = self.K(mask_s)
        return mask_s.detach()

    def classify_font(self, img):
        pred = self.font_clf(img)[0]
        pred = pred.detach().numpy()
        chosen = np.argmax(pred, axis=-1)
        font_path = os.path.join(self.font_dir, self.font_list[chosen])
        return font_path

    def fuse_img(self,
                 background_img,
                 source_img,
                 original_text_mask,
                 target_text_mask):
        img = self.fusion_net(torch.cat((background_img,
                                         source_img,
                                         original_text_mask,
                                         target_text_mask), dim=1))
        return self.K(img)


class Roi:
    def __init__(self, sub_img, bbox, text, angle, model_factory):
        self.img = sub_img
        self.bbox = bbox
        self.text = text
        self.angle = angle
        self.model_factory = model_factory
        self.torch_mask, self.mask = self.extract_mask()

    def extract_mask(self):
        torch_img = torch.as_tensor(self.img)
        torch_img = torch.permute(torch_img, (2, 0, 1))
        with torch.no_grad():
            torch_mask = self.model_factory.extract_mask([torch_img])
        torch_mask = torch_mask.detach()[0]
        mask = torch.permute(torch_mask, (1, 2, 0)).numpy()
        mask = (mask * 255).astype("uint8")
        _, mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.resize(mask, (self.img.shape[1], self.img.shape[0]))

        return torch_mask, mask

    def __str__(self):
        x1, y1, x2, y2 = self.bbox
        result = f"""
        =====================
        Bbox: ({x1}, {y1}, {x2}, {y2})
        Text: {self.text}
        Angle: {self.angle}
        """
        return result

    def __repr__(self):
        return self.__str__()


class TextSwapper:
    def __init__(self, model_factory, img):
        self.model_factory = model_factory
        self.img = img
        self.rois = []

    def detect_text(self):
        result = self.model_factory.detect_text(self.img)
        for item in result:
            print(item)
            bbox, text, angle = item
            if text[1] < 0.9:
                continue
            bbox = np.array(bbox)
            x1, y1, x2, y2 = expand_bbox(bbox, self.img.shape)
            sub_img = self.img[y1:y2, x1:x2].copy()
            roi = Roi(sub_img, [x1, y1, x2, y2],
                      text[0], angle[0], self.model_factory)
            self.rois.append(roi)
        print(self.rois)



if __name__ == "__main__":
    model_factory = ModelFactory("./weights", "./fonts")
    img = cv2.imread("./custom_feed/sample1.png")
    text_swapper = TextSwapper(model_factory, img)
    text_swapper.detect_text()
