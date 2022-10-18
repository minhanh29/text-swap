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
from model import mask_extraction_net, fusion_net, FontClassifier
from utils import *
from datagen import datagen_srnet, example_dataset, To_tensor, manual_dataset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR, draw_ocr
from lama import Inpainter


height = 64
PADDING = 3

pil_to_tensor = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Grayscale(1)
])


def make_mask(mask_margin, width, height):
    mask_margin = 5
    mask = np.zeros((height, width))
    top_mask = np.linspace(1, 0, mask_margin)
    top_mask = np.repeat(np.expand_dims(top_mask, axis=-1), width, axis=-1)
    bottom_mask = np.linspace(0, 1, mask_margin)
    bottom_mask = np.repeat(np.expand_dims(bottom_mask, axis=-1), width, axis=-1)
    left_mask = np.linspace(1, 0, mask_margin)
    left_mask = np.repeat(np.expand_dims(left_mask, axis=0), height, axis=0)
    right_mask = np.linspace(0, 1, mask_margin)
    right_mask = np.repeat(np.expand_dims(right_mask, axis=0), height, axis=0)

    mask[:mask_margin, :] = top_mask
    mask[-mask_margin:, :] = bottom_mask
    mask[:, :mask_margin] = np.maximum(mask[:, :mask_margin], left_mask)
    mask[:, -mask_margin:] = np.maximum(mask[:, -mask_margin:], right_mask)
    mask = np.expand_dims(mask, axis=-1)
    mask = np.repeat(mask, 3, axis=-1)

    return mask


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
        img_batch.append(img)
        cnt += 1

    img_batch = torch.stack(img_batch)

    return img_batch


def expand_bbox(bbox, img_shape, w_ratio=0.1, h_ratio=0.3):
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1
    padding_w = int(w * w_ratio)
    padding_h = int(h * h_ratio)

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
                             # det_model_dir=os.path.join(model_dir, "en_PP-OCRv3_det_infer"),
                             # rec_model_dir=os.path.join(model_dir, "en_PP-OCRv3_rec_infer"),
                             det_model_dir=os.path.join(model_dir, "ch_ppocr_server_v2.0_det_infer"),
                             rec_model_dir=os.path.join(model_dir, "en_PP-OCRv3_rec_infer"),
                             cls_model_dir=os.path.join(model_dir, "ch_ppocr_mobile_v2.0_cls_infer"))

        self.mask_net = mask_extraction_net(in_channels = 3).to(self.device)
        checkpoint = torch.load(os.path.join(model_dir, "mask_net.pth"), map_location=self.device)
        self.mask_net.load_state_dict(checkpoint['model'])
        self.mask_net.eval()

        self.fusion_net = fusion_net(in_channels = 8).to(self.device)
        checkpoint_bg = torch.load(os.path.join(model_dir, "fusion_net.pth"), map_location=self.device)
        self.fusion_net.load_state_dict(checkpoint_bg['model'])
        self.fusion_net.eval()

        self.inpainter = Inpainter(os.path.join(model_dir, "lama-fourier"))

        self.font_clf = FontClassifier(in_channels=1, num_classes=len(self.font_list)).to(self.device)
        checkpoint = torch.load(os.path.join(model_dir, "font_classifier.pth"), map_location=self.device)
        self.font_clf.load_state_dict(checkpoint['model'])
        self.font_clf.eval()

        self.K = torch.nn.ZeroPad2d((0, 1, 1, 0))
        self.my_pad = torch.nn.ZeroPad2d((3, 3, 3, 3))

    def detect_text(self, img):
        '''
        img: BGR image
        '''
        return self.ocr.ocr(img, cls=True)

    def extract_background(self, img_list, mask_list):
        new_img_list = []
        new_mask_list = []
        for img, mask in zip(img_list, mask_list):
            img = np.transpose(img, (2, 0, 1))
            img = img.astype('float32') / 255
            mask = mask.astype('float32') / 255

            new_img_list.append(img)
            new_mask_list.append(mask)

        return self.inpainter.predict(new_img_list, new_mask_list)

    def extract_mask(self, img):
        img = img/127.5 - 1
        img = preprocess(img)
        mask_s = self.mask_net(img)
        mask_s = self.K(mask_s)
        return mask_s.detach()

    def detect_font(self, img):
        pred = self.font_clf(img)
        pred = pred.detach().numpy()
        chosen = np.argmax(pred, axis=-1)
        font_path = []
        for idx in chosen:
            font_path.append(os.path.join(self.font_dir, self.font_list[int(idx)]))
        return font_path

    def fuse_img(self,
                 background_img,
                 source_img,
                 original_text_mask,
                 target_text_mask):

        background_img = background_img.float() / 127.5 - 1
        source_img = source_img.float() / 127.5 - 1

        background_img = preprocess(background_img)
        source_img = preprocess(source_img)
        original_text_mask = preprocess(original_text_mask)
        target_text_mask = preprocess(target_text_mask)

        img = self.fusion_net(torch.cat((background_img,
                                         source_img,
                                         original_text_mask,
                                         target_text_mask), dim=1))
        return self.K(img)


class BBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def get(self):
        return self.x1, self.y1, self.x2, self.y2

    def abs_inner(self, innerBox):
        x1, y1, x2, y2 = innerBox.get()
        return BBox(self.x1 + x1, self.y1 + y1, self.x1 + x2, self.y1 + y2)

    def __str__(self):
        return f"({self.x1}, {self.y1}, {self.x2}, {self.y2})"

    def __repr__(self):
        return self.__str__()


class Roi:
    def __init__(self, id, sub_img, bbox, text, angle, model_factory):
        self.id = id

        self.padding = 3
        # sub_img[:self.padding] = 0
        # sub_img[-self.padding:] = 0
        # sub_img[:, :self.padding] = 0
        # sub_img[:, -self.padding:] = 0

        self.img = sub_img
        torch_img = torch.as_tensor(self.img)
        torch_img = torch.permute(torch_img, (2, 0, 1))
        self.torch_img = torch_img.float()

        self.bbox = BBox(*bbox)
        self.text = text
        self.angle = angle
        self.model_factory = model_factory

        self.bg_torch_img = sub_img

        self.extract_mask()
        self.find_bounding_rect()
        self.detect_font()
        self.create_target_text("Minh Anh")

    def update_bg(self, new_full_img):
        x1, y1, x2, y2 = self.bbox.get()
        self.bg_img = new_full_img[y1:y2, x1:x2].copy()
        bg_img = new_full_img[y1:y2, x1:x2].copy()
        # bg_img[:self.padding] = 0
        # bg_img[-self.padding:] = 0
        # bg_img[:, :self.padding] = 0
        # bg_img[:, -self.padding:] = 0
        bg_torch_img = torch.as_tensor(bg_img)
        bg_torch_img = torch.permute(bg_torch_img, (2, 0, 1))
        self.bg_torch_img = bg_torch_img.float()

    def extract_mask(self):
        with torch.no_grad():
            torch_mask = self.model_factory.extract_mask(torch.unsqueeze(self.torch_img, dim=0))[0]
        torch_mask = torch_mask.detach()
        mask = torch.permute(torch_mask, (1, 2, 0)).numpy()
        mask = (mask * 255).astype("uint8")
        cv2.imwrite(f"./custom_feed/{self.text.split('/')[0]}.png", mask)
        mask[mask > 120] = 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.resize(mask, (self.img.shape[1], self.img.shape[0]))

        torch_blur = transforms.GaussianBlur((3, 3))
        self.torch_mask = torch_blur(torch_mask)
        self.mask = mask

    def find_bounding_rect(self):
        x, y, w, h = cv2.boundingRect(self.mask)
        x1, y1, x2, y2 = expand_bbox([x, y, x+w, y+h], self.mask.shape,
                                     w_ratio=0.05, h_ratio=0.1)
        self.minBbox = BBox(x1, y1, x2, y2)

    def get_abs_inner_box(self):
        x1, y1, x2, y2 = self.minBbox.get()
        img = self.mask[y1:y2, x1:x2]

        return img, self.bbox.abs_inner(self.minBbox)

    def detect_font(self):
        self.font_path = self.model_factory.detect_font(torch.unsqueeze(self.torch_mask, dim=0))[0]
        print(self.font_path)

    def create_target_text(self, text, angle=0):
        # canvas_width = self.torch_mask.shape[2]
        # canvas_height = self.torch_mask.shape[1]
        canvas_width = self.img.shape[1]
        canvas_height = self.img.shape[0]
        shape = (canvas_width, canvas_height)
        target_w = int(canvas_width * 0.9)
        target_h = int(canvas_height*0.7)
        target_shape = (target_w, target_h)

        fontsize = 30
        pre_remain = None
        text_h = target_h
        while True:
            # get text bbox
            img_center = (canvas_width//2, canvas_height//2)
            img = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
            draw = ImageDraw.Draw(img)
            myFont = ImageFont.truetype(self.font_path, fontsize)
            draw.text(img_center, text, font=myFont, fill=(255, 255, 255), anchor="mm")
            rect = img.getbbox()

            res_shape = (int(rect[2] - rect[0]), int(rect[3] - rect[1]))
            text_h = res_shape[1]
            remain = np.min(np.array(target_shape) - np.array(res_shape))
            if pre_remain is not None:
                m = pre_remain * remain
                if m <= 0:
                    if m < 0 and remain < 0:
                        fontsize -= 1
                    if m == 0 and remain != 0:
                        if remain < 0:
                            fontsize -= 1
                        elif remain > 0:
                            fontsize += 1
                    break
            if remain < 0:
                if fontsize == 2:
                    break
                fontsize -= 1
            else:
                fontsize += 1
            pre_remain = remain

        img_center = (canvas_width//2, canvas_height//2)
        img = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Custom font style and font size
        myFont = ImageFont.truetype(self.font_path, fontsize)
        draw.text((canvas_width//2, text_h + (canvas_height-text_h)//2), text, font=myFont, fill=(255, 255, 255), anchor="mb")
        # img.save(f"./custom_feed/{self.text}.png")
        # img = img.rotate(angle, expand=False)
        self.target_mask_np = np.array(img)
        res_mask = img.resize((self.torch_mask.shape[2], self.torch_mask.shape[1]))
        self.target_mask = pil_to_tensor(res_mask).float() / 255.

    def fuse_img(self):
        fuse_img = self.model_factory.fuse_img(torch.unsqueeze(self.bg_torch_img, dim=0),
                                               torch.unsqueeze(self.torch_img, dim=0),
                                               torch.unsqueeze(self.torch_mask, dim=0),
                                               torch.unsqueeze(self.target_mask, dim=0))
        fuse_img = fuse_img.squeeze(0).detach().to('cpu')
        fuse_img = F.to_pil_image((fuse_img + 1)/2)
        fuse_img = np.array(fuse_img)
        fuse_img = cv2.resize(fuse_img, (self.bg_img.shape[1], self.bg_img.shape[0]))

        target_mask_np = self.target_mask_np.astype("float32")/255.
        fuse_img = fuse_img * target_mask_np + (1 - target_mask_np) * self.bg_img
        self.fuse_img = fuse_img

    def __str__(self):
        result = f"""

        =====================
        Bbox: {self.bbox}
        Min Bbox: {self.minBbox}
        Text: {self.text}
        Angle: {self.angle}
        """
        # cv2.imwrite(f"./custom_feed/{self.id}.png", self.img)
        # cv2.imwrite(f"./custom_feed/{self.id}_mask.png", self.mask)
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
        for i, item in enumerate(result):
            print(item)
            boxes, text, angle = item
            if text[1] < 0.8:
                continue
            boxes = np.array(boxes)
            x1 = np.min(boxes[:, 0])
            x2 = np.max(boxes[:, 0])
            y1 = np.min(boxes[:, 1])
            y2 = np.max(boxes[:, 1])

            x1, y1, x2, y2 = expand_bbox([x1, y1, x2, y2], self.img.shape)
            sub_img = self.img[y1:y2, x1:x2].copy()
            roi = Roi(i, sub_img, [x1, y1, x2, y2],
                      text[0], angle[0], self.model_factory)
            self.rois.append(roi)
        print(self.rois)

    def create_mask(self):
        mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype="uint8")
        total_area = mask.shape[0] * mask.shape[1]
        for roi in self.rois:
            img, bbox = roi.get_abs_inner_box()
            x1, y1, x2, y2 = bbox.get()
            area = (y2-y1)*(x2-x1)
            if area/total_area < 0.01 or (y2-y1)/mask.shape[0] < 0.06 or (x2-x1)/mask.shape[1] < 0.05:
                x1, y1, x2, y2 = roi.bbox.get()
                mask[y1:y2, x1:x2] = 255
            else:
                mask[y1:y2, x1:x2] = img
        cv2.imwrite(f"./custom_feed/poster_mask.png", mask)
        return mask

    def extract_background(self):
        mask = self.create_mask()
        res_img = self.model_factory.extract_background([self.img], [mask])
        res_img = res_img[0]
        self.bg_img = res_img
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("./custom_feed/poster_bg.png", res_img)

    def update_bg(self):
        for roi in self.rois:
            roi.update_bg(self.bg_img)

    def fuse_img(self):
        for roi in self.rois:
            roi.fuse_img()

        result_img = self.bg_img.copy()
        for roi in self.rois:
            x1, y1, x2, y2 = roi.bbox.get()
            result_img[y1:y2, x1:x2] = roi.fuse_img
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"./custom_feed/poster_fuse.png", result_img)
        return result_img


if __name__ == "__main__":
    model_factory = ModelFactory("./weights", "./fonts")
    img = cv2.imread("./custom_feed/poster.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    text_swapper = TextSwapper(model_factory, img)
    text_swapper.detect_text()
    text_swapper.extract_background()
    text_swapper.update_bg()
    text_swapper.fuse_img()
