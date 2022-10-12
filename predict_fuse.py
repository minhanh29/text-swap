import os
import math
from scipy import stats
import cv2
import argparse
import cfg
import torch
from tqdm import tqdm
import numpy as np
from model import Generator, Discriminator, Vgg19, mask_extraction_net, inpainting_net_mask, fusion_net_alone, FontClassifier
from utils import *
from datagen import datagen_srnet, example_dataset, To_tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = "./fonts"
FONT_FILE = "./fonts/font_list.txt"
with open(FONT_FILE, "r", encoding="utf-8") as f:
    font_list = f.readlines()
font_list = [f.strip().split("|")[-1] for f in font_list]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
pil_to_tensor = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Grayscale(1)
])

CHAR_SIZE = 48
INNER_SIZE = CHAR_SIZE - 20
expand_char = 1

def crop_char(mask, bboxes):
    h, w = mask.shape
    batch = []
    cnt = 0
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(x1 - expand_char, w-1))
        y1 = max(0, min(y1 - expand_char, h-1))
        x2 = max(0, min(x2 + expand_char, w))
        y2 = max(0, min(y2 + expand_char, h))
        crop_img = mask[y1:y2, x1:x2]

        mh, mw = crop_img.shape
        target_shape = (INNER_SIZE, INNER_SIZE)
        if mh > mw:
            target_shape = (INNER_SIZE, int(INNER_SIZE * mw / mh))
        else:
            target_shape = (int(INNER_SIZE * mh / mw), INNER_SIZE)

        crop_img = cv2.resize(crop_img, target_shape)
        mh, mw = crop_img.shape

        # pad image to have shape CHAR_SIZE x CHAR_SIZE
        crop_img = torch.from_numpy(crop_img)
        p_t = (CHAR_SIZE - mh)//2
        p_b = CHAR_SIZE - mh - p_t
        p_l = (CHAR_SIZE - mw)//2
        p_r = CHAR_SIZE - mw - p_l
        crop_img1 = torch.nn.functional.pad(crop_img, (p_l, p_r, p_t, p_b)).float() / 255.
        # crop_img2 = torch.nn.functional.pad(crop_img, (p_l, p_r, CHAR_SIZE - mh, 0)).float() / 255.
        pil_img = F.to_pil_image(crop_img1)
        pil_img.save(f"./custom_feed/result/test{cnt}.png")
        cnt += 1
        batch.append(crop_img1)
    return torch.unsqueeze(torch.stack(batch, dim=0), dim=1)


def segment_mask(mask, idx):
    mask = np.squeeze(mask) * 255
    mask = mask.astype("uint8")
    # adapt_thresh = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, -2)
    kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations=1)
    # mask = cv2.erode(mask, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    image_copy = np.stack([mask, mask, mask], axis=-1)
    areas = []
    bboxes = []
    mx1, my1, mx2, my2 = mask.shape[1], mask.shape[0], 0, 0
    for cnt in contours:
        cnt = np.squeeze(cnt, axis=1)
        x1 = np.min(cnt[:, 0])
        x2 = np.max(cnt[:, 0])
        y1 = np.min(cnt[:, 1])
        y2 = np.max(cnt[:, 1])
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 255), 1)
        areas.append(abs(y2-y1)*(x2-x1))
        bboxes.append([x1, y1, x2, y2])

        # overall bbox
        mx1 = min(mx1, x1)
        my1 = min(my1, y1)
        mx2 = max(mx2, x2)
        my2 = max(my2, y2)

    mean_area = np.median(areas)
    clean_bboxes = []
    for bbox, area in zip(bboxes, areas):
        if area > mean_area * 1.7 or area < mean_area * 0.5:
            continue
        clean_bboxes.append(bbox)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.rectangle(image_copy, (mx1, my1), (mx2, my2), (0, 255, 0), 1)
    cv2.imwrite(f"./custom_feed/result/contours{idx}.png", image_copy)

    return crop_char(mask, clean_bboxes), (mx1, my1, mx2, my2)


def gen_data_sample(text, font_path, canvas_width, canvas_height, target_w, target_h):
    shape = (canvas_width, canvas_height)
    padding = 0.1
    border = int(min(shape) * padding)
    # target_shape = tuple(np.array(shape) - 2 * border)
    target_shape = (target_w, target_h)

    fontsize = 12
    pre_remain = None
    while True:
        # get text bbox
        img_center = (canvas_width//2, canvas_height//2)
        img = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        myFont = ImageFont.truetype(font_path, fontsize)
        draw.text(img_center, text, font=myFont, fill=(255, 255, 255), anchor="mm")
        rect = img.getbbox()

        res_shape = (int(rect[2] - rect[0]), int(rect[3] - rect[1]))
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
    myFont = ImageFont.truetype(font_path, fontsize)
    draw.text(img_center, text, font=myFont, fill=(255, 255, 255), anchor="mm")
    return pil_to_tensor(img).float() / 255.


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help = 'Directory containing xxx_i_s and xxx_i_t with same prefix',
                        default = cfg.example_data_dir)
    parser.add_argument('--save_dir', help = 'Directory to save result', default ="./custom_feed/result")
    parser.add_argument('--checkpoint', help = 'ckpt', default = "./weights/mask-train_step-50000.model")
    args = parser.parse_args()

    assert args.input_dir is not None
    assert args.save_dir is not None
    assert args.checkpoint is not None

    print_log('model compiling start.', content_color = PrintColor['yellow'])

    fusion_net = fusion_net_alone(in_channels = 8).to(device)
    checkpoint_bg = torch.load("./weights/fusion_net.pth", map_location=torch.device('cpu'))
    fusion_net.load_state_dict(checkpoint_bg['model'])

    inpaint_net = inpainting_net_mask(in_channels = 4).to(device)
    checkpoint_bg = torch.load("./weights/inpainting_net.pth", map_location=torch.device('cpu'))
    inpaint_net.load_state_dict(checkpoint_bg['model'])

    mask_net = mask_extraction_net(in_channels = 3, get_feature_map=True).to(device)
    checkpoint = torch.load("./weights/mask_net.pth", map_location=torch.device('cpu'))
    mask_net.load_state_dict(checkpoint['model'])

    font_clf = FontClassifier(in_channels=1, num_classes=206).to(device)
    checkpoint = torch.load("./weights/font_classifier_mac.pth", map_location=torch.device('cpu'))
    font_clf.load_state_dict(checkpoint['model'])

    trfms = To_tensor()
    example_data = example_dataset(data_dir= args.input_dir, transform = trfms)
    example_loader = DataLoader(dataset = example_data, batch_size = 1, shuffle = False)
    example_iter = iter(example_loader)

    print_log('Model compiled.', content_color = PrintColor['yellow'])

    print_log('Predicting', content_color = PrintColor['yellow'])

    K = torch.nn.ZeroPad2d((0, 1, 1, 0))
    mask_net.eval()
    inpaint_net.eval()
    fusion_net.eval()
    font_clf.eval()

    with torch.no_grad():
      for step in tqdm(range(len(example_data))):
        try:
          inp = example_iter.next()
        except StopIteration:

          example_iter = iter(example_loader)
          inp = example_iter.next()

        i_t = inp[0].to(device)
        i_s = inp[1].to(device)
        name = str(inp[2][0])

        print(name)
        o_m, mask_feat = mask_net(i_s)
        o_m = K(o_m)
        o_m_t = o_m
        o_m = 1. - o_m_t
        mask_feat = mask_feat.detach().to(device)
        o_b, _ = inpaint_net(i_s, o_m, mask_feat)
        o_b = K(o_b)

        batch_char, bbox = segment_mask(o_m_t.numpy()[0], step)
        chosen = 0
        if len(batch_char) > 0:
            print(torch.min(batch_char))
            font_pred = font_clf(batch_char)
            font_pred = font_pred.numpy()
            font_pred = np.squeeze(font_pred)
            indices = np.argmax(font_pred, axis=-1)
            sum_arr = np.sum(font_pred, axis=0)
            print(sum)
            print(np.max(font_pred, axis=-1))
            print(indices)
            # chosen = stats.mode(indices).mode[0]
            chosen = np.argmax(sum_arr)
            print(chosen, font_list[chosen])

        font_path = os.path.join(FONT_DIR, font_list[chosen])

        target_w = bbox[2] - bbox[0]
        target_h = bbox[3] - bbox[1]
        mask_t = gen_data_sample("nguyễn", font_path, o_b.shape[3], o_b.shape[2], target_w, target_h)
        mask_t = torch.unsqueeze(mask_t, dim=0)

        o_f = fusion_net(torch.cat((o_b, i_s, o_m_t, mask_t), dim=1))

        o_m = o_m_t.squeeze(0).detach().to('cpu')
        o_b = o_b.squeeze(0).detach().to('cpu')
        o_f = o_f.squeeze(0).detach().to('cpu')

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        o_m = F.to_pil_image(o_m)
        o_b = F.to_pil_image((o_b + 1)/2)
        o_f = F.to_pil_image((o_f + 1)/2)

        o_m.save(os.path.join(args.save_dir, name + 'o_m.png'))
        o_b.save(os.path.join(args.save_dir, name + 'o_b.png'))
        o_f.save(os.path.join(args.save_dir, name + 'o_f.png'))


if __name__ == '__main__':
    main()
    print_log('predicting finished.', content_color = PrintColor['yellow'])


