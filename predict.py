import os
import math
from scipy import stats
import cv2
import argparse
import cfg
import torch
from tqdm import tqdm
import numpy as np
from model import mask_extraction_net, inpainting_net_mask, fusion_net, FontClassifier
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

PADDING = 3


def remove_pad(image):
    w, h = image.size
    image = image.crop((PADDING, PADDING, w-PADDING, h-PADDING))
    return image


def segment_mask(mask, idx):
    mask = np.squeeze(mask) * 255
    mask = mask.astype("uint8")
    coords = np.column_stack(np.where(mask > 10))
    y, x, h, w = cv2.boundingRect(coords)
    # center, (width, height), angle = cv2.minAreaRect(coords)
    # box = cv2.boxPoints(cv2.minAreaRect(coords))
    # box = np.int32(box)
    # box = np.flip(box, 1)
    image_copy = np.stack([mask, mask, mask], axis=-1)
    # cv2.line(image_copy, box[0], box[1], (0, 255, 255), 1)
    # cv2.line(image_copy, box[1], box[2], (0, 255, 255), 1)
    # cv2.line(image_copy, box[2], box[3], (0, 255, 255), 1)
    # cv2.line(image_copy, box[3], box[0], (0, 255, 255), 1)
    cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0,255,255), 1)
    cv2.imwrite(f"./custom_feed/result/contours{idx}.png", image_copy)
    # if angle > 45:
    #     angle = -(90 - angle)
    return w, h, 0


def gen_data_sample(text, font_path, canvas_width, canvas_height, target_w, target_h, angle=0):
    shape = (canvas_width, canvas_height)
    target_shape = (target_w, target_h)

    fontsize = 30
    pre_remain = None
    text_h = target_h
    while True:
        # get text bbox
        img_center = (canvas_width//2, canvas_height//2)
        img = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        myFont = ImageFont.truetype(font_path, fontsize)
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
    myFont = ImageFont.truetype(font_path, fontsize)
    draw.text((canvas_width//2, text_h + (canvas_height-text_h)//2), text, font=myFont, fill=(255, 255, 255), anchor="mb")
    # img = img.rotate(angle, expand=False)
    return pil_to_tensor(img).float() / 255.


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help = 'Directory containing xxx_i_s and xxx_i_t with same prefix',
                        default = cfg.example_data_dir)
    parser.add_argument('--save_dir', help = 'Directory to save result', default ="./custom_feed/result")
    args = parser.parse_args()

    print_log('model compiling start.', content_color = PrintColor['yellow'])

    fusion_net = fusion_net(in_channels = 8).to(device)
    checkpoint_bg = torch.load("./weights/fusion_net.pth", map_location=torch.device('cpu'))
    fusion_net.load_state_dict(checkpoint_bg['model'])

    inpaint_net = inpainting_net_mask(in_channels = 4).to(device)
    checkpoint_bg = torch.load("./weights/inpainting_net.pth", map_location=torch.device('cpu'))
    inpaint_net.load_state_dict(checkpoint_bg['model'])

    mask_net = mask_extraction_net(in_channels = 3, get_feature_map=True).to(device)
    checkpoint = torch.load("./weights/mask_net.pth", map_location=torch.device('cpu'))
    mask_net.load_state_dict(checkpoint['model'])

    font_clf = FontClassifier(in_channels=1, num_classes=len(font_list)).to(device)
    checkpoint = torch.load("./weights/font_classifier.pth", map_location=torch.device('cpu'))
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

    torch_blur = transforms.GaussianBlur((3, 3))
    torch_resize = transforms.Resize(size=64)
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

        img_blur = torch.permute(o_m_t[0], (1, 2, 0))
        img_blur = cv2.blur(img_blur.numpy(), (2, 2))
        h = img_blur.shape[0]
        w = img_blur.shape[1]
        bw = int(w*0.15)
        bh = int(h*0.15)
        img_blur = cv2.copyMakeBorder(img_blur, bh, bh, bw, bw, cv2.BORDER_CONSTANT, None, (0, 0, 0))
        img_blur = cv2.resize(img_blur, (w, h))

        o_m_blur = torch.tensor(np.expand_dims(img_blur, axis=-1))
        o_m_blur = torch.permute(o_m_blur, (2, 0, 1))
        o_m_blur = torch.unsqueeze(o_m_blur, dim=0)
        font_pred = font_clf(o_m_blur)
        font_pred = font_pred.detach().numpy()
        chosen = np.argmax(font_pred, axis=-1)[0]
        print(font_list[chosen])

        target_w, target_h, angle = segment_mask(o_m_t.numpy()[0], step)
        target_w = int(0.8 * o_b.shape[3])
        font_path = os.path.join(FONT_DIR, font_list[chosen])
        mask_t = gen_data_sample("Xin chào", font_path, o_b.shape[3], o_b.shape[2], target_w, target_h, 0)
        mask_t = torch.unsqueeze(mask_t, dim=0)

        o_f = fusion_net(torch.cat((o_b, i_s, o_m_t, mask_t), dim=1))
        o_f = K(o_f)

        i_s = i_s.squeeze(0).detach().to('cpu')
        o_m = o_m_t.squeeze(0).detach().to('cpu')
        o_b = o_b.squeeze(0).detach().to('cpu')
        o_f = o_f.squeeze(0).detach().to('cpu')

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        o_m = F.to_pil_image(o_m)
        o_b = F.to_pil_image((o_b + 1)/2)
        o_f = F.to_pil_image((o_f + 1)/2)
        i_s = F.to_pil_image((i_s + 1)/2)

        o_m = remove_pad(o_m)
        o_b = remove_pad(o_b)
        o_f = remove_pad(o_f)
        i_s = remove_pad(i_s)

        o_m = np.array(o_m)
        _, o_m = cv2.threshold(o_m, 127,255,cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        print(o_m.shape)
        o_m = cv2.dilate(o_m, kernel, iterations=2)
        cv2.imwrite(os.path.join(args.save_dir, name + 'o_m.png'), o_m)
        # o_m.save(os.path.join(args.save_dir, name + 'o_m.png'))
        o_b.save(os.path.join(args.save_dir, name + 'o_b.png'))
        o_f.save(os.path.join(args.save_dir, name + 'o_f.png'))
        i_s.save(os.path.join(args.save_dir, name + 'i_s.png'))


if __name__ == '__main__':
    main()
    print_log('predicting finished.', content_color = PrintColor['yellow'])


