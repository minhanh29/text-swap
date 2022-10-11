from PIL import Image, ImageDraw, ImageFont
import string
import os
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

FILE_TEXT = "./SRNet-Datagen/Synthtext/data/texts.txt"
FONT_DIR = "./fonts"
FONT_FILE = "./fonts/font_list.txt"
OUT_DIR = "./results/text_classification_data"
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)


class TextDataset(Dataset):
    def __init__(self, file_text, font_dir, font_file, train=True):
        with open(file_text, "r", encoding="utf-8") as f:
            text_list = f.readlines()
        self.text_list = [line.strip() for line in text_list]
        print("Got", len(self.text_list), "words")

        self.font_dir = font_dir
        with open(font_file, "r", encoding="utf-8") as f:
            font_list = f.readlines()
        self.font_list = [f.strip().split("|") for f in font_list]
        print("Got", len(self.font_list), "fonts")

        self.org_canvas_width = 150
        self.org_canvas_height = 150
        self.shape = (self.org_canvas_width, self.org_canvas_height)
        self.padding = 20

        self.fontsize = 50
        self.img_center = (self.org_canvas_width//2, self.org_canvas_height//2)
        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Grayscale(1),
            transforms.Resize(size=40, max_size=48)
        ])

        self.train = train

        s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
        s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
        self.char_list = s0 + s1

    def __len__(self):
        if self.train:
            return len(self.text_list)
        else:
            return 640

    def __getitem__(self, idx):
        text = random.choice(self.char_list)
        label, font_path = random.choice(self.font_list)
        font_path = os.path.join(self.font_dir, font_path)
        img = self.gen_data_sample(text, font_path, idx)

        p_t = (48 - img.shape[1])//2
        p_b = 48 - img.shape[1] - p_t
        p_l = (48 - img.shape[2])//2
        p_r = 48 - img.shape[2] - p_l
        img = torch.nn.functional.pad(img, (p_l, p_r, p_t, p_b))
        return img, int(label)

    def gen_data_sample(self, text, font_path, idx):
        img = Image.new('RGB', self.shape, (0, 0, 0))
        draw = ImageDraw.Draw(img)
        myFont = ImageFont.truetype(font_path, self.fontsize)
        draw.text(self.img_center, text, font=myFont, fill=(255, 255, 255), anchor="mm")
        rect = img.getbbox()

        canvas_width, canvas_height = (int(rect[2] - rect[0]), int(rect[3] - rect[1]))
        canvas_width += self.padding
        canvas_height += self.padding

        canvas_width = int(canvas_width)
        canvas_height = int(canvas_height)

        img = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Custom font style and font size
        draw.text((canvas_width//2, canvas_height//2), text,
                  font=myFont, fill=(255, 255, 255), anchor="mm")

        if random.random() < 0.3:
            img = img.rotate(random.randint(-40, 40), expand=True)

        img.save(os.path.join(OUT_DIR, f"{idx}.png"))

        return self.transform(img)


def main():
    dataset = TextDataset(FILE_TEXT, FONT_DIR, FONT_FILE)
    for cnt in range(10):
        dataset.__getitem__(cnt)


if __name__ == "__main__":
    main()
