import torch
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from gen_font_data import TextDataset
from model import FontClassifier
from tqdm import tqdm
import cfg

torch_blur = transforms.GaussianBlur((5, 5))
torch_v_flip = transforms.RandomVerticalFlip(0.2)
torch_h_flip = transforms.RandomHorizontalFlip(0.2)


def custom_collate(batch):

    img_batch, label_batch = [], []

    w_sum = 0
    for item in batch:

        t_b= item[0]
        h, w = t_b.shape[1:]
        scale_ratio = cfg.data_shape[0] / h
        w_sum += int(w * scale_ratio)

    to_h = cfg.data_shape[0]
    to_w = w_sum // cfg.batch_size
    to_w = int(round(to_w / 8)) * 8
    to_w = max(to_h, to_w)
    to_scale = (to_h, to_w)
    torch_resize = transforms.Resize(to_scale)
    cnt = 0
    for item in batch:
        img, label = item

        if random.uniform(0., 1.) < 0.8:
            p_t = random.randint(0, 70)
            p_b = random.randint(0, 70)
            p_l = random.randint(0, 70)
            p_r = random.randint(0, 70)
            img = torch.nn.functional.pad(img, (p_l, p_r, p_t, p_b))
        img = torch_resize(img)

        if random.uniform(0., 1.) < 0.15:
            img = torch_blur(img)

        if random.uniform(0., 1.) < 0.15:
            img = img + (0.02**0.5)*torch.randn(1, to_h, to_w)
            img = torch.clamp(img, 0., 1.)

        img = torch_h_flip(img)
        img = torch_v_flip(img)

        # img_to_save = F.to_pil_image(img)
        # img_to_save.save(f"./results/text_classification_data/{cnt}.png")
        cnt += 1
        img_batch.append(img)
        label_batch.append(label)

    img_batch = torch.stack(img_batch)
    label_batch = torch.tensor(label_batch)

    return [img_batch, label_batch]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILE_TEXT = "./SRNet-Datagen/Synthtext/data/texts.txt"
FONT_DIR = "./fonts"
FONT_FILE = "./fonts/font_list.txt"

epochs = 3
global_step = 0

test_dataset = TextDataset(FILE_TEXT, FONT_DIR, FONT_FILE, train=False)

num_classes = len(test_dataset.font_list)
model = FontClassifier(1, num_classes).to(device)
checkpoint = torch.load("./weights/font_classifier.pth", map_location="cpu")
model.load_state_dict(checkpoint['model'])

# torch.save({
#     "model": model.state_dict(),
# }, "./weights/font_classifier_win2.pth")

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)

loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    print("Evaluating...")
    pbar = tqdm(test_dataloader)
    model.eval()
    total_loss = 0.
    cnt = 0
    for img, target in pbar:
        img = img.to(device)
        target = target.to(device)
        pred = model(img)

        loss = loss_func(pred, target)
        total_loss += loss.item()
        cnt += 1
        pbar.set_postfix({
            "loss": loss.item()
        })

    print("Eval loss:", total_loss/cnt)
