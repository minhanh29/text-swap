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
    print(to_scale)
    torch_resize = transforms.Resize(to_scale)
    torch_blur = transforms.GaussianBlur((5, 5))

    cnt = 0
    for item in batch:
        img, label = item

        img = torch_resize(img)

        if random.random() < 0.15:
            img = torch_blur(img)

        if random.random() < 0.15:
            img = img + (0.02**0.5)*torch.randn(1, to_h, to_w)
            img = torch.clamp(img, 0., 1.)

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

epochs = 5
global_step = 0

train_dataset = TextDataset(FILE_TEXT, FONT_DIR, FONT_FILE, train=True)
test_dataset = TextDataset(FILE_TEXT, FONT_DIR, FONT_FILE, train=False)

num_classes = 206
model = FontClassifier(1, num_classes).to(device)
for p in model.parameters():
    p.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=0.001)
checkpoint = torch.load("./weights_mac/4970.pth", map_location="cpu")
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optim'])
global_step = checkpoint["global_step"]

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, collate_fn=custom_collate)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate)

loss_func = torch.nn.CrossEntropyLoss()
# torch.save({
#     "model": model.state_dict(),
#     "optim": optimizer.state_dict(),
#     "global_step": global_step,
# }, f"./weights/{global_step}.pth")

for epoch in range(epochs):
    print("Training...")
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    pbar = tqdm(train_dataloader)
    pbar.set_description(f"Epoch {epoch}/{epochs}")
    total_loss = 0.
    cnt = 0
    for img, target in pbar:
        global_step += 1
        optimizer.zero_grad()
        img = img.float().to(device)
        target = target.to(device)
        pred = model(img)

        # target = target.float()
        loss = loss_func(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        cnt += 1
        pbar.set_postfix({
            "loss": total_loss/cnt
        })

        if cnt % 200 == 0:
            torch.save({
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "global_step": global_step,
            }, os.path.join("./weights_mac", f"{global_step}.pth"))

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
    torch.save({
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "global_step": global_step,
    }, os.path.join("./weights_mac", f"{global_step}.pth"))
