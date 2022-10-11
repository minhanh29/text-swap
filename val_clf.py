import torch
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from gen_font_data import TextDataset
from model import FontClassifier
from tqdm import tqdm
import numpy as np

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
checkpoint = torch.load("./weights_mac/6855.pth", map_location="cpu")
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optim'])
global_step = checkpoint["global_step"]

torch.save({
    "model": model.state_dict()
}, "./weights/font_classifier_mac2.pth")

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

loss_func = torch.nn.CrossEntropyLoss()
model.eval()
total_loss = 0.
cnt = 0
true_pos = 0
total_val = 0
for _ in range(3):
    pbar = tqdm(test_dataloader)
    for img, target in pbar:
        img = img.float()
        img = img.float().to(device)
        target = target.to(device)
        pred = model(img)
        logits = pred.detach().numpy()
        logits = np.argmax(logits, axis=-1)
        true_pos += np.count_nonzero(logits == target.numpy())
        total_val += target.numpy().size

        loss = loss_func(pred, target)
        total_loss += loss.item()
        cnt += 1
        pbar.set_postfix({
            "loss": loss.item()
        })

print("True", true_pos)
print("Total", total_val)
print("Acc", true_pos/total_val)
print("Eval loss:", total_loss/cnt)
