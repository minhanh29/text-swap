# Predict script
# author: Niwhskal

import os
import argparse
import cfg
import torch
from tqdm import tqdm
import numpy as np
from model import Generator, Discriminator, Vgg19, mask_extraction_net
from utils import *
from datagen import datagen_srnet, example_dataset, To_tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help = 'Directory containing xxx_i_s and xxx_i_t with same prefix',
                        default = cfg.example_data_dir)
    parser.add_argument('--save_dir', help = 'Directory to save result', default ="./results/mask")
    parser.add_argument('--checkpoint', help = 'ckpt', default = "./weights/mask-train_step-50000.model")
    args = parser.parse_args()

    assert args.input_dir is not None
    assert args.save_dir is not None
    assert args.checkpoint is not None

    print_log('model compiling start.', content_color = PrintColor['yellow'])

    G = mask_extraction_net(in_channels = 3, get_feature_map=True).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    G.load_state_dict(checkpoint['mask_generator'])

    trfms = To_tensor()
    example_data = example_dataset(data_dir= args.input_dir, transform = trfms)
    example_loader = DataLoader(dataset = example_data, batch_size = 1, shuffle = False)
    example_iter = iter(example_loader)

    print_log('Model compiled.', content_color = PrintColor['yellow'])

    print_log('Predicting', content_color = PrintColor['yellow'])

    G.eval()

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

        o_m, _ = G(i_s)

        o_m = o_m.squeeze(0).detach().to('cpu')

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        o_m = F.to_pil_image(o_m)

        o_m.save(os.path.join(args.save_dir, name + 'o_m.png'))


if __name__ == '__main__':
    main()
    print_log('predicting finished.', content_color = PrintColor['yellow'])

