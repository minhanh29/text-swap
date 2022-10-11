import os
import argparse
import cfg
import torch
from tqdm import tqdm
import numpy as np
from model import Generator, Discriminator, Vgg19, mask_extraction_net, inpainting_net_mask
from utils import *
from datagen import datagen_srnet, example_dataset, To_tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from nnMorpho.operations import dilation


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def dilation_pytorch(image, strel, origin=(0, 0), border_value=0):
    # first pad the image to have correct unfolding; here is where the origins is used
    image_pad = torch.nn.functional.pad(image, [origin[0], strel.shape[0] - origin[0] - 1, origin[1], strel.shape[1] - origin[1] - 1], mode='constant', value=border_value)
    # Unfold the image to be able to perform operation on neighborhoods
    image_unfold = torch.nn.functional.unfold(image_pad.unsqueeze(0).unsqueeze(0), kernel_size=strel.shape)
    # Flatten the structural element since its two dimensions have been flatten when unfolding
    strel_flatten = torch.flatten(strel).unsqueeze(0).unsqueeze(-1)
    # Perform the greyscale operation; sum would be replaced by rest if you want erosion
    sums = image_unfold + strel_flatten
    # Take maximum over the neighborhood
    result, _ = sums.max(dim=1)
    # Reshape the image to recover initial shape
    return torch.reshape(result, image.shape)


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

    inpaint_net = inpainting_net_mask(in_channels = 4).to(device)
    checkpoint_bg = torch.load("./weights/bg-train_step-36500.model", map_location=torch.device('cpu'))
    inpaint_net.load_state_dict(checkpoint_bg['bg_generator'])

    G = mask_extraction_net(in_channels = 3, get_feature_map=True).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    G.load_state_dict(checkpoint['mask_generator'])

    trfms = To_tensor()
    example_data = example_dataset(data_dir= args.input_dir, transform = trfms)
    example_loader = DataLoader(dataset = example_data, batch_size = 1, shuffle = False)
    example_iter = iter(example_loader)

    print_log('Model compiled.', content_color = PrintColor['yellow'])

    print_log('Predicting', content_color = PrintColor['yellow'])

    K = torch.nn.ZeroPad2d((0, 1, 1, 0))
    G.eval()
    inpaint_net.eval()

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

        o_m, mask_feat = G(i_s)
        o_m = K(o_m)
        # o_m_t = dilation(o_m, torch.tensor(np.ones((3, 3))), origin=(1,1)).float()
        kernel = np.array([ [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1] ], dtype=np.float32)
        kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)) # size: (1, 1, 3, 3)
        # o_m_t = torch.clamp(torch.nn.functional.conv2d(o_m, kernel_tensor, padding=(1, 1)), 0, 1)
        o_m_t = o_m
        o_m = 1. - o_m_t
        mask_feat = mask_feat.detach().to(device)
        o_b, _ = inpaint_net(i_s, o_m, mask_feat)

        o_m = o_m_t.squeeze(0).detach().to('cpu')
        o_b = o_b.squeeze(0).detach().to('cpu')

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        o_m = F.to_pil_image(o_m)
        o_b = F.to_pil_image((o_b + 1)/2)

        o_m.save(os.path.join(args.save_dir, name + 'o_m.png'))
        o_b.save(os.path.join(args.save_dir, name + 'o_b.png'))


if __name__ == '__main__':
    main()
    print_log('predicting finished.', content_color = PrintColor['yellow'])

