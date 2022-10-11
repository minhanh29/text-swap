import os
import argparse
import cfg
import torch
from tqdm import tqdm
import numpy as np
from model import Generator, Discriminator, Vgg19, mask_extraction_net, inpainting_net_mask, fusion_net_alone
from utils import *
from datagen import datagen_srnet, example_dataset, To_tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from nnMorpho.operations import dilation


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def gen_data_sample(text, font_path, canvas_width, canvas_height):
    shape = (canvas_width, canvas_height)
    padding = 0.1
    border = int(min(shape) * padding)
    target_shape = tuple(np.array(shape) - 2 * border)

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
    img = Image.new('RGB', (canvas_width, canvas_height), (127, 127, 127))
    draw = ImageDraw.Draw(img)

    # Custom font style and font size
    myFont = ImageFont.truetype(font_path, fontsize)
    draw.text(img_center, text, font=myFont, fill=0, anchor="mm")


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

    fusion_net = fusion_net_alone(in_channels = 8).to(device)
    checkpoint_bg = torch.load("./light_weights/fusion_net.pth", map_location=torch.device('cpu'))
    fusion_net.load_state_dict(checkpoint_bg['model'])

    inpaint_net = inpainting_net_mask(in_channels = 4).to(device)
    checkpoint_bg = torch.load("./light_weights/inpainting_net.pth", map_location=torch.device('cpu'))
    inpaint_net.load_state_dict(checkpoint_bg['model'])

    mask_net = mask_extraction_net(in_channels = 3, get_feature_map=True).to(device)
    checkpoint = torch.load("./light_weights/mask_net.pth", map_location=torch.device('cpu'))
    mask_net.load_state_dict(checkpoint['model'])


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

    # torch.save({
    #     "model": mask_net.state_dict()
    # }, "./light_weights/mask_net.pth")

    # torch.save({
    #     "model": inpaint_net.state_dict()
    # }, "./light_weights/inpainting_net.pth")

    # torch.save({
    #     "model": fusion_net.state_dict()
    # }, "./light_weights/fusion_net.pth")

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

        o_m, mask_feat = mask_net(i_s)
        o_m = K(o_m)
        o_m_t = o_m
        o_m = 1. - o_m_t
        mask_feat = mask_feat.detach().to(device)
        o_b, _ = inpaint_net(i_s, o_m, mask_feat)
        o_b = K(o_b)

        o_f = fusion_net(torch.cat((o_b, i_s, o_m_t, o_m_t), dim=1))

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


