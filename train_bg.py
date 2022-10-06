import numpy as np
import random
import os
import torch
import torchvision.transforms
from torch.nn import functional as F1
from utils import *
import cfg
from tqdm import tqdm
import torchvision.transforms.functional as F
from skimage.transform import resize
from skimage import io
from model import Generator, Discriminator, Vgg19, inpainting_net_mask, mask_extraction_net
from torchvision import models, transforms, datasets
from loss import build_bg_loss, build_discriminator_loss
from datagen import datagen_bg, example_dataset, To_tensor
from torch.utils.data import Dataset, DataLoader
from inpaint_loss import InpaintingLoss


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def custom_collate(batch):

    i_s_batch, t_b_batch = [], []
    mask_t_batch = []

    w_sum = 0

    for item in batch:

        t_b = item[1]
        h, w = t_b.shape[:2]
        scale_ratio = cfg.data_shape[0] / h
        w_sum += int(w * scale_ratio)

    to_h = cfg.data_shape[0]
    to_w = w_sum // cfg.bg_batch_size
    to_w = int(round(to_w / 8)) * 8
    to_scale = (to_h, to_w)

    for item in batch:
        i_s, t_b, mask_t = item

        mask_t = np.expand_dims(resize(mask_t, to_scale, preserve_range=True), axis = -1)

        i_s = resize(i_s, to_scale, preserve_range=True)
        t_b = resize(t_b, to_scale, preserve_range=True)

        i_s = i_s.transpose((2, 0, 1))
        t_b = t_b.transpose((2, 0, 1))
        mask_t = mask_t.transpose((2, 0, 1))

        i_s_batch.append(i_s)
        t_b_batch.append(t_b)
        mask_t_batch.append(mask_t)

    i_s_batch = np.stack(i_s_batch)
    t_b_batch = np.stack(t_b_batch)
    mask_t_batch = np.stack(mask_t_batch)

    i_s_batch = torch.from_numpy(i_s_batch.astype(np.float32) / 127.5 - 1.)
    t_b_batch = torch.from_numpy(t_b_batch.astype(np.float32) / 127.5 - 1.)
    mask_t_batch =torch.from_numpy(mask_t_batch.astype(np.float32) / 255.)


    return [i_s_batch, t_b_batch, mask_t_batch]

def clip_grad(model):

    for h in model.parameters():
        # h.data.clamp_(-0.01, 0.01)
        h.data.clamp_(-1, 1)


def main():

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)

    device = torch.device("cuda")
    train_name = "bg_" + get_train_name()
    print_log('Initializing Text Conversion', content_color = PrintColor['yellow'])

    train_data = datagen_bg(cfg)
    train_data = DataLoader(dataset = train_data, batch_size = cfg.bg_batch_size, shuffle = False, collate_fn = custom_collate,  pin_memory = True)

    trfms = To_tensor()
    example_data = example_dataset(transform = trfms)
    example_loader = DataLoader(dataset = example_data, batch_size = 1, shuffle = False)

    print_log('training start.', content_color = PrintColor['yellow'])

    G = inpainting_net_mask(in_channels=4).cuda()
    G_solver = torch.optim.Adam(G.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    g_loss_func = InpaintingLoss()

    mask_net = mask_extraction_net(in_channels=3, get_feature_map=True).cuda()
    checkpoint = torch.load(cfg.mask_ckpt_path)
    mask_net.load_state_dict(checkpoint['mask_generator'])
    requires_grad(mask_net, False)

    try:
        if cfg.bg_ckpt_path is not None:
            checkpoint = torch.load(cfg.bg_ckpt_path)
            G.load_state_dict(checkpoint['bg_generator'])
            G_solver.load_state_dict(checkpoint['bg_g_optimizer'])
            print('Resuming after loading...')

    except FileNotFoundError:
        print('checkpoint not found')
        pass

    gen_loss_val = 0
    grad_loss_val = 0


    trainiter = iter(train_data)
    example_iter = iter(example_loader)

    K = torch.nn.ZeroPad2d((0, 1, 1, 0))

    requires_grad(G, True)

    # for step in tqdm(range(cfg.max_iter)):
    for step in range(cfg.max_iter):
        if ((step+1) % cfg.save_ckpt_interval == 0):
            model_list = os.listdir(cfg.bg_checkpoint_savedir)
            if not os.path.isdir(cfg.bg_checkpoint_savedir):
                os.makedirs(cfg.bg_checkpoint_savedir)
            model_list = [file for file in model_list if ".model" in file]
            num_ckpts = 5
            if len(model_list) > num_ckpts*2:
                model_list.sort(key=lambda x: int(x.split(".")[0].split("-")[-1]))
                num_elim = len(model_list) - num_ckpts
                elim_ckpt = model_list[:num_elim]
                for ckpt in elim_ckpt:
                    ckpt_path = os.path.join(cfg.bg_checkpoint_savedir, ckpt)
                    os.remove(ckpt_path)
                    print("Deleted", ckpt_path)

            torch.save(
                {
                    'bg_generator': G.state_dict(),
                    'bg_g_optimizer': G_solver.state_dict(),
                },
                cfg.bg_checkpoint_savedir+f'bg-train_step-{step+1}.model',
            )
        try:
          i_s, t_b, mask_t = trainiter.next()

        except StopIteration:

          trainiter = iter(train_data)
          i_s, t_b, mask_t = trainiter.next()

        i_s = i_s.cuda()
        t_b = t_b.cuda()
        mask_t = mask_t.cuda()

        G_solver.zero_grad()

        o_mask, mask_feat = mask_net(i_s)
        o_mask = K(o_mask)
        my_mask = None
        my_mask = o_mask.detach()
        # if random.random() < 0.6:
        #     my_mask = o_mask.detach()
        # else:
        #     my_mask = mask_t
        my_mask = 1. - my_mask
        mask_feat = mask_feat.detach().cuda()

        o_b, _ = G(i_s, my_mask, mask_feat)

        o_b = K(o_b)

        g_loss, loss_dict = g_loss_func(i_s, my_mask, o_b, t_b)
        g_loss.backward()
        G_solver.step()

        if ((step+1) % cfg.write_log_interval == 0):
            print('Iter: {}/{} | L: {:.4f} | V: {:.4f} | H: {:.4f} | Per: {:.4f} | Style: {:.4f}'.format(
                step+1,
                cfg.max_iter,
                loss_dict["comp_l1"].item(),
                loss_dict["valid"].item(),
                loss_dict["hole"].item(),
                loss_dict["perc"].item(),
                loss_dict["style"].item(),
                loss_dict["tv"].item()))

        if ((step+1) % cfg.gen_example_interval == 0) or step == 0:
            savedir = os.path.join(cfg.example_result_dir, train_name, 'iter-' + str(step+1).zfill(len(str(cfg.max_iter))))
            i_s = i_s.to('cpu')
            o_b = o_b.to('cpu')
            my_mask = my_mask.to('cpu')
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            cnt = 0
            for i_s_img, o_b_img, mask_img in zip(i_s, o_b, my_mask):
                mask_img = F.to_pil_image(mask_img)
                o_b_img = F.to_pil_image((o_b_img + 1)/2)
                i_s_img = F.to_pil_image((i_s_img + 1)/2)

                o_b_img.save(os.path.join(savedir, f'o_b_{cnt}.png'))
                i_s_img.save(os.path.join(savedir, f'i_s_{cnt}.png'))
                mask_img.save(os.path.join(savedir, f'mask_{cnt}.png'))
                cnt += 1

if __name__ == '__main__':
    main()

