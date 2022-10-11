import numpy as np
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
from model import Generator, Discriminator, Vgg19, text_conversion_net_light, mask_extraction_net
from torchvision import models, transforms, datasets
from loss import build_text_conversion_loss, build_discriminator_loss, build_dice_loss, build_vgg_loss, build_gan_loss
from datagen import datagen_text_conversion, example_dataset, To_tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from inpaint_loss import TextSwapLoss


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def custom_collate(batch):

    i_t_batch, i_s_batch = [], []
    t_sk_batch, t_t_batch, = [], []
    mask_t_batch = []

    w_sum = 0

    for item in batch:

        t_b = item[1]
        h, w = t_b.shape[:2]
        scale_ratio = cfg.data_shape[0] / h
        w_sum += int(w * scale_ratio)

    to_h = cfg.data_shape[0]
    to_w = w_sum // cfg.batch_size
    to_w = int(round(to_w / 8)) * 8
    to_scale = (to_h, to_w)

    for item in batch:

        i_t, i_s, mask_t = item

        i_t = resize(i_t, to_scale, preserve_range=True)
        i_s = resize(i_s, to_scale, preserve_range=True)
        mask_t = np.expand_dims(resize(mask_t, to_scale, preserve_range=True), axis = -1)
        # i_t = np.expand_dims(resize(i_t, to_scale, preserve_range=True), axis = -1)


        i_t = i_t.transpose((2, 0, 1))
        i_s = i_s.transpose((2, 0, 1))
        mask_t = mask_t.transpose((2, 0, 1))
        # i_t = i_t.transpose((2, 0, 1))

        i_t_batch.append(i_t)
        i_s_batch.append(i_s)
        mask_t_batch.append(mask_t)

    i_t_batch = np.stack(i_t_batch)
    i_s_batch = np.stack(i_s_batch)
    mask_t_batch = np.stack(mask_t_batch)

    i_t_batch = torch.from_numpy(i_t_batch.astype(np.float32) / 127.5 - 1.)
    i_s_batch = torch.from_numpy(i_s_batch.astype(np.float32) / 127.5 - 1.)
    mask_t_batch =torch.from_numpy(mask_t_batch.astype(np.float32) / 255.)
    # i_t_batch =torch.from_numpy(i_t_batch.astype(np.float32) / 255.)


    return [i_t_batch, i_s_batch, mask_t_batch]

def clip_grad(model):

    for h in model.parameters():
        # h.data.clamp_(-0.01, 0.01)
        h.data.clamp_(-1, 1)

def main():

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)

    device = torch.device("cuda")
    train_name = "swap_" + get_train_name()
    print_log('Initializing Text Conversion', content_color = PrintColor['yellow'])

    train_data = datagen_text_conversion(cfg)
    train_data = DataLoader(dataset = train_data, batch_size = cfg.batch_size, shuffle = False, collate_fn = custom_collate,  pin_memory = True)

    trfms = To_tensor()
    example_data = example_dataset(transform = trfms)
    example_loader = DataLoader(dataset = example_data, batch_size = 1, shuffle = False)

    print_log('training start.', content_color = PrintColor['yellow'])

    G = text_conversion_net_light(in_channels=1).cuda()
    g_loss_func = TextSwapLoss()
    D = Discriminator(in_channels=1).cuda()

    mask_net = mask_extraction_net(in_channels=3).cuda()
    checkpoint = torch.load(cfg.mask_ckpt_path)
    mask_net.load_state_dict(checkpoint['mask_generator'])
    requires_grad(mask_net, False)

    G_solver = torch.optim.Adam(G.parameters(), lr=cfg.learning_rate)
    D_solver = torch.optim.Adam(D.parameters(), lr=cfg.learning_rate)

    try:
        if cfg.text_conversion_ckpt_path is not None:
            checkpoint = torch.load(cfg.text_conversion_ckpt_path)
            G.load_state_dict(checkpoint['text_generator'])
            G_solver.load_state_dict(checkpoint['text_g_optimizer'])
            D.load_state_dict(checkpoint['text_disc'])
            D_solver.load_state_dict(checkpoint['text_d_optimizer'])
            print('Resuming after loading...')

    except FileNotFoundError:
        print('checkpoint not found')
        pass

    G = DataParallel(G).to(device)
    D = DataParallel(D).to(device)

    requires_grad(G, True)
    requires_grad(D, True)
    gen_loss_val = 0
    grad_loss_val = 0


    trainiter = iter(train_data)
    example_iter = iter(example_loader)

    K = torch.nn.ZeroPad2d((0, 1, 1, 0))
    if not os.path.isdir(cfg.checkpoint_savedir):
        os.makedirs(cfg.checkpoint_savedir)

    # for step in tqdm(range(cfg.max_iter)):
    for step in range(cfg.max_iter):
        if ((step+1) % cfg.save_ckpt_interval == 0):
            model_list = os.listdir(cfg.checkpoint_savedir)
            model_list = [file for file in model_list if ".model" in file]
            num_ckpts = 5
            if len(model_list) > num_ckpts*2:
                model_list.sort(key=lambda x: int(x.split(".")[0].split("-")[-1]))
                num_elim = len(model_list) - num_ckpts
                elim_ckpt = model_list[:num_elim]
                for ckpt in elim_ckpt:
                    ckpt_path = os.path.join(cfg.checkpoint_savedir, ckpt)
                    os.remove(ckpt_path)
                    print("Deleted", ckpt_path)

            torch.save(
                {
                    'text_generator': G.module.state_dict(),
                    'text_g_optimizer': G_solver.state_dict(),
                    'text_disc': D.module.state_dict(),
                    'text_d_optimizer': D_solver.state_dict(),
                },
                cfg.checkpoint_savedir+f'text-train_step-{step+1}.model',
            )
        try:
          i_t, i_s, mask_t = trainiter.next()

        except StopIteration:

          trainiter = iter(train_data)
          i_t, i_s, mask_t = trainiter.next()

        i_t = i_t.cuda()
        i_s = i_s.cuda()
        mask_t = mask_t.cuda()

        # requires_grad(G, True)
        # requires_grad(D, False)
        G_solver.zero_grad()
        D_solver.zero_grad()

        mask_s = mask_net(i_s)
        mask_s = K(mask_s)
        mask_s = mask_s.detach()
        o_t = G(i_t, mask_s)

        o_t = K(o_t)
        o_pred = D(o_t)

        # l_m_l1 = torch.mean(torch.abs(mask_t - o_t))
        l_gan = torch.mean((1. - o_pred )**2)
        # l_dice = build_dice_loss(mask_t, o_t)
        # g_loss = l_m_l1 + l_dice
        loss1, loss_dict = g_loss_func(o_t, mask_t, mask_s)
        g_loss = loss1 + l_gan
        g_loss.backward()
        G_solver.step()


        # Discriminator
        # requires_grad(G, False)
        # requires_grad(D, True)
        # D.freeze_bn()
        D_solver.zero_grad()
        o_t = o_t.detach()
        o_pred = D(o_t)
        o_true = D(mask_t)

        d_real_loss = torch.mean((1. - o_true)**2)
        d_real_loss.backward()

        d_fake_loss = torch.mean(o_pred**2)
        d_fake_loss.backward()
        d_loss = d_real_loss + d_fake_loss
        # d_loss.backward()
        D_solver.step()
        # clip_grad(D)

        if ((step+1) % cfg.write_log_interval == 0):
            print('Iter: {}/{} | L: {:.4f} | Per: {:.4f} | Style: {:.4f} | G: {:.4f} | R: {:.4f} | F: {:.4f}'.format(
            # print('Iter: {}/{} | L: {:.4f} | G: {:.4f} | D: {:.4f}'.format(
                step+1,
                cfg.max_iter,
                # l_m_l1.item(),
                # l_gan.item(),
                # d_loss.item()))
                loss_dict["l1"].item(),
                # loss_dict["dice"].item(),
                loss_dict["perc"].item(),
                loss_dict["style"].item(),
                l_gan.item(),
                d_real_loss.item(),
                d_fake_loss.item()))

        if ((step+1) % cfg.gen_example_interval == 0) or step == 0:
            savedir = os.path.join(cfg.example_result_dir, train_name, 'iter-' + str(step+1).zfill(len(str(cfg.max_iter))))
            i_t = i_t.to('cpu')
            i_s = i_s.to('cpu')
            o_t = o_t.to('cpu')
            mask_s = mask_s.to('cpu')
            mask_t = mask_t.to('cpu')
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            cnt = 0
            for i_t_img, i_s_img, o_t_img, mask_s_img, mask_t_img in zip(i_t, i_s, o_t, mask_s, mask_t):
                i_t_img = F.to_pil_image((i_t_img + 1)/2)
                i_s_img = F.to_pil_image((i_s_img + 1)/2)
                o_t_img = F.to_pil_image(o_t_img)
                mask_s_img = F.to_pil_image(mask_s_img)
                mask_t_img = F.to_pil_image(mask_t_img)

                i_t_img.save(os.path.join(savedir, f'i_t_{cnt}.png'))
                i_s_img.save(os.path.join(savedir, f'i_s_{cnt}.png'))
                o_t_img.save(os.path.join(savedir, f'o_t_{cnt}.png'))
                mask_s_img.save(os.path.join(savedir, f'mask_s_{cnt}.png'))
                mask_t_img.save(os.path.join(savedir, f'mask_t_{cnt}.png'))
                cnt += 1

if __name__ == '__main__':
    main()

