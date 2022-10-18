import numpy as np
import os
import torch
import torchvision.transforms
from utils import *
import cfg
from tqdm import tqdm
import torchvision.transforms.functional as F
from skimage.transform import resize
from skimage import io
from model import Generator, Discriminator, Vgg19, fusion_net, mask_extraction_net
from torchvision import models, transforms, datasets
from loss import build_generator_loss, build_discriminator_loss, build_gan_loss, build_vgg_loss, build_l1_loss
from datagen import datagen_srnet, example_dataset, To_tensor
from torch.utils.data import Dataset, DataLoader
from inpaint_loss import FusionLoss


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def custom_collate(batch):

    i_t_batch, i_s_batch = [], []
    t_sk_batch, t_t_batch, t_b_batch, t_f_batch = [], [], [], []
    mask_t_batch = []

    w_sum = 0

    for item in batch:

        t_b= item[4]
        h, w = t_b.shape[:2]
        scale_ratio = cfg.data_shape[0] / h
        w_sum += int(w * scale_ratio)

    to_h = cfg.data_shape[0]
    to_w = w_sum // cfg.batch_size
    to_w = int(round(to_w / 8)) * 8
    to_scale = (to_h, to_w)

    for item in batch:

        i_t, i_s, t_sk, t_t, t_b, t_f, mask_t = item


        i_t = resize(i_t, to_scale, preserve_range=True)
        i_s = resize(i_s, to_scale, preserve_range=True)
        t_sk = np.expand_dims(resize(t_sk, to_scale, preserve_range=True), axis = -1)
        t_t = resize(t_t, to_scale, preserve_range=True)
        t_b = resize(t_b, to_scale, preserve_range=True)
        t_f = resize(t_f, to_scale, preserve_range=True)
        mask_t = np.expand_dims(resize(mask_t, to_scale, preserve_range=True), axis = -1)


        i_t = i_t.transpose((2, 0, 1))
        i_s = i_s.transpose((2, 0, 1))
        t_sk = t_sk.transpose((2, 0, 1))
        t_t = t_t.transpose((2, 0, 1))
        t_b = t_b.transpose((2, 0, 1))
        t_f = t_f.transpose((2, 0, 1))
        mask_t = mask_t.transpose((2, 0, 1))

        i_t_batch.append(i_t)
        i_s_batch.append(i_s)
        t_sk_batch.append(t_sk)
        t_t_batch.append(t_t)
        t_b_batch.append(t_b)
        t_f_batch.append(t_f)
        mask_t_batch.append(mask_t)

    i_t_batch = np.stack(i_t_batch)
    i_s_batch = np.stack(i_s_batch)
    t_sk_batch = np.stack(t_sk_batch)
    t_t_batch = np.stack(t_t_batch)
    t_b_batch = np.stack(t_b_batch)
    t_f_batch = np.stack(t_f_batch)
    mask_t_batch = np.stack(mask_t_batch)

    i_t_batch = torch.from_numpy(i_t_batch.astype(np.float32) / 127.5 - 1.)
    i_s_batch = torch.from_numpy(i_s_batch.astype(np.float32) / 127.5 - 1.)
    t_sk_batch = torch.from_numpy(t_sk_batch.astype(np.float32) / 255.)
    t_t_batch = torch.from_numpy(t_t_batch.astype(np.float32) / 127.5 - 1.)
    t_b_batch = torch.from_numpy(t_b_batch.astype(np.float32) / 127.5 - 1.)
    t_f_batch = torch.from_numpy(t_f_batch.astype(np.float32) / 127.5 - 1.)
    mask_t_batch =torch.from_numpy(mask_t_batch.astype(np.float32) / 255.)


    return [i_t_batch, i_s_batch, t_sk_batch, t_t_batch, t_b_batch, t_f_batch, mask_t_batch]

def clip_grad(model):

    for h in model.parameters():
        # h.data.clamp_(-0.01, 0.01)
        h.data.clamp_(-1, 1)

def main():

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)

    train_name = "fuse_" + get_train_name()

    print_log('Initializing SRNET', content_color = PrintColor['yellow'])

    train_data = datagen_srnet(cfg)

    train_data = DataLoader(dataset = train_data, batch_size = cfg.batch_size, shuffle = False, collate_fn = custom_collate,  pin_memory = True)

    trfms = To_tensor()
    example_data = example_dataset(transform = trfms)

    example_loader = DataLoader(dataset = example_data, batch_size = 1, shuffle = False)

    print_log('training start.', content_color = PrintColor['yellow'])

    G = fusion_net(in_channels = 8).cuda()
    # D1 = Discriminator(in_channels = 6).cuda()
    g_loss_func = FusionLoss()

    mask_net = mask_extraction_net(in_channels=3).cuda()
    checkpoint = torch.load(cfg.mask_ckpt_path)
    mask_net.load_state_dict(checkpoint['mask_generator'])
    requires_grad(mask_net, False)

    # vgg_features = Vgg19().cuda()

    G_solver = torch.optim.Adam(G.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    # D1_solver = torch.optim.Adam(D1.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))

    if not os.path.isdir(cfg.fuse_checkpoint_savedir):
        os.makedirs(cfg.fuse_checkpoint_savedir)

    try:
        if cfg.ckpt_path is not None:
            checkpoint = torch.load(cfg.ckpt_path)
            G.load_state_dict(checkpoint['fuse_generator'])
            # D1.load_state_dict(checkpoint['fuse_discriminator'])

            G_solver.load_state_dict(checkpoint['fuse_g_optimizer'])
            # D1_solver.load_state_dict(checkpoint['fuse_d_optimizer'])

            print('Resuming after loading...')

    except FileNotFoundError:

      print('checkpoint not found')
      pass

    # requires_grad(G, False)

    # requires_grad(D1, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0


    trainiter = iter(train_data)
    example_iter = iter(example_loader)

    K = torch.nn.ZeroPad2d((0, 1, 1, 0))

    # for step in tqdm(range(cfg.max_iter)):
    for step in range(cfg.max_iter):

        # D1_solver.zero_grad()

        if ((step+1) % cfg.save_ckpt_interval == 0):

            model_list = os.listdir(cfg.fuse_checkpoint_savedir)
            model_list = [file for file in model_list if ".model" in file]
            num_ckpts = 5
            if len(model_list) > num_ckpts*2:
                model_list.sort(key=lambda x: int(x.split(".")[0].split("-")[-1]))
                num_elim = len(model_list) - num_ckpts
                elim_ckpt = model_list[:num_elim]
                for ckpt in elim_ckpt:
                    ckpt_path = os.path.join(cfg.fuse_checkpoint_savedir, ckpt)
                    os.remove(ckpt_path)
                    print("Deleted", ckpt_path)
            torch.save(
                {
                    'fuse_generator': G.state_dict(),
                    # 'fuse_discriminator': D1.state_dict(),
                    'fuse_g_optimizer': G_solver.state_dict(),
                    # 'fuse_d_optimizer': D1_solver.state_dict(),
                },
                cfg.fuse_checkpoint_savedir+f'fuse-train_step-{step+1}.model',
            )

        try:

          i_t, i_s, t_sk, t_t, t_b, t_f, mask_t = trainiter.next()

        except StopIteration:

          trainiter = iter(train_data)
          i_t, i_s, t_sk, t_t, t_b, t_f, mask_t = trainiter.next()

        i_t = i_t.cuda()
        i_s = i_s.cuda()
        t_sk = t_sk.cuda()
        t_t = t_t.cuda()
        t_b = t_b.cuda()
        t_f = t_f.cuda()
        mask_t = mask_t.cuda()

        # requires_grad(G, True)
        # requires_grad(D1, False)

        G_solver.zero_grad()

        mask_s = mask_net(i_s)
        mask_s = K(mask_s)
        mask_s = mask_s.detach()

        o_f = G(torch.cat((t_b, i_s, mask_s, mask_t), dim=1))
        o_f = K(o_f)

        # i_df_pred = torch.cat((o_f, i_s), dim = 1)
        # o_df_pred = D1(i_df_pred)

        # i_vgg = torch.cat((t_f, o_f), dim = 0)
        # o_vgg = vgg_features(i_vgg)

        # l_f_gan = build_gan_loss(o_df_pred)
        # l_hole = 5 * build_l1_loss(mask_t * t_f, mask_t * o_f)
        # l_f_l1 = build_l1_loss(t_f, o_f)
        # l_f_vgg_per, l_f_vgg_style = build_vgg_loss(o_vgg)
        # l_f_vgg_per = l_f_vgg_per
        # l_f_vgg_style =  l_f_vgg_style
        # # g_loss = 0.07 * l_f_gan + 0.05 * l_f_vgg_per + 500 * l_f_vgg_style + l_f_l1
        # g_loss = 0.05 * l_f_vgg_per + 500 * l_f_vgg_style + l_f_l1 + l_hole
        g_loss, (l_f_l1, l_hole, l_f_vgg_per, l_f_vgg_style) = g_loss_func(t_f, o_f, mask_t)

        g_loss.backward()
        G_solver.step()

        # requires_grad(G, False)
        # requires_grad(D1, True)

        # D1_solver.zero_grad()
        # o_f = o_f.detach()

        # i_df_true = torch.cat((t_f, i_s), dim = 1)
        # i_df_pred = torch.cat((o_f, i_s), dim = 1)

        # o_df_true = D1(i_df_true)
        # o_df_pred = D1(i_df_pred)

        # df_real_loss = -torch.mean(torch.log(torch.clamp(o_df_true, cfg.epsilon, 1.0)))
        # df_real_loss.backward()

        # df_fake_loss = -torch.mean(torch.log(torch.clamp(1.0 - o_df_pred, cfg.epsilon, 1.0)))
        # df_fake_loss.backward()
        # df_loss = df_real_loss + df_fake_loss

        # D1_solver.step()

        if ((step+1) % cfg.write_log_interval == 0):
            print('Iter: {}/{} | L1: {:.4f} | H {:.4f} | Per: {:.4f} | Style: {:.4f}'
                  .format(step+1, cfg.max_iter,
                 l_f_l1.item(),
                 l_hole.item(),
                 l_f_vgg_per.item(),
                 l_f_vgg_style.item()))
            # print('Iter: {}/{} | G: {:.4f} | R: {:.4f} | F: {:.4f} | L1 {:.4f} | Per: {:.4f} | Style: {:.4f}'
            #       .format(step+1, cfg.max_iter,
            #       l_f_gan.item(),
            #      df_real_loss.item(),
            #      df_fake_loss.item(),
            #      l_f_l1.item(),
            #      l_f_vgg_per.item(),
            #      l_f_vgg_style.item()))

        if ((step+1) % cfg.gen_example_interval == 0) or step == 0:

            savedir = os.path.join(cfg.example_result_dir, train_name, 'iter-' + str(step+1).zfill(len(str(cfg.max_iter))))

            i_t = i_t.to('cpu')
            i_s = i_s.to('cpu')
            t_b = t_b.to('cpu')
            o_f = o_f.to('cpu')
            mask_s = mask_s.to('cpu')
            mask_t = mask_t.to('cpu')
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            cnt = 0
            for i_t_img, i_s_img, t_b_img, o_f_img, mask_s_img, mask_t_img in zip(i_t, i_s, t_b, o_f, mask_s, mask_t):
                i_t_img = F.to_pil_image((i_t_img + 1)/2)
                i_s_img = F.to_pil_image((i_s_img + 1)/2)
                t_b_img = F.to_pil_image((t_b_img + 1)/2)
                o_f_img = F.to_pil_image((o_f_img + 1)/2)
                mask_s_img = F.to_pil_image(mask_s_img)
                mask_t_img = F.to_pil_image(mask_t_img)

                i_t_img.save(os.path.join(savedir, f'i_t_{cnt}.png'))
                i_s_img.save(os.path.join(savedir, f'i_s_{cnt}.png'))
                t_b_img.save(os.path.join(savedir, f't_b_{cnt}.png'))
                o_f_img.save(os.path.join(savedir, f'o_f_{cnt}.png'))
                mask_s_img.save(os.path.join(savedir, f'mask_s_{cnt}.png'))
                mask_t_img.save(os.path.join(savedir, f'mask_t_{cnt}.png'))
                cnt += 1

if __name__ == '__main__':
    main()
