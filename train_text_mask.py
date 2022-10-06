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
from model import Generator, Discriminator, Vgg19, text_conversion_net, mask_extraction_net
from torchvision import models, transforms, datasets
from loss import build_text_conversion_loss, build_discriminator_loss, build_dice_loss
from datagen import datagen_mask, example_dataset, To_tensor
from torch.utils.data import Dataset, DataLoader


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def custom_collate(batch):
    t_f_batch, mask_t_batch = [], []

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
        t_f, mask_t = item

        t_f = resize(t_f, to_scale, preserve_range=True)
        mask_t = np.expand_dims(resize(mask_t, to_scale, preserve_range=True), axis = -1)

        t_f = t_f.transpose((2, 0, 1))
        mask_t = mask_t.transpose((2, 0, 1))

        t_f_batch.append(t_f)
        mask_t_batch.append(mask_t)

    t_f_batch = np.stack(t_f_batch)
    mask_t_batch = np.stack(mask_t_batch)

    t_f_batch = torch.from_numpy(t_f_batch.astype(np.float32) / 127.5 - 1.)
    mask_t_batch =torch.from_numpy(mask_t_batch.astype(np.float32) / 255.)

    return [t_f_batch, mask_t_batch]


def clip_grad(model):

    for h in model.parameters():
        # h.data.clamp_(-0.01, 0.01)
        h.data.clamp_(-1, 1)

def main():

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)

    device = torch.device("cuda")
    train_name = "mask_" + get_train_name()
    print_log('Initializing Text Conversion', content_color = PrintColor['yellow'])

    train_data = datagen_mask(cfg)
    train_data = DataLoader(dataset = train_data, batch_size = cfg.batch_size, shuffle = False, collate_fn = custom_collate,  pin_memory = True)

    trfms = To_tensor()
    example_data = example_dataset(transform = trfms)
    example_loader = DataLoader(dataset = example_data, batch_size = 1, shuffle = False)

    print_log('training start.', content_color = PrintColor['yellow'])

    G = mask_extraction_net(in_channels=3).cuda()
    G_solver = torch.optim.Adam(G.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))

    try:
        if cfg.mask_ckpt_path is not None:
            checkpoint = torch.load(cfg.mask_ckpt_path)
            G.load_state_dict(checkpoint['mask_generator'])
            G_solver.load_state_dict(checkpoint['mask_g_optimizer'])
            print('Resuming after loading...')

    except FileNotFoundError:
        print('checkpoint not found')
        pass


    gen_loss_val = 0
    grad_loss_val = 0


    trainiter = iter(train_data)
    example_iter = iter(example_loader)

    K = torch.nn.ZeroPad2d((0, 1, 1, 0))

    # for step in tqdm(range(cfg.max_iter)):
    for step in range(cfg.max_iter):
        if ((step+1) % cfg.save_ckpt_interval == 0):
            if not os.path.isdir(cfg.checkpoint_savedir):
                os.makedirs(cfg.checkpoint_savedir)
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
                    'mask_generator': G.state_dict(),
                    'mask_g_optimizer': G_solver.state_dict(),
                },
                cfg.checkpoint_savedir+f'text-train_step-{step+1}.model',
            )
        try:
          t_f, mask_t = trainiter.next()

        except StopIteration:

          trainiter = iter(train_data)
          t_f, mask_t = trainiter.next()

        t_f = t_f.cuda()
        mask_t = mask_t.cuda()

        G_solver.zero_grad()

        o_m = G(t_f)

        o_m = K(o_m)

        l_m_l1 = torch.mean(torch.abs(mask_t - o_m))
        l_dice = build_dice_loss(mask_t, o_m)
        g_loss = l_m_l1 + l_dice
        g_loss.backward()
        G_solver.step()


        if ((step+1) % cfg.write_log_interval == 0):
            print('Iter: {}/{} | L: {:.4f} | Dice: {:.4f}'.format(
                step+1,
                cfg.max_iter,
                l_m_l1.item(),
                l_dice.item()))

        if ((step+1) % cfg.gen_example_interval == 0) or step == 0:
            savedir = os.path.join(cfg.example_result_dir, train_name, 'iter-' + str(step+1).zfill(len(str(cfg.max_iter))))
            t_f = t_f.to('cpu')
            o_m = o_m.to('cpu')
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            cnt = 0
            for t_f_img, o_m_img in zip(t_f, o_m):
                o_m_img = F.to_pil_image(o_m_img)
                t_f_img = F.to_pil_image((t_f_img + 1)/2)

                o_m_img.save(os.path.join(savedir, f'o_m_{cnt}.png'))
                t_f_img.save(os.path.join(savedir, f't_f_{cnt}.png'))
                cnt += 1

if __name__ == '__main__':
    main()

