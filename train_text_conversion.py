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
from model import Generator, Discriminator, Vgg19, text_conversion_net, text_conversion_net_seft_atn
from torchvision import models, transforms, datasets
from loss import build_text_conversion_loss, build_discriminator_loss
from datagen import datagen_text_conversion, example_dataset, To_tensor
from torch.utils.data import Dataset, DataLoader


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

        i_t, i_s, t_sk, t_t, mask_t = item


        i_t = resize(i_t, to_scale, preserve_range=True)
        i_s = resize(i_s, to_scale, preserve_range=True)
        t_sk = np.expand_dims(resize(t_sk, to_scale, preserve_range=True), axis = -1)
        t_t = resize(t_t, to_scale, preserve_range=True)
        mask_t = np.expand_dims(resize(mask_t, to_scale, preserve_range=True), axis = -1)


        i_t = i_t.transpose((2, 0, 1))
        i_s = i_s.transpose((2, 0, 1))
        t_sk = t_sk.transpose((2, 0, 1))
        t_t = t_t.transpose((2, 0, 1))
        mask_t = mask_t.transpose((2, 0, 1))

        i_t_batch.append(i_t)
        i_s_batch.append(i_s)
        t_sk_batch.append(t_sk)
        t_t_batch.append(t_t)
        mask_t_batch.append(mask_t)

    i_t_batch = np.stack(i_t_batch)
    i_s_batch = np.stack(i_s_batch)
    t_sk_batch = np.stack(t_sk_batch)
    t_t_batch = np.stack(t_t_batch)
    mask_t_batch = np.stack(mask_t_batch)

    i_t_batch = torch.from_numpy(i_t_batch.astype(np.float32) / 127.5 - 1.)
    i_s_batch = torch.from_numpy(i_s_batch.astype(np.float32) / 127.5 - 1.)
    t_sk_batch = torch.from_numpy(t_sk_batch.astype(np.float32) / 255.)
    t_t_batch = torch.from_numpy(t_t_batch.astype(np.float32) / 127.5 - 1.)
    mask_t_batch =torch.from_numpy(mask_t_batch.astype(np.float32) / 255.)


    return [i_t_batch, i_s_batch, t_sk_batch, t_t_batch, mask_t_batch]

def clip_grad(model):

    for h in model.parameters():
        # h.data.clamp_(-0.01, 0.01)
        h.data.clamp_(-1, 1)

def main():

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)

    device = torch.device("cuda")
    train_name = get_train_name()
    print_log('Initializing Text Conversion', content_color = PrintColor['yellow'])

    train_data = datagen_text_conversion(cfg)
    train_data = DataLoader(dataset = train_data, batch_size = cfg.batch_size, shuffle = False, collate_fn = custom_collate,  pin_memory = True)

    trfms = To_tensor()
    example_data = example_dataset(transform = trfms)
    example_loader = DataLoader(dataset = example_data, batch_size = 1, shuffle = False)

    print_log('training start.', content_color = PrintColor['yellow'])

    G = text_conversion_net(in_channels=3).cuda()
    # D1 = Discriminator(in_channels=6).cuda()
    # D2 = Discriminator(in_channels=7).cuda()

    G_solver = torch.optim.Adam(G.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    # D1_solver = torch.optim.Adam(D1.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    # D2_solver = torch.optim.Adam(D2.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))

    try:
        if cfg.text_conversion_ckpt_path is not None:
            checkpoint = torch.load(cfg.text_conversion_ckpt_path)
            G.load_state_dict(checkpoint['text_generator'])
            # D1.load_state_dict(checkpoint['text_discriminator1'])
            # D2.load_state_dict(checkpoint['text_discriminator2'])
            G_solver.load_state_dict(checkpoint['text_g_optimizer'])
            # D1_solver.load_state_dict(checkpoint['text_d_optimizer1'])
            # D2_solver.load_state_dict(checkpoint['text_d_optimizer2'])
            print('Resuming after loading...')

    except FileNotFoundError:
        print('checkpoint not found')
        pass


    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0


    trainiter = iter(train_data)
    example_iter = iter(example_loader)

    K = torch.nn.ZeroPad2d((0, 1, 1, 0))

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
                    'text_generator': G.state_dict(),
                    # 'text_discriminator1': D1.state_dict(),
                    # 'text_discriminator2': D2.state_dict(),
                    'text_g_optimizer': G_solver.state_dict(),
                    # 'text_d_optimizer1': D1_solver.state_dict(),
                    # 'text_d_optimizer2': D2_solver.state_dict(),
                },
                cfg.checkpoint_savedir+f'text-train_step-{step+1}.model',
            )
        try:
          i_t, i_s, t_sk, t_t, mask_t = trainiter.next()

        except StopIteration:

          trainiter = iter(train_data)
          i_t, i_s, t_sk, t_t, mask_t = trainiter.next()

        i_t = i_t.cuda()
        i_s = i_s.cuda()
        t_sk = t_sk.cuda()
        t_t = t_t.cuda()
        mask_t = mask_t.cuda()

        labels = [t_sk, t_t]

        # requires_grad(G, True)
        # requires_grad(D1, False)
        # requires_grad(D2, False)

        G_solver.zero_grad()

        o_sk, o_t = G(i_t, i_s)

        o_sk = K(o_sk)
        o_t = K(o_t)

        # i_df_pred = torch.cat((i_s, o_t), dim = 1)
        # o_df_pred = D1(i_df_pred)

        # i_dt_pred = torch.cat((i_t, o_t, o_sk), dim = 1)
        # o_dt_pred = D2(i_dt_pred)

        # out_g = [o_sk, o_t, mask_t]
        # out_d = [o_df_pred, o_dt_pred]

        # g_loss, l_f_gan, l_t_gan, l_t_l1, l_t_sk = build_text_conversion_loss(out_g, out_d, labels)
        # g_loss, l_t_l1, l_t_sk = build_text_conversion_loss(out_g, labels)
        l_t_l1 = torch.mean(torch.abs(o_t - t_t))
        l_sk = torch.mean(torch.abs(o_sk - mask_t))
        g_loss = l_t_l1 + l_sk
        g_loss.backward()
        G_solver.step()

        #g_scheduler.step()

        # requires_grad(G, False)
        # requires_grad(D1, True)
        # requires_grad(D2, True)

        # # o_sk, o_t = G(i_t, i_s)
        # D1_solver.zero_grad()
        # D2_solver.zero_grad()

        # # remove grad completely
        # o_sk = o_sk.detach()
        # o_t = o_t.detach()

        # # o_sk = K(o_sk)
        # # o_t = K(o_t)

        # i_df_true = torch.cat((i_s, t_t), dim = 1)
        # i_df_pred = torch.cat((i_s, o_t), dim = 1)

        # o_df_true = D1(i_df_true)
        # df_real_loss = -torch.mean(torch.log(torch.clamp(o_df_true, cfg.epsilon, 1.0)))
        # df_real_loss.backward()

        # o_df_pred = D1(i_df_pred)
        # df_fake_loss = -torch.mean(torch.log(torch.clamp(1.0 - o_df_pred, cfg.epsilon, 1.0)))
        # df_fake_loss.backward()

        # D1_solver.step()
        # # clip_grad(D1)

        # i_dt_true = torch.cat((i_t, t_t, t_sk), dim = 1)
        # i_dt_pred = torch.cat((i_t, o_t, o_sk), dim = 1)

        # o_dt_true = D2(i_dt_true)
        # dt_real_loss = -torch.mean(torch.log(torch.clamp(o_dt_true, cfg.epsilon, 1.0)))
        # dt_real_loss.backward()

        # o_dt_pred = D2(i_dt_pred)
        # dt_fake_loss = -torch.mean(torch.log(torch.clamp(1.0 - o_dt_pred, cfg.epsilon, 1.0)))
        # dt_fake_loss.backward()

        # D2_solver.step()

        if ((step+1) % cfg.write_log_interval == 0):
            # print('Iter: {}/{} | Sk: {:.4f} | L: {:.4f} | Gf: {:.4f} | Gt: {:.4f} | Ff: {:.4f} | Rf: {:.4f} | Ft: {:.4f} | Rt: {:.4f}'.format(
            print('Iter: {}/{} | L: {:.4f} | Sk: {:.4f}'.format(
                step+1,
                cfg.max_iter,
                l_t_l1.item(),
                l_sk.item()))
                # l_f_gan.item(),
                # l_t_gan.item(),
                # df_fake_loss.item(),
                # df_real_loss.item(),
                # dt_fake_loss.item(),
                # dt_real_loss.item()))

        if ((step+1) % cfg.gen_example_interval == 0) or step == 0:

            savedir = os.path.join(cfg.example_result_dir, train_name, 'iter-' + str(step+1).zfill(len(str(cfg.max_iter))))

            with torch.no_grad():
                try:
                  inp = example_iter.next()
                except StopIteration:
                  example_iter = iter(example_loader)
                  inp = example_iter.next()

                i_t = inp[0].cuda()
                i_s = inp[1].cuda()
                name = str(inp[2][0])

                o_sk, o_t = G(i_t, i_s)
                # o_t = G(i_t, i_s)

                o_sk = o_sk.squeeze(0).to('cpu')
                o_t = o_t.squeeze(0).to('cpu')
                i_t = i_t.squeeze(0).to('cpu')
                i_s = i_s.squeeze(0).to('cpu')

                if not os.path.exists(savedir):
                    os.makedirs(savedir)

                o_sk = F.to_pil_image(o_sk)
                o_t = F.to_pil_image((o_t + 1)/2)
                i_t = F.to_pil_image((i_t + 1)/2)
                i_s = F.to_pil_image((i_s + 1)/2)

                o_sk.save(os.path.join(savedir, name + 'o_sk.png'))
                o_t.save(os.path.join(savedir, name + 'o_t.png'))
                i_t.save(os.path.join(savedir, name + 'i_t.png'))
                i_s.save(os.path.join(savedir, name + 'i_s.png'))

if __name__ == '__main__':
    main()

