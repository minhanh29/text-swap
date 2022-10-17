import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.data import MyDataset
from saicinpainting.training.trainers import load_checkpoint


class Inpainter:
    def __init__(self, model_path):
        self.device = torch.device("cpu")

        train_config_path = os.path.join(model_path, 'config.yml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(model_path, "lama.pth")
        self.model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        self.model.freeze()
        self.model.to(self.device)

    def predict(self, img_list, mask_list):
        dataset = MyDataset(img_list, mask_list, pad_out_to_modulo=8)
        with torch.no_grad():
            for img_i in range(len(dataset)):
                batch = move_to_device(default_collate([dataset[img_i]]), self.device)
                batch['mask'] = (batch['mask'] > 0) * 1
                batch = self.model(batch)
                cur_res = batch["inpainted"][0].permute(1, 2, 0).detach().cpu().numpy()

                unpad_to_size = batch.get('unpad_to_size', None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

                # RGB image
                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                return cur_res
        return None
