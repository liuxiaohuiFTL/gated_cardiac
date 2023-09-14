import torch
import torch.nn as nn
import utils
import numpy as np
import utils
import torchvision
import os
import math

def data_transform(X):
    return 2 * X - 1.0
  
def inverse_data_log_transform(X):
     return (10**(X*np.log10(109)))-1
      

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)
    #return (X + 1.0) / 2.0

    # 将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量。

class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, r=None):
        image_folder = os.path.join(self.args.image_folder,str(self.config.sampling.fold))
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                y=str(y)
                y=y.replace("('_",'')
                y=y.replace(".npy',)",'')
                print(f"starting processing from image {y}")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 6 else x
                x_cond = x[:, :1, :, :,:].to(self.diffusion.device)
                # !!!这里条件也要变换到-1,1 这里在调试log的时候要注释，因为在dataloader构建的时候就已经变换到-1,1了
                # x_cond=data_transform(x_cond)
                # 进行采样
                x_output = self.diffusive_restoration(x_cond, r=r)
                # 变换到0-1
                x_output = inverse_data_transform(x_output)
                # x_cond = inverse_data_transform(x_cond)
                # tensor -> numpy 
                # x_cond_np = np.squeeze(x_cond.detach().cpu().numpy()[0])
                x_output_np = np.squeeze(x_output.detach().cpu().numpy()[0])
                # # 0-1 ->nomal
                # x_cond_np = inverse_data_log_transform(x_cond_np)
                x_output_np = inverse_data_log_transform(x_output_np)

                
                utils.logging.save_image(x_output_np, os.path.join(image_folder, f"NH3_rest_16_AC_{y}.tif"))
                # utils.logging.save_image(x_cond_np, os.path.join(image_folder, f"NH3_rest_16_NAC_{y}.tif"))


    def diffusive_restoration(self, x_cond, r=None):
        x = torch.randn(x_cond.size(), device=self.diffusion.device)
        if r != None:
            p_size = self.config.data.image_size
            h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
            corners = [(i, j) for i in h_list for j in w_list]
            x_output = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size)
        else:           
            x_output = self.diffusion.sample_image(x_cond, x, patch_locs=None, patch_size=None)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list
