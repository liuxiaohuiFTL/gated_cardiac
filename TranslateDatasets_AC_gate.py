import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import torchvision
import utils

class TranslateDatasets(torch.utils.data.Dataset):
    def __init__(self, cls_txt_path):
        self.NAC_lists = []
        self.AC_lists = []
        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])

        with open(cls_txt_path) as data_txt:
            for line in data_txt:
                NAC_, AC_ = line.split()
                self.NAC_lists.append(NAC_)
                self.AC_lists.append(AC_)
    
    def data_log_transform(self,X):
      # 变换到-1,1
       X01 = np.log10(X + 1.0)/np.log10(109)
       X11 = 2 * X01 - 1.0
       return X11
     
    def inverse_data_log_transform(self,X01):
        return (10**(X01*np.log10(109)))-1

    def inverse_data_transform(self,X):
        return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

    def __getitem__(self, index):
        self.files = sorted(self.NAC_lists)
        self.files_index = self.files[index % len(self.files)]
        NAC_path = os.path.join(
            'dataset/SUV_16_crop/NH3_rest_16_NAC', self.files_index.split('/')[4])
        AC_path = NAC_path.replace('NAC', 'AC')

        NAC_image = np.load(NAC_path, encoding='bytes', allow_pickle=True)
        #NAC_image = NAC_image[:,:,30]
        
        NAC_image = np.expand_dims(NAC_image, axis=0)
        NAC_image = NAC_image.astype(np.float32)
        NAC_image=  self.data_log_transform(NAC_image)
        NAC_image = torch.from_numpy(NAC_image)
        NAC_image = NAC_image.permute(0,3,1,2)
       

        

        AC_image = np.load(AC_path, encoding='bytes', allow_pickle=True)
        #AC_image = AC_image[:,:,30]
        AC_image = np.expand_dims(AC_image, axis=0)
        AC_image = AC_image.astype(np.float32)
        
        AC_image = self.data_log_transform(AC_image)
        AC_image = torch.from_numpy(AC_image)
        AC_image = AC_image.permute(0,3,1,2)
        
        img_id=self.files_index.split('/')[-1][15:]
        return torch.cat([NAC_image,AC_image], dim=0), img_id

    def __len__(self):
        return max(len(self.AC_lists), len(self.NAC_lists))
