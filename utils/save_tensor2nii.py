import numpy as np
import os
import SimpleITK as sitk
import math


def writeNii(image_array, image, out_path):
    """
    此函数用于复制或写入nii文件
    image_array:图像的npy矩阵
    image:用于获取仿射矩阵的nii源文件,为sitk对象
    """
    image_save = sitk.GetImageFromArray(image_array)
    image_save.SetDirection(image.GetDirection())
    image_save.SetOrigin((image.GetOrigin()))  # for change image origin
    image_save.SetSpacing(image.GetSpacing())
    sitk.WriteImage(image_save, out_path)


def tensor2nii(out_np, cond_np,save_name):
      cur_file = save_name.split()
      AC_image_path = 
      NAC_image_path = AC_image_path.replace('AC', 'NAC')
      
      
      AC_image_nii = sitk.ReadImage(AC_image_path)
      NAC_image_nii = sitk.ReadImage(NAC_image_path)
      


      writeNii(out_np, NAC_image_nii,save_name)
      writeNii(cond_np, AC_image_nii,save_name.replace('fakeB','realB'))






