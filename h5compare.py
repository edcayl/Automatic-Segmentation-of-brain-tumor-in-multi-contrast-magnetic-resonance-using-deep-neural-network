# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:50:19 2022

@author: PC-FONDECYT-1
"""

from cv2 import TermCriteria_COUNT
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import h5py
import nibabel as nib
import os.path
import cv2




sujeto = 38
plot_all = True
savefile = True

imagen = h5py.File("D:/EduardoCavieres/TesisMagister/Hdense_128/data_test_128/train_brats_%s.h5" %sujeto,"r")
prediccion = "D:/EduardoCavieres/TesisMagister/Resultados/Imgout_23-05-2022_11-55/subject-%s-label.hdr" %(sujeto)
         
     

     
#imagen = h5py.File("D:/EduardoCavieres/IsegDataset/h5out/subject-4-_iSeg-2019-Training.h5")
#img = imagen["data"][0,:,:,:]


from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage import measure



def convert_label_back(label_img):
    '''
    function that converts 0, 1, 2, 4 to 0, 1, 2, 3 labels for BG, CSF, GM and WM
    '''
    label_processed = np.where(label_img==1, 1, label_img)
    label_processed = np.where(label_processed==2, 2, label_processed)
    label_processed = np.where(label_processed==3, 4, label_processed)
    return label_processed

def read_med_image(file_path, dtype):
    #----Transformar imagen------
    img_stk = sitk.ReadImage(file_path)
    #size = (256,192,144)
    #img_stk = sitk.ReadImage(file_path)
    img_np = sitk.GetArrayFromImage(img_stk)
    #img_np = img_np[:,24:216,48:192]
    #background = np.zeros(size)
    #background[50:205,:,:] = img_np
    #img_np = background
    #img_np = img_np.astype(dtype)
    return img_np, img_stk

def read_med(file_path, dtype):
    #----Transformar imagen------
    img_stk = sitk.ReadImage(file_path)
    size = (256,192,144)
    #img_stk = sitk.ReadImage(file_path)
    img_np = sitk.GetArrayFromImage(img_stk)
    img_np = img_np[:,24:216,48:192]
    background = np.zeros(size)
    background[50:205,:,:] = img_np
    img_np = background
    img_np = img_np.astype(dtype)
    return img_np, img_stk    

def getimg(file_path, dtype):
    #----Transformar imagen------
    img_stk = sitk.ReadImage(file_path)
    img_np = sitk.GetArrayFromImage(img_stk)
    img_np = sitk.GetArrayFromImage(img_stk)

    return img_np





#inputs_tmp_T1, inputs_tmp_T2, inputs_tmp_T1ce, inputs_tmp_Flair

img_t1 = imagen["data"][0,0,:,:,:]

#img_t1 = np.flip(img_t1, (0))
img_t1 =  img_t1[:,:,::-1]
img_t1 =  img_t1[:,::-1,:]
#img_t1 = np.flip(img_t1, (2))
#img_t1 = np.flip(img_t1, (1))
img_t1 = np.transpose(img_t1, (1,2,0))


#plt.imshow(img_t1[30,:,:])
plt.imshow(img_t1[:,30,:])
#plt.imshow(img_t1[:,:,30])

#img_t1 = np.moveaxis(img_t1, 0, 2)
#sitk.WriteImage(img_t1, 'image_t1.hdr')
#cv2.imwrite('image_t1.hdr', img_t1)


final_t1 = nib.Nifti1Image(img_t1,np.eye(4))

ornt = np.array([[0, 1],
                [1, -1],
                [2, -1]])

#final_t1 = final_t1.as_reoriented(ornt)




img_t2 = imagen["data"][0,1,:,:,:]
img_t2 =  img_t2[:,:,::-1]
img_t2 =  img_t2[:,::-1,:]
img_t2 = np.transpose(img_t2, (1,2,0))
#img_t2 = getimg(img_t2, dtype=np.float32)
#img_t2 = np.moveaxis(img_t2, -3, 0)
final_t2 = nib.Nifti1Image(img_t2,np.eye(4))
img_t1ce = imagen["data"][0,2,:,:,:]
img_t1ce =  img_t1ce[:,:,::-1]
img_t1ce =  img_t1ce[:,::-1,:]
img_t1ce = np.transpose(img_t1ce, (1,2,0))

#img_t1ce = getimg(img_t1ce, dtype=np.float32)
final_t1ce = nib.Nifti1Image(img_t1ce,np.eye(4))
img_flair = (imagen["data"][0,3,:,:,:])
img_flair =  img_flair[:,:,::-1]
img_flair =  img_flair[:,::-1,:]
img_flair = np.transpose(img_flair, (1,2,0))
#img_flair = getimg(img_flair, dtype=np.float32)
final_flair = nib.Nifti1Image(img_flair,np.eye(4))

img_label = imagen["label"][0,0,:,:,:]


img_label =  img_label[:,:,::-1]
img_label =  img_label[:,::-1,:]
img_label = np.transpose(img_label, (1,2,0))
img_label = convert_label_back(img_label)

#img_label = np.flip(img_label, (0))


#img_label = getimg(img_label, dtype=np.float32)
final_label = nib.Nifti1Image(img_label,np.eye(4))
predict,_ = read_med_image(prediccion, dtype=np.uint8) 
predict =  predict[:,:,::-1]
predict =  predict[:,::-1,:]
predict = np.transpose(predict, (2,1,0))
predict = nib.Nifti1Image(predict,np.eye(4))



if savefile == True:
    nib.save(final_t1, os.path.join("D:/EduardoCavieres/TesisMagister/TransformacionH5_Vtk", 'img_%s_128p.t1.nii.gz' %sujeto))
    nib.save(final_t2, os.path.join("D:/EduardoCavieres/TesisMagister/TransformacionH5_Vtk", 'img_%s_128p.t2.nii.gz' %sujeto))
    nib.save(final_t1ce, os.path.join("D:/EduardoCavieres/TesisMagister/TransformacionH5_Vtk", 'img_%s_128p.t1ce.nii.gz' %sujeto))
    nib.save(final_flair, os.path.join("D:/EduardoCavieres/TesisMagister/TransformacionH5_Vtk", 'img_%s_128p.flair.nii.gz' %sujeto))
    nib.save(final_label, os.path.join("D:/EduardoCavieres/TesisMagister/TransformacionH5_Vtk", 'img_%s_128p.seg.nii.gz' %sujeto))
  #  nib.save(predict, os.path.join("D:/EduardoCavieres/TesisMagister/TransformacionH5_Vtk", 'predict_%s_128p.seg.nii.gz' %sujeto))