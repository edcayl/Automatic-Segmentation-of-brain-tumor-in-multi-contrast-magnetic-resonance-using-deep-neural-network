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

sujeto = 99
plot_all = True

imagen = h5py.File("D:/EduardoCavieres/TesisMagister/data_val/train_brats_4.h5","r")
img = imagen["data"][0,2,:,80,:]


from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage import measure



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



#imagen = "D:\EduardoCavieres\TesisMagister\Resultados\Imgout_16-04-2022_10-31\subject-%s-label.hdr" %(sujeto)

#im, im_itk= read_med_image(imagen, dtype=np.float32) 

#plt.imshow(img)
#plt.show()



#if sujeto <  10:
#    suj_seg = "00%s" %sujeto
#elif sujeto < 100:
#    suj_seg = "0%s" %sujeto 
#else:
#    suj_seg = sujeto
#segmentacion = "D:/EduardoCavieres/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_%s/BraTS20_Training_%s_seg.nii" %(suj_seg,suj_seg)


#se, se_itk= read_med(segmentacion, dtype=np.float32) 
#z,y,x = 100, 100, 100
#plt.imshow(seg)
#plt.show()
imgplot = plt.imshow(img)
plt.show()

if False:#plot_all == True:
    img = im[z,:,:]
    seg = se[z,:,:]
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(seg)
    ax.set_title('Segmentacion')
    plt.colorbar(ticks=[0, 1, 2, 3], orientation='horizontal')
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(img)
    plt.colorbar(ticks=[0, 1, 2, 3], orientation='horizontal')
    ax.set_title('Prediccion')
    plt.show()


if plot_all == True:
    fig,ax = plt.subplots(2, 3, constrained_layout=False)
    ax[0,0].imshow(se[z,:,:])
    ax[0,0].set(title = 'Segmentacion Axial')  
    fig.tight_layout()
    #plt.colorbar(ticks=[0, 1, 2, 3], orientation='horizontal')
    ax[1,0].imshow(im[z,:,:])
    ax[1,0].set(title = 'Prediccion Axial')  
    #plt.colorbar(ticks=[0, 1, 2, 3], orientation='horizontal')
    #ax.set_title('Prediccion')
    fig.tight_layout()
    ax[0,1].imshow(se[:,y,:])
    ax[0,1].set(title = 'Segmentacion Coronal')  
    ax[1,1].imshow(im[:,y,:])
    ax[1,1].set(title = 'Prediccion Coronal')  
    #fig.tight_layout()
    ax[0,2].imshow(se[:,:,x])
    ax[0,2].set(title = 'Segmentacion Sagital')  
    ax[1,2].imshow(im[:,:,x])
    ax[1,2].set(title = 'Prediccion Sagital')  
    fig.subplots_adjust(left=0,
                    #bottom=0.1, 
                    right=1, 
                    #top=0.9, 
                    wspace=0, 
                    hspace=0.25)
    #fig.tight_layout()
    plt.show()
