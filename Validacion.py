# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:22:35 2022

@author: PC-FONDECYT-1
"""

from config import *
import time
from loss_func import dice
import SimpleITK as sitk
import glob
from datetime import datetime
from os import listdir, mkdir
from Transformacion_img import Obtener_bordes, Quitar_bordes, resize_ski
from skimage.transform import resize
import h5py

nro_datos = "set"    #unico/set
set_datos = "test"    #val/test/train/dataset
#imagen = "099"    #utilizar siempre string de 3 digitos
val_path =  "D:/EduardoCavieres/TesisMagister/data_%s_64" %set_datos


def convert_label(label_img):
    '''
    function that converts 0, 1, 2, 4 to 0, 1, 2, 3 labels for BG, CSF, GM and WM
    '''
    label_processed = np.where(label_img==1, 1, label_img)
    label_processed = np.where(label_processed==2, 2, label_processed)
    label_processed = np.where(label_processed==4, 3, label_processed)
    return label_processed

def convert_label_back(label_img):
    '''
    function that converts 0, 1, 2, 4 to 0, 1, 2, 3 labels for BG, CSF, GM and WM
    '''
    label_processed = np.where(label_img==1, 1, label_img)
    label_processed = np.where(label_processed==2, 2, label_processed)
    label_processed = np.where(label_processed==3, 4, label_processed)
    return label_processed


def read_med_image(file_path, dtype):
    
    img_stk = sitk.ReadImage(file_path)
    img_np = sitk.GetArrayFromImage(img_stk)
    img_np = img_np.astype(dtype)
    return img_np, img_stk


def predict(net,img):
    xstep = 16
    ystep = 16
    zstep = 16
    root_path = img

    #sub = 'BraTS20_Training_%s_' %imagen
    ft1 = os.path.join(root_path + 't1.nii.gz')
    ft2 = os.path.join(root_path + 't2.nii.gz')
    ft1ce = os.path.join(root_path + 't1ce.nii.gz')
    fflair = os.path.join(root_path + 'flair.nii.gz')
    fgt = os.path.join(root_path + 'seg.nii.gz')   # labels 0 1 2 3 

    imT1, imT1_itk = read_med_image(ft1, dtype=np.float32) 
    imT2, imT2_itk = read_med_image(ft2, dtype=np.float32)
    imT1ce, imT1ce_itk = read_med_image(ft1ce, dtype=np.float32) 
    imFlair, imFlair_itk = read_med_image(fflair, dtype=np.float32) 
    imGT, imGT_itk = read_med_image(fgt, dtype=np.uint8)
    borde_x,borde_y,borde_z = Obtener_bordes(imT1)
    imGT = convert_label(imGT)      
    # print(imGT.shape)
    mask = imT1 > 0
    mask = mask.astype(np.bool)
    
    
    imT1_norm = (imT1 - imT1[mask].mean()) / imT1[mask].std()
    imT2_norm = (imT2 - imT2[mask].mean()) / imT2[mask].std()
    imT1ce_norm = (imT1ce - imT1ce[mask].mean()) / imT1ce[mask].std()
    imFlair_norm = (imFlair - imFlair[mask].mean()) / imFlair[mask].std()
        
    
    imT1_norm = Quitar_bordes(imT1_norm,borde_x,borde_y,borde_z)
    imT2_norm = Quitar_bordes(imT2_norm,borde_x,borde_y,borde_z)
    imT1ce_norm = Quitar_bordes(imT1ce_norm,borde_x,borde_y,borde_z)
    imFlair_norm = Quitar_bordes(imFlair_norm,borde_x,borde_y,borde_z)
    imT1 = Quitar_bordes(imT1,borde_x,borde_y,borde_z)
    imGT = Quitar_bordes(imGT,borde_x,borde_y,borde_z)
    #print(np.unique(imGT))
    imT1_norm = resize_ski(imT1_norm,orden =1)
    imT2_norm = resize_ski(imT2_norm,orden =1)
    imT1ce_norm = resize_ski(imT1ce_norm,orden =1)
    imFlair_norm = resize_ski(imFlair_norm,orden =1)
    imT1 = resize_ski(imT1,orden =1)
    imGT = resize_ski(imGT)
    #imGT = resize(imGT, (64,64,64), order=0, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=False)
    
    img_h5 = h5py.File("D:/EduardoCavieres/TesisMagister/data_test_64/train_brats_%s.h5" %(subject_id),"r")
    t1_h5 =  img_h5["data"][0,0,:,:,:]
    t1_h5 = np.transpose(t1_h5, (0,2,1))
    t2_h5 = img_h5["data"][0,1,:,:,:]
    t2_h5 = np.transpose(t2_h5, (0,2,1))
    t1ce_h5 = img_h5["data"][0,2,:,:,:]
    t1ce_h5 = np.transpose(t1ce_h5, (0,2,1))
    flair_h5 = img_h5["data"][0,3,:,:,:]
    flair_h5 = np.transpose(flair_h5, (0,2,1))
    seg_h5 = img_h5["label"][0,0,:,:,:]
    seg_h5 = np.transpose(seg_h5, (0,2,1))
    
    #print(np.unique(imGT))
    input1 = t1_h5[:, :, :, None]
    input2 = t2_h5[:, :, :, None]
    input3 = t1ce_h5[:, :, :, None]
    input4 = flair_h5[:, :, :, None]

    inputs = np.concatenate((input1, input2, input3, input4), axis=3)
    inputs = inputs[None, :, :, :, :]
    image = inputs.transpose(0, 4, 1, 3, 2)
    image = torch.from_numpy(image).float().to(device)   

    _, _, C, H, W = image.shape
    deep_slices   = np.arange(0, C - crop_size[0] + xstep, xstep)
    height_slices = np.arange(0, H - crop_size[1] + ystep, ystep)
    width_slices  = np.arange(0, W - crop_size[2] + zstep, zstep)

    whole_pred = np.zeros((1,)+(num_classes,) + image.shape[2:])
    count_used = np.zeros((image.shape[2], image.shape[3], image.shape[4])) + 1e-5

    with torch.no_grad():
        
        outputs = net(image)
    
    
    
    whole_pred = outputs.data.cpu().numpy()    
  
    whole_pred = whole_pred[0, :, :, :, :]
    whole_pred = np.argmax(whole_pred, axis=0)
    
    whole_pred = whole_pred.transpose(0,2,1)
    whole_pred = (imT1 != 0) * whole_pred
    d0,px0_i,px0_pred,px0_seg = dice(whole_pred, seg_h5, 0)
    d1,px1_i,px1_pred,px1_seg = dice(whole_pred, seg_h5, 1)
    d2,px2_i,px2_pred,px2_seg = dice(whole_pred, seg_h5, 2)
    d3,px3_i,px3_pred,px3_seg = dice(whole_pred, seg_h5, 3)
    da = np.mean([d1, d2, d3])
    px1 = np.array([px1_i,px1_pred,px1_seg])
    px2 = np.array([px2_i,px2_pred,px2_seg])
    px3 = np.array([px3_i,px3_pred,px3_seg])
    px = np.array([px1,px2,px3])
    return [round(d0*100,2), round(d1*100,2), round(d2*100,2), round(d3*100,2), round(da*100,2), px]

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DenseResNet(num_init_features=32, growth_rate=16, block_config=(4, 4, 4, 4),num_classes=4).to(device)
    
    hora = datetime.now()
    hora = hora.strftime(("%d-%m-%Y_%H-%M-%S"))
    directorio = 'D:/EduardoCavieres/TesisMagister/Resultados/output_validation_%s.txt' %hora
    mkdir(directorio)
    f1 = open('%s/output_validation_%s.txt' %(directorio,hora), 'a+')
    f_dice = open('%s/Analisis_Dice_%s.txt' %(directorio,hora), 'a+')
    
    
    
    
    model = 300
    f1.write( 'Model: %s ;Group: %s \n'    %(model, set_datos))
    f_dice.write( 'Model: %s ;Group: %s \n' %(model, set_datos))
    
    model = str(model).zfill(5)

    
    saved_state_dict = torch.load( './checkpoints/model_epoch_'+ model +'.pth' )
    net.load_state_dict(saved_state_dict)
    net.eval()
    
    
    
    onlyfiles = listdir(val_path)
    test_subj = []
    for n in onlyfiles:
        test_subj.append(int(n[12:-3]))    #obtener grupo sujetos test
    test_subj.sort()
    
    
    for subject_id in (test_subj): 
        subject_id = int(subject_id)
        
        ###
        if True:
            if (subject_id) > 99: 
              imagen = "D:\EduardoCavieres\MICCAI_BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_%d\BraTS20_Training_%d_" % (subject_id,subject_id) 
            elif (subject_id) > 9: 
              imagen = "D:\EduardoCavieres\MICCAI_BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_0%d\BraTS20_Training_0%d_" % (subject_id,subject_id)
            else:
              imagen = "D:\EduardoCavieres\MICCAI_BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_00%d\BraTS20_Training_00%d_" % (subject_id,subject_id)    
        #######   
        
        if False:
            imagen = "D:/EduardoCavieres/TesisMagister/data_val/train_brats_%s.h5" %(subject_id)
           
        
        
        
        
      
        d = predict(net,imagen)
    
        print( 'Model: %s Subject: %s Background: %2.2f NCR/NET: %2.2f PE: %2.2f ET: %2.2f Mean: %2.2f\n' % (model,subject_id, d[0], d[1], d[2], d[3], d[4]) )
        f1.write( 'Model: %s ;Subject: %s ;Background: %2.2f ;Non-enhancing_tumor_core: %2.2f ;Peritumoral_Edema: %2.2f ;Enhancing_Tumor: %2.2f ;Mean: %2.2f\n' % (model, subject_id, d[0], d[1], d[2], d[3], d[4]) )
        f_dice.write( 'Subject:%s   ;Net_inter:%s,Net_pred:%s,Net_seg:%s   ;PE_inter:%s,PE_pred:%s,PE_seg:%s   ;ET_inter:%s,ET_pred:%s,ET_seg:%s \n' % (subject_id, d[5][0][0], d[5][0][1], d[5][0][2], d[5][1][0],d[5][1][1], d[5][1][2], d[5][2][0], d[5][2][1], d[5][2][2])  )
        f1.flush()
        f_dice.flush()
        f_dice.flush()
    
    
    f_log = open("net_log.txt","a+") 
    f_log.write("%s \t ,validate \t ,Model:%s;Subject:%s;Background:%2.2f;NCR/NET:%2.2f;PE:%2.2f;ET:%2.2f;Mean:%2.2f\n" %(hora,model, subject_id, d[0], d[1], d[2], d[3], d[4]))
    f_log.close()
    f1.close()
    
    
    