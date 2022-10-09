# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:47:22 2022

@author: PC-FONDECYT-1
"""
import numpy as np
#from skimage.transform import resize

def Obtener_bordes(IMref):
    
    #out_y = (np.sum(IMref,axis=0)).astype(int)
    out_y = (np.sum(IMref,axis=1) < np.max(IMref)*IMref.shape[1]).astype(int)
     
    #out_x = (np.sum(IMref,axis=1)).astype(int)
    out_x = (np.sum(IMref,axis=0) < np.max(IMref)*IMref.shape[0]).astype(int)
            
    #out_z = (np.sum(IMref,axis=2)).astype(int)
    #out_z = (np.sum(out_z,axis=1) == 0).astype(int)   
    
    return(out_x,out_y)

def Quitar_bordes(IM,axx,axy,axz):
    img_sinbordes = np.delete(IM, np.argwhere(axy==1),1)
    img_sinbordes = np.delete(img_sinbordes, np.argwhere(axx==1),2)
    img_sinbordes = np.delete(img_sinbordes, np.argwhere(axz==1),0) 
    
    return(img_sinbordes)


def resize_ski(IM, orden=0,tamano = (64,64,64),al=None,als=None):
          img_transformada = resize(IM, tamano, order=0, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=False)
          
          return(img_transformada)