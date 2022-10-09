# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:50:30 2022

@author: PC-FONDECYT-1
"""


def s_grupo(grupo):

    t = open("D:/EduardoCavieres/TesisMagister/sep_suj.txt","r")
    
    group = "none"
    s_train = []
    s_test = []
    s_val = []
    
    for lineas in t:
        if "[" in lineas:
            init = lineas.index("[")
            
            elementos = (lineas[init:].strip("\n").split(" "))  
            elementos = list(filter(None,elementos))
            elementos[0] = elementos[0].strip("[")
            elementos = list(filter(None,elementos))
            
            if group == "none":
                #print("group:")
                if "sujetos de entrenamiento" in lineas:
                    group = "train"
                    #print("train")
                    for n in elementos:
                        s_train.append(n)
                if "sujetos de validacion" in lineas:   
                    group = "val"
                    #print("val")
                    for n in elementos:
                        s_val.append(n)
                if "sujetos de test" in lineas:
                    group = "test"
                    #print("test")
                    for n in elementos:
                        s_test.append(int(n))
                                
        elif group != "none":
            elementos = (lineas.strip("\n").split(" "))
            elementos = list(filter(None,elementos))
            elementos[len(elementos)-1] = elementos[len(elementos)-1].strip("]")
            elementos = list(filter(None,elementos))
            if group == "train":
                for n in elementos:
                    s_train.append(int(n))
            if group == "val":
                for n in elementos:
                    s_val.append(int(n))
            if group == "test":
                for n in elementos:
                    s_test.append(int(n))     
                    
        if "]" in lineas:
            #print("fin")
            group = "none"
    
    t.close()  

    if grupo  == "train":
        return(s_train)
    if grupo  == "val":
        return(s_val)
    if grupo  == "test":
        return(s_test)
          
    
    