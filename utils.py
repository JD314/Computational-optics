#Version 21-04-2024
#-Añadido fusionm y tajadam basados en coordenadas matriciales.
#Version 11-12-2024
#-Cambiado propft a usar solo torch.

import torch
import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
import cv2

class parameters:
    object_size = (800, 800)
    resolution = (1200, 2048)
    sep = 80


def tajada(M1,M2,x,y):
    """
    Esta funcion permite recortar una seccion centrada en x, y del array M1 de
    tamaño igual al array M2. Las coordenadas x,y son dadas en un sistema cartesiano centrado.

    Input:
    M1 : numpy array, arreglo del que se recorta una sección.
    M2 : numpy array, arreglo que determina el tamaño del recorte.
    x : integer, coordenada en x del centro del recorte en un sistema cartesiano centrado para M1.
    y : Integer, coordenada en y del centro del recorte en un sistema cartesiano centrado para M1..

    Output: 
        numpy array, con forma igual a M2.


    Raises
    ValueError: Si el tamaño del area a recortar o las coordenadas exceden el tamaño de la matriz original se disparara este error.
    """
    m=M1.shape
    mm=M2.shape
    if mm[-1]>m[-1] or mm[-2]>m[-2]:
        raise ValueError('La matriz recortada debe ser mas pequeña que la original')
    gym=((m[-2]-mm[-2])//2)

    gyp=((m[-2]+mm[-2])//2)

    gxm=((m[-1]-mm[-1])//2)

    gxp=((m[-1]+mm[-1])//2)

    x=np.int32(x)

    y=np.int32(y)

    MC=M1[...,gym+y:gyp+y,gxm+x:gxp+x]
    return MC

def fusion(M1,M2,x,y):
    """
    Esta funcion permite insertar M2 a una seccion centrada en (x, y) del array M1.
    Las coordenadas x,y son dadas en un sistema cartesiano centrado.

    Parameters
    ----------
    M1 : Array Numpy
        Arreglo al que se inserta una sección.
    M2 : Array Numpy
        Arreglo a insertar.
    x : Integer
        Coordenada en x del centro del recorte en un sistema cartesiano centrado para M1.
    y : Integer
        Coordenada en y del centro del recorte en un sistema cartesiano centrado para M1..

    Raises
    ------
    ValueError
        Si el tamaño del area a insertar o las coordenadas exceden el tamaño de la matriz original se disparara este error.

    Returns
    -------
    MC
        Arreglo de numpy con forma igual a M1.

    """
    m=M1.shape
    mm=M2.shape
    if mm[-2]>m[-2] or mm[-1]>m[-1]:
        raise ValueError('La matriz a instertar debe ser mas pequeña que la original')
    gym=((m[-2]-mm[-2])//2)

    gyp=((m[-2]+mm[-2])//2)

    gxm=((m[-1]-mm[-1])//2)

    gxp=((m[-1]+mm[-1])//2)

    x=np.int32(x)
    y=np.int32(y)
    # print(gyp-(gym))

    M1[...,gym+y:gyp+y,gxm+x:gxp+x]=M2

    MC=M1
    return MC

def itfourier(x):
    x=torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x)))
    return x

def tfourier(x):
    x=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))
    return x

def corr2(A,B):
    cc=(((A-torch.mean(A))*((B-torch.mean(B)))).sum())/(torch.sqrt((A-torch.mean(A)).pow(2).sum()*(B-torch.mean(B)).pow(2).sum()))
    return cc

def campo_de_prueba():
    """"Campo óptico de prueba"""
    objeto = np.array([
       [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
       [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
       [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
       [1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]])
    
    img = cv2.resize(objeto, parameters.object_size, interpolation=cv2.INTER_NEAREST) 

    # -- Create base optic field --
    campo = np.zeros(parameters.resolution)
    
    # Shape the object in the optic field
    campo = fusion(campo, img, -parameters.object_size[0]/2-80 , 0)

    # -- Add random phase mask --
    random_phase = np.random.rand(parameters.resolution[0], parameters.resolution[1]) * 2 * np.pi # máscara de fase aleatoria (rads)
    complex_phase_mask = np.exp(1j*random_phase)

    campo_optico = campo * complex_phase_mask
    
    return campo_optico

def field_from_image(img_path, sep=80):
    """
    Take an image and create the optical field associated with a random phase mask
    
    Input:
        - img_path : file location image in format .jpg, .png
    Output:
        - optic_field: numpy array
        """

    # -- read image --
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)/255
    img = cv2.resize(img, parameters.object_size)

    # -- Create base optic field --
    campo = np.zeros(parameters.resolution)
    
    # Shape the object in the optic field
    campo = fusion(campo, img, -parameters.object_size[0]/2-sep , 0)

    # -- Add random phase mask --
    random_phase = np.random.rand(parameters.resolution[0], parameters.resolution[1]) * 2 * np.pi # máscara de fase aleatoria (rads)
    complex_phase_mask = np.exp(1j*random_phase)

    campo_optico = campo * complex_phase_mask
    
    return campo_optico

bin = np.vectorize( lambda x: 0 if (0<x<=np.pi) else (1 if  np.pi < x < 2*np.pi else x) ) 
