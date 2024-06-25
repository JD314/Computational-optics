#Version 21-04-2024
#-Añadido fusionm y tajadam basados en coordenadas matriciales.
#Version 11-12-2024
#-Cambiado propft a usar solo torch.

import torch
import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
import cv2

# The codes are optimized to work with the DMD device, conforming to the following parameters.
class parameters:
    coord = (-544,0)
    object_size = (800, 800)
    resolution = (1200, 2048)
    sep = 80


def tajada(M1,M2,x,y):
    """
    This function allows you to trim an x, y centered section of the array M1 of equal size to the array M2.
    equal in size to the array M2. The x,y coordinates are given in a centered Cartesian system.

    Input:
    -----------
    M1 : numpy array
        array from which a section is trimmed.
    M2 : numpy array
        array that determines the size of the clipping.
    x : int
        x-coordinate of the center of the clipping in a Cartesian system centered on M1.
    y : int 
        y-coordinate of the clipping center in a Cartesian system centered on M1.

    Output:
    -----------
        numpy array 
            with shape equal to M2.


    Raises
    -----------
    ValueError: If the size of the area to be trimmed or the coordinates exceed the size of the original array this error will be triggered.
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
    This function allows to insert M2 to a section centered at (x, y) of the array M1.
    The x,y coordinates are given in a centered Cartesian system.

    Input:
    ----------
    M1 : numpy array
        Array into which a section is inserted.
    M2 : numpy array
        Array to insert.
    x : int
        x-coordinate of the center of the cutout in a Cartesian system centered for M1.
    y : int
        y-coordinate of the center of the clipping in a Cartesian system centered for M1....

    Raises
    ------
    ValueError
        If the size of the area to insert or the coordinates exceed the size of the original matrix this error will be triggered.

    Output:
    -------
    MC
        Numpy array with shape equal to M1.

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
    """
    Performs the 2D inverse Fourier transform on a tensor.

    This function applies the two-dimensional (2D) inverse Fourier transform to an `x` input tensor.
    It uses frequency shift changes to center the zero-frequency before and after the transform.

    Input:
    ---------
    x : torch.Tensor
        The input tensor on which the 2D inverse Fourier transform will be performed.

    Output:
    --------
    torch.Tensor
        The resulting tensor after applying the 2D inverse Fourier transform.
    """
    x=torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x)))
    return x

def tfourier(x):
    """
    Performs the 2D Fourier transform on a tensor.

    This function applies the two-dimensional (2D) Fourier transform to an `x` input tensor.
    It uses frequency shift changes to center the zero-frequency before and after the transform.

    Input:
    ---------
    x : torch.Tensor
        The input tensor on which the 2D Fourier transform will be performed.

    Output:
    --------
    torch.Tensor
        The resulting tensor after applying the 2D Fourier transform.
    """
    x=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))
    return x

def corr2(A,B):
    """
    Calculates the correlation coefficient between two matrices.

    This function computes the Pearson correlation coefficient between the entries `A` and `B`.
    The correlation coefficient is a measure of the linear relationship between two matrices.

    Input:
    -----------
    A : torch.Tensor
        The first input matrix.
    B : torch.Tensor
        The second input matrix.

    Output:
    --------
    torch.Tensor
        The correlation coefficient between `A` and `B`. This value is a scalar that
        measures the strength and direction of the linear relationship between the two matrices.
    """
    cc=(((A-torch.mean(A))*((B-torch.mean(B)))).sum())/(torch.sqrt((A-torch.mean(A)).pow(2).sum()*(B-torch.mean(B)).pow(2).sum()))
    return cc

def test_field():
    """
    Generates a test optical field using a random phase mask by default.

    This function creates a simulated test optical field using a random phase mask.
    It does not require additional input parameters.

    Returns:
    --------
    optical_field : numpy.ndarray
        A numpy array representing the optical field generated with a random phase mask."""
    
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

def field_from_image(img_path, sep=80, tensor=True):
    """
    Takes an image and creates the optical field associated with a random phase mask.

    Input:
    -----------
    img_path : str
        Location of the image file in .jpg or .png format.
    sep (w): int, optional 
        Separation used to position the image in the optical field (default is 80).
    tensor: bool
        Condition to use torch tensor
    Ouptut:
    --------
    optical_field : numpy.ndarray or torch.tensor
        An array/tensor representing the resulting optical field after applying the random phase mask to
        the random phase mask to the processed image.
    """
    # -- read image --
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)/255
    img = cv2.resize(img, parameters.object_size)

    # -- Create base optic field --
    campo = np.zeros(parameters.resolution)
    
    # Shape the object in the optic field
    campo = fusion(campo, img, parameters.coord[0], parameters.coord[1])

    # -- Add random phase mask --
    random_phase = np.random.rand(parameters.resolution[0], parameters.resolution[1]) * 2 * np.pi # máscara de fase aleatoria (rads)
    complex_phase_mask = np.exp(1j*random_phase)

    campo_optico = campo * complex_phase_mask
    
    if tensor:
        return torch.from_numpy(campo_optico)
    
    return campo_optico

def bin_tensor(x):
    """
    Vectorized function that applies a conditional operation to a tensor in PyTorch.

    This function applies a conditional operation to each element of the tensor `x`:
    - If 0 < x <= π, it assigns 0.
    - If π < x < 2π, it assigns 1.

    Input:
    -----------
    x : torch.Tensor
        Input tensor to which the function will be applied.

    Output:
    --------
    torch.Tensor
        Tensor with the modified elements according to the specified conditions.
    """

    condition1 = (0 < x) & (x <= np.pi)
    condition2 = (np.pi < x) & (x < 2*np.pi)
    
    result = torch.where(condition1, torch.tensor(0.0), x)
    result = torch.where(condition2, torch.tensor(1.0), result)
    
    return result

bin = np.vectorize( lambda x: 0 if (0<x<=np.pi) else (1 if  np.pi < x < 2*np.pi else x) ) 

def masks(height=1200, weight=2048, number=1):
    """
    Create a series of identical random phase masks.

    This function generates `number` identical random phase masks of dimensions (`height`, `weight`),
    where each element of the mask is a complex number with a random phase uniformly distributed between 0 and 2π.

    Input:
    -----------
    height : int
        The height of each phase mask.
    weight : int
        The width of each phase mask.
    number : int
        The number of phase masks to generate.

    Output:
    --------
    torch.Tensor
        A complex tensor of dimensions (`number`, `height`, `weight`), where each mask is identical
        and contains complex numbers of unit magnitude with random phases.
    """
     
    angle_base = (2 * np.pi) * torch.rand(number, height, weight)
    complex_mask = torch.exp(1j * angle_base)
    return complex_mask

def cut_image(optic_field):
    """
    Extracts the absolute magnitude of a region of interest (ROI) from a given optical field.

    This function calculates the absolute magnitude of the region of interest (ROI) within the provided optical field.
    The ROI is defined and extracted using the `slice` function with the coordinates and dimensions specified in `parameters`.

    Parameters:
    -----------
    optic_field : torch.Tensor
        Tensor representing the optical field from which the ROI will be extracted.

    Returns:
    --------
    torch.Tensor
        Tensor with the absolute magnitude of the region of interest (ROI) of the optical field.
    """
    # -- Region of interest --
    ROI = tajada(optic_field, np.zeros(parameters.object_size), parameters.coord[0], parameters.coord[1])
    
    return torch.abs(ROI)

def normalize_angle(phase):
    """
    Normalizes angles to a specified range [0, 2π).

    This function takes a tensor of `phase` angles and normalizes them to be within the range [0, 2π).
    It uses the modulus operation (`torch.mod`) to adjust the angles.

    Parameters:
    -----------
    phase : torch.Tensor
        Tensor containing the angles to be normalized.

    Returns:
    --------
    torch.Tensor
        Tensor containing the angles normalized to the range [0, 2π).
    """
    return torch.remainder(phase, 2*np.pi)
    
def normalize_image(img):
    pass

