import numpy as np
from typing import List

def conv1d_out(input_shape:int, kernel_size:int, padding:int, stride:int, dialation:int):
        
    output_length = np.floor(((input_shape + 2*padding - dialation * (kernel_size - 1) -1) / stride) + 1)

    return output_length

def conv2d_out(input_shape: List[int], kernel_size: List[int], padding: List[int], stride: List[int], dialation: List[int]):

    output_hight = np.floor(((input_shape[0] + 2*padding[0] - dialation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)
        
    output_width = np.floor(((input_shape[1] + 2*padding[1] - dialation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)

    return [output_hight, output_width]

def conv3_out(input_shape: List[int], kernel_size: List[int], padding: List[int], stride: List[int], dialation: List[int]):

    output_depth = np.floor(((input_shape[0] + 2*padding[0] - dialation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)

    output_hight = np.floor(((input_shape[1] + 2*padding[1] - dialation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)
        
    output_width = np.floor(((input_shape[2] + 2*padding[2] - dialation[2] * (kernel_size[2] - 1) - 1) / stride[2]) + 1)

    return [output_depth, output_hight, output_width]


def convt1d_out(input_shape:int, kernel_size:int, padding:int, output_padding:int, stride:int, dialation:int):
        
    output_length = (input_shape - 1) * stride - 2*padding + dialation * (kernel_size - 1) + output_padding + 1

    return output_length

def convt2d_out(input_shape: List[int], kernel_size: List[int], padding: List[int], output_padding: List[int],stride: List[int], dialation: List[int]):
        
    output_hight = (input_shape[0] - 1) * stride[0] - 2*padding[0] + dialation[0] * (kernel_size[0] - 1) + output_padding[0] + 1

    output_width = (input_shape[1] - 1) * stride[1] - 2*padding[1] + dialation[1] * (kernel_size[1] - 1) + output_padding[1] + 1

    return [output_hight, output_width]

def convt3d_out(input_shape: List[int], kernel_size: List[int], padding: List[int], output_padding: List[int], stride: List[int], dialation: List[int]):
        
    output_depth = (input_shape[0] - 1) * stride[0] - 2*padding[0] + dialation[0] * (kernel_size[0] - 1) + output_padding[0] + 1

    output_hight = (input_shape[1] - 1) * stride[1] - 2*padding[1] + dialation[1] * (kernel_size[1] - 1) + output_padding[1] + 1

    output_width = (input_shape[2] - 1) * stride[2] - 2*padding[2] + dialation[2] * (kernel_size[2] - 1) + output_padding[2] + 1

    return [output_depth, output_hight, output_width]