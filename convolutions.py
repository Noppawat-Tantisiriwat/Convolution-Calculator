import numpy as np
from formulae import *

# Convolutions
class ConvBlock:
    
    def __init__(self, dim, kernel_size, padding, stride, dialation,name="Conv"):
        
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dialation = dialation
        self.name = name
    
    def forward(self, input_shape):

        if self.dim == 1:
            
            return conv1d_out(input_shape=input_shape,
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        stride=self.stride,
                                        dialation=self.dialation)

        elif self.dim == 2:

            return conv2d_out(input_shape=input_shape,
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        stride=self.stride,
                                        dialation=self.dialation)

        elif self.dim == 3:
            
            return conv3d_out(input_shape=input_shape,
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        stride=self.stride,
                                        dialation=self.dialation)

        else:
            raise NotImplementedError


    def __str__(self):

        return      {"name" : self.name,
                    "dimention" :self.dim,
                    "kernel_size":self.kernel,
                    "padding":self.padding,
                    "stride":self.stride,
                    "dilation":self.dialation}


# Convolution Transpose
class ConvTBlock(ConvBlock):

    def __init__(self, dim, kernel_size, padding, output_padding, stride, dialation,name="Conv"):
        super(ConvTBlock, self).__init__(dim, kernel_size, padding, stride, dialation)
        
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.output_padding = output_padding
        self.stride = stride
        self.dialation = dialation
        self.name = name
        
    def forward(self, input_shape):
        if self.dim == 1:
            
            return convt1d_out(input_shape=input_shape,
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        output_padding=self.output_padding,
                                        stride=self.stride,
                                        dialation=self.dialation)

        elif self.dim == 2:

            return convt2d_out(input_shape=input_shape,
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        output_padding=self.output_padding,
                                        stride=self.stride,
                                        dialation=self.dialation)

        elif self.dim == 3:
            
            return convt3d_out(input_shape=input_shape,
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        output_padding=self.output_padding,
                                        stride=self.stride,
                                        dialation=self.dialation)

        else:
            raise NotImplementedError


    def __str__(self):

        return      {"name" : self.name,
                    "dimention" :self.dim,
                    "kernel_size":self.kernel,
                    "padding":self.padding,
                    "output_padding":self.output_padding,
                    "stride":self.stride,
                    "dilation":self.dialation}
