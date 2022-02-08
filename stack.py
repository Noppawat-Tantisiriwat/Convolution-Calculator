from typing import List

class Stack:

    def __init__(self, function_list: List, name="calc1"):

        self.functions = function_list
        self.name = name

    def forward(self, input_shape):

        x = input_shape

        for function in self.functions:
            x = function.forward(x)

        return x

    def __str__(self):

        return self.name + "\n ____________ " + "layers : " + str(len(self.functions)) 

