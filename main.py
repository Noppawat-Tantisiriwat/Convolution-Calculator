from stack import Stack
from formulae import *
from convolutions import ConvBlock, ConvTBlock

# example use
encoder2d = Stack([
    ConvBlock(2,[3, 3], [1, 1], [3, 3], [0, 0]),
    ConvBlock(2,[4, 4], [0, 0], [3, 3], [1, 1]),
    ConvBlock(2,[3, 3], [1, 1], [1, 1], [0, 0]),
    ConvBlock(2,[4, 4], [0, 0], [1, 1], [1, 1]),
    ConvBlock(2,[3, 3], [1, 1], [1, 1], [0, 0])
], name="encoder2d")


decoder2d = Stack([
    ConvTBlock(2,[3, 3], [1, 1], [0, 0], [1, 1], [0, 0]),
    ConvTBlock(2,[4, 4], [0, 0], [0, 0], [1, 1], [1, 1]),
    ConvTBlock(2,[3, 3], [1, 1], [0, 0], [1, 1], [0, 0]),
    ConvTBlock(2,[4, 4], [0, 0], [0, 0], [3, 3], [1, 1]),
    ConvTBlock(2,[3, 3], [1, 1], [2, 2], [3, 3], [0, 0])
], name="decoder2d")

ae = Stack([
    encoder2d,
    decoder2d
])


result = ae.forward([244, 244])

print(result == [244, 244])