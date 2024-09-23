import numpy as np


def affine_forward(x, w, b):

    out = None
    x_reshaped = np.reshape(x, (x.shape[0], -1))
    out = x_reshaped.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):

    x, w, b = cache
    dx, dw, db = None, None, None
    dw = np.reshape(x, (x.shape[0], -1)).T.dot(dout)
    dw = np.reshape(dw, w.shape)

    db = np.sum(dout, axis=0, keepdims=False)

    dx = dout.dot(w.T)
    dx = np.reshape(dx, x.shape)
    return dx, dw, db


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):

        outputs = 1 / (1 + np.exp(-x))
        cache = outputs
        return outputs, cache

    def backward(self, dout, cache):

        dx = None
        dx = dout * cache * (1 - cache)
        return dx


class Relu:
    def __init__(self):
        pass

    def forward(self, x):

        outputs = None
        cache = None

        return outputs, cache

    def backward(self, dout, cache):

        dx = None
        x = cache  # Cache contains the input 'x' from the forward pass
        dx = dout * (x > 0).float()  # Gradient computation
        pass

        return dx


class LeakyRelu:
    def __init__(self, slope=0.01):
        self.slope = slope

    def forward(self, x):

        outputs = None
        cache = x
        outputs = np.maximum(self.alpha * x, x)
        self.mask = (x <= 0)
        pass

        return outputs, cache

    def backward(self, dout, cache):
        dx = None
        x = cache
        dx = dout.copy()  # Initialize gradient dx as a copy of dout
        dx[self.mask] *= self.alpha
        pass
        return dx


class Tanh:
    def __init__(self):
        pass

    def forward(self, x):

        outputs = None
        cache = None
        outputs = np.tanh(x)
        cache = x  # We cache the input for the backward pass
        return outputs, cache

    def backward(self, dout, cache):
        # The derivative of tanh is (1 - tanh^2)
        dx = dout * (1 - np.tanh(cache)**2)
        return dx
