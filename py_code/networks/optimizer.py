import numpy as np

class SGD(object):
    def __init__(self, model, loss_func, learning_rate=1e-4):
        self.model = model
        self.loss_func = loss_func
        self.lr = learning_rate
        self.grads = None

    def backward(self, y_pred, y_true):
        """
        Compute the gradients wrt the weights of your model
        """
        dout = self.loss_func.backward(y_pred, y_true)
        self.model.backward(dout)

    def _update(self, w, dw, lr):
        """
        Update a model parameter
        """
        w -= lr * dw
        return w

    def step(self):
        """
        Perform an update step with the update function, using the current
        gradients of the model
        """

        # Iterate over all parameters
        for name in self.model.grads.keys():

            # Unpack parameter and gradient
            w = self.model.params[name]
            dw = self.model.grads[name]

            # Update the parameter
            w_updated = self._update(w, dw, lr=self.lr)
            self.model.params[name] = w_updated

            # Reset gradient
            self.model.grads[name] = 0.0


class sgd_momentum(object):

    def __init__(self, model, loss_func, learning_rate=1e-4, **kwargs):
        self.model = model
        self.loss_func = loss_func
        self.lr = learning_rate
        self.grads = None
        self.optim_config = kwargs.pop('optim_config', {})
        self._reset()

    def _reset(self):
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_configs.items()}
            self.optim_configs[p] = d

    def backward(self, y_pred, y_true):
        """
        Compute the gradients wrt the weights of your model
        """
        dout = self.loss_func.backward(y_pred, y_true)
        self.model.backward(dout)

    def _update(self, w, dw, config, lr):
        """
        Update a model parameter
        """
        if config is None:
            config = {}
        config.setdefault('momentum', 0.9)
        v = config.get('velocity', np.zeros_like(w))
        next_w = None

        mu = config['momentum']
        learning_rate = lr
        v = mu * v - learning_rate * dw
        next_w = w + v
        config['velocity'] = v

        return next_w, config

    def step(self):
        """
        Perform an update step with the update function, using the current
        gradients of the model
        """

        # Iterate over all parameters
        for name in self.model.grads.keys():

            # Unpack parameter and gradient
            w = self.model.params[name]
            dw = self.model.grads[name]

            config = self.optim_configs[name]

            # Update the parameter
            w_updated, config = self._update(w, dw, config, lr=self.lr)
            self.model.params[name] = w_updated
            self.optim_configs[name] = config
            # Reset gradient
            self.model.grads[name] = 0.0


class Adam(object):

    def __init__(self, model, loss_func, learning_rate=1e-4, **kwargs):
        self.model = model
        self.loss_func = loss_func
        self.lr = learning_rate
        self.grads = None

        self.optim_config = kwargs.pop('optim_config', {})

        self._reset()

    def _reset(self):
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_configs.items()}
            self.optim_configs[p] = d

    def backward(self, y_pred, y_true):
        """
        Compute the gradients wrt the weights of your model
        """
        dout = self.loss_func.backward(y_pred, y_true)
        self.model.backward(dout)

    def _update(self, w, dw, config, lr):
        """
        Update a model parameter
        """
        if config is None:
            config = {}
        config.setdefault('beta1', 0.9)
        config.setdefault('beta2', 0.999)
        config.setdefault('epsilon', 1e-4)
        config.setdefault('m', np.zeros_like(w))
        config.setdefault('v', np.zeros_like(w))
        config.setdefault('t', 0)
        next_w = None

        learning_rate = lr
        m = config['m']
        v = config['v']
        t = config['t']
        beta1 = config['beta1']
        beta2 = config['beta2']
        eps = config['epsilon']

        m = beta1 * m + (1 - beta1) * dw
        m_hat = m / (1 - np.power(beta1, t + 1))
        v = beta2 * v + (1 - beta2) * (dw ** 2)
        v_hat = v / (1 - np.power(beta2, t + 1))
        next_w = w - learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        config['t'] = t + 1
        config['m'] = m
        config['v'] = v


        return next_w, config

    def step(self):
        """
        Perform an update step with the update function, using the current
        gradients of the model
        """

        # Iterate over all parameters
        for name in self.model.grads.keys():

            # Unpack parameter and gradient
            w = self.model.params[name]
            dw = self.model.grads[name]

            config = self.optim_configs[name]

            # Update the parameter
            w_updated, config = self._update(w, dw, config, lr=self.lr)
            self.model.params[name] = w_updated
            self.optim_configs[name] = config
            # Reset gradient
            self.model.grads[name] = 0.0
