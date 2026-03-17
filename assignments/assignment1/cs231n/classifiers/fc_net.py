from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        
        
        scores = None
        out1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, cache2 = affine_forward(out1, self.params['W2'], self.params['b2'])
        
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
        loss, grads = 0, {}
        loss, dloss = softmax_loss(scores, y)
        
     
        # Add L2 regularization (with 0.5 factor!)
        loss += 0.5 * self.reg * (
            np.sum(self.params['W1'] ** 2) +
            np.sum(self.params['W2'] ** 2)
        )
    
        grads = {}
    
        # Backward pass
        dout2, grads["W2"], grads["b2"] = affine_backward(dloss, cache2)
        dout1, grads["W1"], grads["b1"] = affine_relu_backward(dout2, cache1)
    
        # Add regularization to gradients
        grads["W2"] += self.reg * self.params["W2"]
        grads["W1"] += self.reg * self.params["W1"]

        return loss, grads

    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True



class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        
        layer_dims = [input_dim] + hidden_dims + [num_classes]

        for i in range(self.num_layers):
            self.params[f'W{i+1}'] = weight_scale * np.random.randn(
                layer_dims[i], layer_dims[i+1]
            )
            self.params[f'b{i+1}'] = np.zeros(layer_dims[i+1])
        
            # If using batchnorm, initialize gamma and beta for hidden layers only
            if self.normalization in ["batchnorm", "layernorm"] and i < self.num_layers - 1:
                self.params[f'gamma{i+1}'] = np.ones(layer_dims[i+1])
                self.params[f'beta{i+1}'] = np.zeros(layer_dims[i+1])

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        scores = X
        caches = {}
        
        # Forward through hidden layers
        for i in range(1, self.num_layers):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
        
            scores, fc_cache = affine_forward(scores, W, b)
        
            # BatchNorm (hidden layers only)
            if self.normalization == "batchnorm":
                gamma = self.params[f"gamma{i}"]
                beta = self.params[f"beta{i}"]
                scores, bn_cache = batchnorm_forward(
                    scores, gamma, beta, self.bn_params[i-1]
                )
            else:
                bn_cache = None
        
            scores, relu_cache = relu_forward(scores)
        
            # Dropout
            if self.use_dropout:
                scores, do_cache = dropout_forward(scores, self.dropout_param)
            else:
                do_cache = None
        
            caches[i] = (fc_cache, bn_cache, relu_cache, do_cache)
        
        # Final affine layer (no relu, no dropout)
        W = self.params[f"W{self.num_layers}"]
        b = self.params[f"b{self.num_layers}"]
        
        scores, final_cache = affine_forward(scores, W, b)

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        loss, dscores = softmax_loss(scores, y)

        # Add L2 regularization
        for i in range(1, self.num_layers + 1):
            W = self.params[f"W{i}"]
            loss += 0.5 * self.reg * np.sum(W * W)
    
        grads = {}
        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        
        
        dx, dW, db = affine_backward(dscores, final_cache)
        grads[f"W{self.num_layers}"] = dW + self.reg * self.params[f"W{self.num_layers}"]
        grads[f"b{self.num_layers}"] = db
        
        
        for i in reversed(range(1, self.num_layers)):
            fc_cache, bn_cache, relu_cache, do_cache = caches[i]
        
            # Dropout backward
            if self.use_dropout:
                dx = dropout_backward(dx, do_cache)
        
            # ReLU backward
            dx = relu_backward(dx, relu_cache)
        
            # Batchnorm backward
            if self.normalization == "batchnorm":
                dx, dgamma, dbeta = batchnorm_backward(dx, bn_cache)
                grads[f"gamma{i}"] = dgamma
                grads[f"beta{i}"] = dbeta
        
            # Affine backward
            dx, dW, db = affine_backward(dx, fc_cache)
        
            grads[f"W{i}"] = dW + self.reg * self.params[f"W{i}"]
            grads[f"b{i}"] = db

        return loss, grads


    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True