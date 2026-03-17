from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss

        # Gradient calculation:
        for j in range(num_classes):
            if j == y[i]:
                # dL/ds_j = p_j - 1 for the correct class
                dW[:, j] += (p[j] - 1) * X[i]
            else:
                # dL/ds_j = p_j for all other classes
                dW[:, j] += p[j] * X[i]

    # Average and add regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W
 
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    # --- Step 1: Compute scores (N, C) ---
    scores = X.dot(W)

    # --- Step 2: Numerical Stability ---
    # Subtract max from each row (keepdims is vital for broadcasting)
    scores -= np.max(scores, axis=1, keepdims=True)

    # --- Step 3: Compute Probabilities (N, C) ---
    exp_scores = np.exp(scores)
    # Divide each row by its sum
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    correct_class_probs = probs[np.arange(num_train), y]
    loss = -np.sum(np.log(correct_class_probs)) / num_train
    loss += reg * np.sum(W * W) # Regularization


    dscores = probs # (N, C)
    # Subtract 1 from the correct class positions (the p - 1 term)
    dscores[np.arange(num_train), y] -= 1
    
    # Backpropagate to weights: (D, N) dot (N, C) = (D, C)
    dW = X.T.dot(dscores) / num_train
    dW += 2 * reg * W # Gradient of regularization



    return loss, dW
