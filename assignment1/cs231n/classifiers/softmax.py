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
  train_num = X.shape[0]
  class_num = W.shape[1]
    
  for i in xrange(train_num):
    f_values = X[i].dot(W)
    f_values -= np.max(f_values)
    p = np.exp(f_values)/np.sum(np.exp(f_values))
    loss += -np.log(p[y[i]])
    for k in range(class_num):
        if (k == y[i]):
            dW[:,k] += (p[k]-1)*X[i]
        else:
            dW[:,k] += p[k]*X[i]
        
        
        
  loss = loss/train_num
  loss = loss+ reg*np.sum(W*W)
  dW = dW/train_num
  dW = dW+ 2*reg*W
  
    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  train_num = X.shape[0]
  
  f_values = X.dot(W)
  max_value = np.max(f_values, axis = 1, keepdims=True)
  f_values -= max_value
  f_values = np.exp(f_values)
  sum_value = np.sum(f_values, axis = 1, keepdims = True)
  p = f_values/sum_value
  loss  = np.sum(-np.log(p[np.arange(train_num),y]))
  
  index = np.zeros_like(p)
  index[np.arange(train_num), y] = 1
  dW = X.T.dot(p - index)
   
  loss = loss/train_num
  loss = loss+ reg*np.sum(W*W)
  dW = dW/train_num
  dW = dW+ 2*reg*W
  

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

