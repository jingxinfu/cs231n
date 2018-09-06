# @Author: jingxin
# @Date:   06-Apr-2018
# @Email:  jingxinfu.tj@gmail.com
# @Last modified by:   jingxin
# @Last modified time: 06-Sep-2018
# @License: MIT: https://opensource.org/licenses/MIT



import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]

  for i in range(num_train):
    scores = np.dot(X[i,:],W)
    shift_scores = scores - max(scores)
    correct_class = shift_scores[y[i]]
    loss += -correct_class + np.log(np.sum(np.exp(shift_scores)))
    for j in range(num_class):
      temp = np.exp(shift_scores[j])/np.sum(np.exp(shift_scores))
      if j == y[i]:
        dW[:,j] += (-1 + temp) * X[i,:]
      else:
        dW[:,j] +=  temp * X[i,:]

  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += 2*reg*W


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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = np.dot(X,W)
  shift_scores = scores - np.max(scores,axis=1).reshape(-1,1)
  coeff_X = np.exp(shift_scores)/np.sum(np.exp(shift_scores),axis=1).reshape(-1,1)
  coeff_X[range(num_train),y] -= 1

  loss = np.sum(- shift_scores.reshape(-1,1) + np.sum(np.exp(shift_scores),axis=1))
  loss /=num_train
  loss += reg*np.sum(W*W)

  dW = np.dot(X.T,coeff_X)
  dW /=num_train
  dW += 2*reg*W



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
