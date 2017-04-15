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
  
  for i in xrange(num_train):    
    scores = X[i].dot(W)
    ei = 0
    for j in xrange(num_class):        
        ei += np.exp(scores[j])
    
    for k in xrange(num_class):
        if k == y[i]:
            dW[:,k] += X[i]*(np.exp(scores[k])/ei - 1)
        else:
            dW[:,k] += X[i]*(np.exp(scores[k])/ei)
    
    
    loss -= np.log(np.exp(scores[y[i]])/ei)
  loss /= num_train    
  loss += 0.5 * reg * np.sum(W**2)

  dW /= num_train    
  dW += reg * W
 
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
    
  #for i in xrange(num_train):    
  scores = X.dot(W)     
  scores -= np.max(scores)
  ei = np.sum(np.exp(scores), axis=1)
  loss = np.sum(-np.log(np.exp(scores[range(num_train),y])/ei)) 
    
  loss /= num_train    
  loss += 0.5 * reg * np.sum(W**2)
    
  e_mat = np.exp(scores)/ei.reshape(-1,1) 
  #e_mat = np.exp(scores)/np.matrix(ei).T
  e_mat[range(num_train),y] -= 1    
  dW += np.dot(X.T, e_mat)  
     
  dW /= num_train    
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

