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
    num_classes = W.shape[1]
    scores = X.dot(W)
    y_pos = [yi * num_classes + yv for yi, yv in enumerate(y)]

    y_mat = np.zeros(num_train * num_classes)
    y_mat[y_pos] = 1
    y_mat = y_mat.reshape(num_train, num_classes)

    correct_scores = scores.reshape(num_classes * num_train)[y_pos].reshape([num_train, 1])
    loss = -correct_scores + np.log(np.sum(np.exp(scores), 1))

    denominator = np.sum(np.exp(scores), 1).reshape(num_train, 1)

    # scores -= np.max(scores, 1).reshape(500, 1)  # stablizing
    dScores = np.exp(scores) / denominator - y_mat
    dW = X.T.dot(dScores)
    dW /= num_train

    loss = np.mean(loss) + 0.5 * reg * np.sum(W * W)
    dW += 0.5 * reg * 2 * W

    # num_examples = X.shape[0]
    # scores = np.dot(X, W)

    # # compute the class probabilities
    # exp_scores = np.exp(scores)
    # probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

    # # compute the loss: average cross-entropy loss and regularization
    # corect_logprobs = -np.log(probs[range(num_examples),y])
    # data_loss = np.sum(corect_logprobs)/num_examples
    # reg_loss = 0.5*reg*np.sum(W*W)
    # loss = data_loss + reg_loss

    # # compute the gradient on scores
    # dscores = probs
    # dscores[range(num_examples),y] -= 1
    # dscores /= num_examples

    # # backpropate the gradient to the parameters (W,b)
    # dW = np.dot(X.T, dscores)
    # db = np.sum(dscores, axis=0, keepdims=True)

    # dW += reg*W # regularization gradient
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
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return softmax_loss_naive(W, X, y, reg)
    # return loss, dW
