import numpy as np
import torch

# PARAMETERS
"""
    n  # number of nodes
    d_x = 2  # dimension of the input feature (2 for TSP in ALTSTSP article)
    d_h = 128  # dimension of node embedding (128 in ALTSTSP article)
    N  # number of attention layers
    M  # number of heads (8 in ALTSTSP article)
    dim_FF = 512  # dimension of the FF hidden sublayer (512 in ALTSTSP article)
"""

# dimensions
# n = 100  # number of nodes
N = 3  # number of attention layers
M = 8  # number of heads (8 in ALTSTSP article)
# d_x = 2  # dimension of the input feature (2 for TSP in ALTSTSP article)
d_h = 128  # dimension of node embedding (128 in ALTSTSP article)
dim_FF = 512  # dimension of the FF hidden sublayer (512 in ALTSTSP article)

# matrices
# W = np.ones((d_h, d_x))  # TODO learn W
# b = np.ones(d_h)  # TODO learn b
# x = np.ones((n, d_x))  # TODO input
c = np.zeros()  # context node


def encode(x, W, b):

    (n, d_x) = x.shape()
    h = np.zeros((N, n, d_h))

    # Compute the initial nodes embedding for each node
    for i in range(n):
        h[0][i] = np.dot(W, x[i]) + b

    # Update the nodes N times with 1 MHA and 1 FF
    for j in range(1, N+1):
        h[j] = attention_layer(h[j-1], j)

    # Compute h_graph : the graph embedding which is the mean of all the node embeddings after the N attention layers
    h_graph = 1 / N * np.sum(h[N], 1)  # TODO check if 1 is the right axis (if not, it is zero)

    return h[N], h_graph


def attention_layer(h, j):
    # TODO
    (n, d_h) = h.shape()
    h_temp = np.zeros((n, d_h))  # TODO can be replaces as just an h_temp vector inside the loop to use les space -- might be a problem for learning ?
    h_output = np.zeros((n, d_h))
    for i in range(n):  # for each node
        # STEP 1 : MHA (+ add a skip-connection + batch normalization)
        h_temp[i] = batch_normalization(h[i] + mha(h))
        # STEP 2 : FF layer (+ add a skip-connection + batch normalization)
        h_output[i] = batch_normalization(h_temp[i] + ff(h_temp[i]))
    return h_output

def batch_normalization(h_vector):
    d_h = len(h_vector)
    # TODO batch nomalization
    return h_vector

def mha(h):
    #  TODO mha
    n, d_h = h.shape()
    output = np.zeros(d_h)
    return output

def ff(h_vector):
    # TODO feed forward layer
    return h_vector