import math
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import random
import tensorflow as tf
from collections import defaultdict
import pandas as pd
import random
from text_encoding.get_text_embedding import get_text_embed

def glorot(shape, name=None):
    """
        Glorot & Bengio (AISTATS 2010) init.
    """
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def adj_to_bias(adj, sizes, nhood=1):
    """
        create bias
    """
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)

def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)
    return feat_norm

def sparse_matrix(matrix):
    sigma = 0.001
    matrix = matrix.astype(np.int32)
    result = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        if matrix[i,0] == 0:
           result[i,0] = sigma
        else:
           result[i,0] = 1
    return result

def gaussiansimilarity(interaction, m, n):
    """
        create GIPKS for diseases and biomarkers
        interaction_shape:(m*n)
    """

    # disease Gaussian
    gamad = n / (math.pow(np.linalg.norm(interaction), 2))
    C = interaction.T.conjugate()
    kd = np.zeros((n, n))

    D = np.matmul(C.T.conjugate(), C)
    for i in range(n):
        for j in range(i, n):
            kd[i][j] = np.exp(-gamad * (D[i][i] + D[j][j] - 2 * D[i][j]))

    kd = kd + kd.T.conjugate() - np.diag(np.diag(kd))

    # biomarker Gaussian
    gammam = m / (math.pow(np.linalg.norm(interaction), 2))
    km = np.zeros((m, m))
    E = np.matmul(C, C.T.conjugate())
    for i in range(m):
        for j in range(i, m):
            km[i][j] = np.exp(-gammam * (E[i][i] + E[j][j] - 2 * E[i][j]))
    km = km + km.T.conjugate() - np.diag(np.diag(km))
    return kd, km

def generate(M, test_arr, dataset):
    """
        generate disease and biomarker similarity
        return: F1, F2
    """
    labels = np.loadtxt('data/' + dataset + '/adj.txt')
    for i in range(len(test_arr)):
        disease = labels[test_arr[i], 0]
        biomarker = labels[test_arr[i], 1]
        M[int(disease) - 1, int(biomarker) - 1] = 0  # todo:创建完新的M后，需要重新计算Gaussian

    n = len(M)
    m = len(M[0, :])

    D_GSM, B_GSM = gaussiansimilarity(M, m=m, n=n)
    D_SSM = np.loadtxt('data/' + dataset + '/D_SSM.txt')

    if dataset == 'HMDD':
        B_SM = np.loadtxt('data/' + dataset + '/M_PSM.txt')
    elif dataset == 'HMDAD':
        B_SM = np.loadtxt('data/' + dataset + '/M_SSM.txt')
    elif dataset == 'LncRNADisease':
        B_SM = np.loadtxt('data/' + dataset + '/L_PSM.txt')

    F1 = np.zeros((n, n))
    F2 = np.zeros((m, m))

    for i in range(n):
        for j in range(n):
            if D_SSM[i, j] != 0:
                F1[i, j] = (D_SSM[i, j] + D_GSM[i, j]) / 2
            else:
                F1[i, j] = D_GSM[i, j]

    for i in range(m):
        for j in range(m):
            if B_SM[i, j] != 0:
                F2[i, j] = (B_SM[i, j] + B_GSM[i, j]) / 2
            else:
                F2[i, j] = B_GSM[i, j]

    return F1, F2

def load_data(train_arr, test_arr, dataset):
    labels = np.loadtxt('data/'+dataset+'/adj.txt')
    n = np.max(labels[:,0])
    m = np.max(labels[:,1])
    n = n.astype(np.int32)
    m = m.astype(np.int32)

    logits_test = sp.csr_matrix((labels[test_arr,2],(labels[test_arr,0]-1, labels[test_arr,1]-1)),shape=(n,m)).toarray()
    logits_test = logits_test.reshape([-1,1])  

    logits_train = sp.csr_matrix((labels[train_arr,2],(labels[train_arr,0]-1, labels[train_arr,1]-1)),shape=(n,m)).toarray()
    logits_train = logits_train.reshape([-1,1])

    train_mask = np.array(logits_train[:,0], dtype=np.bool).reshape([-1,1])
    test_mask = np.array(logits_test[:,0], dtype=np.bool).reshape([-1,1])
    
    M = np.loadtxt('data/'+dataset+'/interaction.txt')

    interaction = np.vstack((np.hstack((np.zeros(shape=(n,n),dtype=int),M)),np.hstack((M.transpose(),np.zeros(shape=(m,m),dtype=int)))))

    F1, F2 = generate(M, test_arr, dataset)

    features = np.vstack((np.hstack((F1,np.zeros(shape=(F1.shape[0],F2.shape[1]),dtype=int))), np.hstack((np.zeros(shape=(F2.shape[0],F1.shape[0]),dtype=int), F2))))
    features = normalize_features(features)

    return interaction, features, sparse_matrix(logits_train), logits_test, train_mask, test_mask, labels

def load_text_feat(arr, dataset):
    """
        load text feature
    """
    disease_text, biomarker_text = get_text_embed(arr,dataset)
    return disease_text, biomarker_text

def masked_accuracy(preds, labels, mask, negative_mask):
    """
        Calculation of loss
    """
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    error = tf.square(preds-labels)
    mask += negative_mask
    mask = tf.cast(mask, dtype=tf.float32)
    error *= mask
    return tf.sqrt(tf.reduce_mean(error))

def construct_labels_with_scores(outs, labels, test_arr, label_neg):
    """
        build the labels and scores
    """
    scores = []
    for i in range(len(test_arr)):
        l = test_arr[i]
        scores.append(outs[int(labels[l,0]-1),int(labels[l,1]-1)])
    for i in range(label_neg.shape[0]):
        scores.append(outs[int(label_neg[i,0]),int(label_neg[i,1])])
    test_labels = np.ones((len(test_arr),1))
    temp = np.zeros((label_neg.shape[0],1))
    test_labels1 = np.vstack((test_labels,temp))
    test_labels1 = np.array(test_labels1,dtype=np.bool).reshape([-1,1])
    return test_labels1, scores