import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import time
import traceback

def standardize(data):  # auxiliary function
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def elem_sympoly(lamda, k):
    N = len(lamda)
    E = np.zeros((k + 1, N + 1))
    E[0, :] = 1
    for l in range(1, k + 1):
        for n in range(1, N + 1):
            E[l, n] = E[l, n - 1] + lamda[n - 1] * E[l - 1, n - 1]
    return E

def sample_k(lamda, k):
    E = elem_sympoly(lamda, k)

    i = len(lamda) - 1
    remaining = k - 1
    S = np.zeros((k, ))

    while remaining > 0:
        if i == remaining:
            marg = 1
        else:
            marg = lamda[i] * E[remaining, i] / E[remaining + 1, i + 1]

        if np.random.uniform() < marg:
            S[remaining] = i
            remaining -= 1

        i -= 1
    return S.astype('int')

def decompose_kernel_original(M, exponent=6):
    D, V = np.linalg.eigh(M + 10**(-exponent) * np.identity(M.shape[0]))
    C = dict()
    C['M'] = M
    C['V'] = np.real(V)
    C['D'] = np.real(D)
    return C

def decompose_kernel(M, exponent=6):
    M += 10**(-exponent) * np.identity(M.shape[0])
    D, V = np.linalg.eigh(M)
    D = np.maximum(D, 1e-6)   
    V = np.real(V)
    V /= np.linalg.norm(V, axis=0)    
    return  {'M': M, 'V': V, 'D': D} 

def RFF(data, D_feature):
    N_ground, D_data = data.shape

    Ws = np.random.normal(size=(int(D_feature / 2),
                                D_data))  
    feature = np.zeros((D_feature, N_ground))
    feature[1::2] = np.sqrt(2 / D_feature) * np.sin(np.matmul(Ws, data.T))
    feature[::2] = np.sqrt(2 / D_feature) * np.cos(np.matmul(Ws, data.T))

    return feature

def sample_dual_dpp_original(B, C, k):
    v = sample_k(C['D'], k)

    k = len(v)
    V = (C['V'])[:, v]

    V = V * 1. / np.sqrt(C['D'][v])
    Y = np.zeros((k, )).astype(int)

    for i in range(k - 1, -1, -1):
        P = np.sum(np.square(B @ V), axis=1)
        P = P / np.sum(P)

        Y[i] = (np.argwhere(np.random.uniform() <= np.cumsum(P)))[0]

        S = B[Y[i], :] @ V
        j = np.argwhere(S != 0)[0]
        Vj = V[:, j]
        Sj = S[j]
        V = np.delete(V, obj=j, axis=1)
        S = np.delete(S, j)
        V = V - Vj * (S / Sj)

        for a in range(i - 1):
            for b in range(a - 1):
                V[:, a] = V[:, a] - (np.expand_dims(V[:, a], axis=-1).T
                                     @ C['M'] @ V[:, b]) * V[:, b]
            V[:, a] = V[:, a] / np.sqrt(
                np.expand_dims(V[:, a], axis=-1).T @ C['M'] @ V[:, a])

    return np.sort(Y)

def sample_dual_dpp(B, C, k):
    try:
        v = sample_k(C['D'], k)

        k = len(v)
        V = (C['V'])[:, v]
        V = V * 1. / np.sqrt(C['D'][v])
        Y = np.zeros((k, )).astype(int)

        for i in range(k - 1, -1, -1):
            P = np.sum(np.square(B @ V), axis=1)
            P = P / np.sum(P)

            Y[i] = (np.argwhere(np.random.uniform() <= np.cumsum(P)))[0]       

            S = B[Y[i], :] @ V
            j = np.argwhere(S != 0)[0]    
            Vj = V[:, j]
            Sj = S[j]
            V = np.delete(V, obj=j, axis=1)
            S = np.delete(S, j)
            V = V - Vj * (S / Sj)

            for a in range(i): 
                for b in range(a): 
                    V[:, a] = V[:, a] - (np.expand_dims(V[:, a], axis=-1).T
                                         @ C['M'] @ V[:, b]) * V[:, b]
                V[:, a] = V[:, a] / np.sqrt(
                    np.expand_dims(V[:, a], axis=-1).T @ C['M'] @ V[:, a])

        return np.sort(Y)
    except IndexError:
        traceback.print_exc()
        import pdb; pdb.set_trace()

def k_Markov_dual_DPP(feature, id_dpp_left, batch_size, exponent):
    tmp = feature @ feature.T
    L_tmp = decompose_kernel(tmp, exponent)
    id_dpp_new_rel = sample_dual_dpp(feature.T, L_tmp, batch_size).T
    id_dpp_new_abs = id_dpp_left[id_dpp_new_rel]
    return id_dpp_new_rel, id_dpp_new_abs, L_tmp

def Markov_update_RFF(V, id_sampled, exponent=6):
    V = V.T
    A = id_sampled
    A_bar = np.setdiff1d(range(V.shape[0]), A)

    Z = np.identity(V.shape[1]) - V[A, :].T @ (np.linalg.solve(
        V[A, :] @ V[A, :].T + 10**(-exponent) * np.identity(len(A)), V[A, :]))
    return (V[A_bar, :] @ Z.T).T


def preference(feature_prop, prop_bias_type, prop_pred):
    if prop_bias_type == 'exploration':
        quality = np.sum(prop_pred**2, axis=1)
    elif prop_bias_type == 'l2bnd':
        quality = 1
    elif prop_bias_type == 'point':
        quality = 1
    elif prop_bias_type == 'E2vf':
        quality = 1
    elif prop_bias_type == 'aniso':
        quality = 1

    quality = np.tile(quality, (feature_prop.shape[0], 1))
    return np.multiply(feature_prop, quality)

