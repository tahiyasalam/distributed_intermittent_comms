import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as lines
import scipy as sp

# Description: algorithm implementing Optimal Sensor Placement
# for State Reconstruction of Distributed Process Systems (Alonso, 2004)
# Author: Tahiya Salam
# May 2018


# function for colors on plots
# https://matplotlib.org/users/colormaps.html
def get_cmap(n, name):
    return plt.cm.get_cmap(name, n)

# EXAMPLE 1: Diffusion process


# eigenfunction provide basis for solution of diffusion equation
def eigenfunction(i, x):
    func = np.sqrt(2) * np.sin(np.pi * x * i)
    return func


# computation of actual eigenvalues for solution provided by Analysis of Transport Phenomenon - Deen (p. 161)
def eigenvalue(i, max_t):
    t = np.arange(0, max_t)
    vals = np.sqrt(2) / (i * np.pi) * (1 - np.power(-1, i)) * np.exp(-1*np.square(i*np.pi) * t)
    return vals


def function_approximation(eigenfunctions, eigenvalues, x, t, max_i):
    return sum([eigenvalues[i][t] * eigenfunctions[i][x] for i in range(0, max_i)])


def R_operator(vector, matrix):
    N = []
    for row in matrix:
        modified_sigma_1_arr = [vector[t_i] for t_i in row]
        N += [modified_sigma_1_arr]
    return N


def antidiagonal_operator(A):
    return np.diag(np.fliplr(np.matrix(A)))


def sorted_indices_operator(vector):
    return np.argsort(vector)


def summation_operator(sigma_i, ada):
    s_i = 0
    for idx in ada:
        s_i += sigma_i[idx]
    return s_i


def corresponding_sum(ada, S_k):
    return [summation_operator(sigma, ada) for sigma in S_k]


def conditional_sequence(n, m, idx, sigma, a, b, c):

    sigma_i = sigma[idx]

    sorted_sigma_i_indices = sorted_indices_operator(sigma_i)[::-1]

    T = []
    for r in range(0, n - m + 1):
        row = []
        for c0 in range(0, m):
            row += [r + c0]
        T += [row]

    N = R_operator(sorted_sigma_i_indices, T)
    l = n - m + 1
    M = []

    # print(N)
    # print(0, m-1)
    for r in range(0, m-1):
        for s in range(0, l):
            M_s_r = []
            for s_1 in range(s, l):
                M_s_r += [N[s][0:r+1] + N[s_1][r+1:m+1]]
            # print(M_s_r)

            set_of_N = [R_operator(sigma[i], M_s_r) for i in range(0, sigma.shape[0])]
            set_of_u = [antidiagonal_operator([row[r + 1:m + 1] for row in N_i]) for N_i in set_of_N]
            set_of_idx = [sorted_indices_operator(u_i) for u_i in set_of_u]

            bound_1 = [summation_operator(set_of_u[i], seq_i[0: m - r+1]) for (i, seq_i) in enumerate(set_of_idx)]
            bound_2 = [summation_operator(N_i[0][0:r+1], np.arange(0, r+1)) for N_i in set_of_N]
            bounds = [x + y for (x, y) in zip(bound_1, bound_2)]
            idx_inv = np.flipud(set_of_idx[idx])
            bound_idx_inv = summation_operator(set_of_u[idx], idx_inv[0: m - r+1]) + summation_operator(set_of_N[idx][0][0:r], np.arange(0, r))

            # print(bounds[idx])
            if bounds[idx] < a or b < bound_idx_inv or max(bounds) < c:
                continue
            else:
                # print(bounds[idx], a, b, bound_idx_inv, max(bounds), c)
                M += M_s_r
                continue

    return M


def optimal_arrangement(m, sigma, max_itr, epsilon):
    k = sigma.shape[0]  # total number of basis vectors
    n = sigma.shape[1]  # number of total sensors

    L_upper = [0] * k

    # find upper value by sorting the first sigma and picking the m highest values
    sorted_sigma_i_indices = sorted_indices_operator(sigma[0])[::-1][0:m]
    for i in range(0, k):
        L_upper[i] = summation_operator(sigma[i], sorted_sigma_i_indices)

    L_sup = [0] * (max_itr + 1)
    L = [0] * k

    l = 0

    # print(sigma)

    # for l in range(0, max_itr):
    while l < max_itr:
        L_lower = np.array(L_upper) - epsilon

        for i in range(0, k):
            # print(sigma, L_lower[i], L_upper[i], L_sup[l])
            Ada_i = conditional_sequence(n, m, i, sigma, L_lower[i], L_upper[i], L_sup[l])
            #  Ada_i: set of index vectors
            # print("Ada_i: ", Ada_i)

            Sigma = [corresponding_sum(n_i, sigma) for n_i in Ada_i]
            #  Sigma: set where each row is summation operator applied to one index vector applied to Sk
            # print("Sigma_i: ", Sigma)

            L[i] = max([min(Sigma_i) for Sigma_i in Sigma])
            # print("L[i]: ", L[i])

            if L[i] > L_sup[l]:
                # print("greater value found: ", L[i], L_sup[l], L_upper)
                p = np.argmax([min(Sigma_i) for Sigma_i in Sigma])
                ada = Ada_i[p]
                L_sup[l] = L[i]
                continue

            if any(L_sup[l] >= val for val in L_upper):
                j = np.argmax(L_sup[l] >= np.array(L_upper))
                # print('', j, L_upper, L_sup)
                if all(L_sup[l] >= val for val in L_upper):
                    # p = np.argmax(min([Sigma_i for Sigma_i in Sigma]))
                    # print('max for all')
                    return p, ada
                else:
                    # print("next l: ", l)
                    L_sup[l+1] = L_sup[l]
                    for i_0 in range(0, k):
                        if j != i_0:
                            L_upper[i_0] = L_lower[i_0]
                    l = l + 1
                    break


def null(A, eps=1e-12):
    u, s, vh = sp.linalg.svd(A)
    padding = max(0,np.shape(A)[1]-np.shape(s)[0])
    null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
    null_space = sp.compress(null_mask, vh, axis=0)
    return sp.transpose(null_space)


def state_estimation(sensor_locations, phi, v_m):
    m = len(sensor_locations)
    n, k = np.shape(phi)

    PT = np.matrix(np.tile([0] * m, (n, 1)))

    # Check to see this still works with example
    curr_idx = 0
    for s_i in sorted(sensor_locations):
        PT[s_i, curr_idx] = 1
        curr_idx += 1
    P = PT.T

    # n = np.shape(P)[1]
    # u, s, vh = np.linalg.svd(P)
    # # Figure out how to compute orthogonal complement
    # P_orth = u[:, n:]

    P_orth = np.matrix(np.zeros((n, n)))
    orth_complement = null(P)
    diff = n - np.shape(orth_complement)[1]
    P_orth[:, :-diff] = orth_complement

    Q = phi.T * PT
    QT = Q.T
    # print(np.shape(P_orth), np.shape(Q))

    estimate_mat = P_orth*phi*(Q*QT).I*Q

    return estimate_mat*v_m


# sensor_locations = [1, 2]
# phi = np.matrix([[0.5141, 0.04940], [0.5011, -0.8274], [0.6961, 0.5595]])
# state_estimation(sensor_locations, phi, [0.500, 0.555])