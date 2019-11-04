import numpy as np
import scipy.linalg as la
from scipy.stats import ortho_group
import functools
import networkx as nx


def test_decentralized_orth(A, Q, B, neighbors, k, orth_itrs):
    n = A.shape[0]

    # Select weight to initialize to 1
    i_hat = 2
    w = [0] * n
    w[i_hat] = 1  # Set one node to

    # Initialize lists for keeping track of matrix and weight values
    S = [None] * n
    V = np.zeros((n, k))

    # Calculate number of iterations needed for orthonormalization
    num_itrs = orth_itrs

    for l in range(0, num_itrs):
        # Set up all S values to be K_i initially
        for i in range(0, n):
            # Change back to neighbors --> not working with neighbors
            V_i = np.sum([A[i, j] * Q[j, :] for j in range(0, n)], axis=0)
            V_i = np.sum([A[i, j] * Q[j, :] for j in np.unique(np.append(neighbors[i], i))], axis=0)

            # Reshape into row vector
            V_i = np.reshape(V_i, (1, V_i.shape[0]))
            K_i = np.transpose(V_i) @ V_i

            V[i, :] = V_i
            S[i] = K_i

        # Select weight to initialize to 1
        i_hat = 2
        w = [0] * n
        w[i_hat] = 1  # Set one node to

        # Run push sum algorithm to calculate K matrices at each node
        itrs = 100
        K_hat = push_sum(S, B, w, neighbors, n, itrs)

        # print(Q)
        for i in range(0, n):
            R = np.transpose(np.linalg.cholesky(K_hat[i]))

            # print(i, V[i] @ np.linalg.inv(R))
            Q[i, :] = V[i, :] @ np.linalg.inv(R)
        # print('RTR: \n', R.T @ R)
        # print('compare R: \n', V @ np.linalg.inv(R))
        # # print(Q)
        # print("estimated Q: \n", Q)

    return Q


def test_decentralized_orth_1_row_per_node(A, Q, B, neighbors, k, orth_itrs):
    n = A.shape[0]

    # Select weight to initialize to 1
    i_hat = 2
    w = [0] * n
    w[i_hat] = 1  # Set one node to

    # Initialize lists for keeping track of matrix and weight values
    S = [None] * n
    V = np.zeros((n, k))

    # Calculate number of iterations needed for orthonormalization
    num_itrs = orth_itrs

    for l in range(0, num_itrs):
        # Set up all S values to be K_i initially
        for i in range(0, n):
            # Change back to neighbors --> not working with neighbors
            V_i = np.sum([A[i, j] * Q[j, :] for j in range(0, n)], axis=0)
            V_i = np.sum([A[i, j] * Q[j, :] for j in np.unique(np.append(neighbors[i], i))], axis=0)

            # Reshape into row vector
            V_i = np.reshape(V_i, (1, V_i.shape[0]))
            K_i = np.transpose(V_i) @ V_i

            V[i, :] = V_i
            S[i] = K_i

        # Select weight to initialize to 1
        i_hat = 2
        w = [0] * n
        w[i_hat] = 1  # Set one node to

        # Run push sum algorithm to calculate K matrices at each node
        itrs = 100
        K_hat = push_sum(S, B, w, neighbors, n, itrs)

        # print(Q)
        for i in range(0, n):
            R = np.transpose(np.linalg.cholesky(K_hat[i]))

            # print(i, V[i] @ np.linalg.inv(R))
            Q[i, :] = V[i, :] @ np.linalg.inv(R)
        # print('RTR: \n', R.T @ R)
        # print('compare R: \n', V @ np.linalg.inv(R))
        # # print(Q)
        # print("estimated Q: \n", Q)

    return Q


def decentralized_orth_1_row_per_node(A, neighbors, k, orth_itrs):
    n = A.shape[0]
    Q = np.random.rand(n, k)
    B = np.random.rand(n, n)

    # Specify adjacencies between nodes
    B_mask = np.zeros_like(B)
    for idx, nbrs in enumerate(neighbors):
        B_mask[idx, idx] = 1
        B_mask[idx, nbrs] = 1
    B = np.multiply(B, B_mask)

    # Create a column stochastic matrix
    B = B/B.sum(axis=0)[None,:]
    # print(B)

    # Select weight to initialize to 1
    i_hat = 2
    w = [0] * n
    w[i_hat] = 1  # Set one node to

    # Initialize lists for keeping track of matrix and weight values
    S = [None] * n
    V = np.zeros((n,k))

    # Calculate number of iterations needed for orthonormalization
    num_itrs = orth_itrs

    for l in range(0, num_itrs):
        # Set up all S values to be K_i initially
        for i in range(0, n):
            # Change back to neighbors --> not working with neighbors
            V_i = np.sum([A[i, j] * Q[j,:] for j in range(0,n)], axis=0)
            V_i = np.sum([A[i, j] * Q[j,:] for j in np.unique(np.append(neighbors[i], i))], axis=0)

            # Reshape into row vector
            V_i = np.reshape(V_i, (1, V_i.shape[0]))
            K_i = np.transpose(V_i) @ V_i

            V[i,:] = V_i
            S[i] = K_i

        # Select weight to initialize to 1
        i_hat = 2
        w = [0] * n
        w[i_hat] = 1  # Set one node to

        # Run push sum algorithm to calculate K matrices at each node
        itrs = 100
        K_hat = push_sum(S, B, w, neighbors, n, itrs)

        # print(Q)
        for i in range(0, n):

            R = np.transpose(np.linalg.cholesky(K_hat[i]))

            # print(i, V[i] @ np.linalg.inv(R))
            Q[i, :] = V[i, :] @ np.linalg.inv(R)
        # print('RTR: \n', R.T @ R)
        # print('compare R: \n', V @ np.linalg.inv(R))
        # # print(Q)
        # print("estimated Q: \n", Q)

    return Q, R


def test_decentralized_orth(A, Q, B, rows_for_agent, neighbors, k, orth_itrs):
    num_agents = len(rows_for_agent)

    # Modify B matrix to take into account nodes_for_agent

    n = A.shape[0]

    # Select weight to initialize to 1
    i_hat = 2
    w = [0] * num_agents
    w[i_hat] = 1  # Set one node to

    # Initialize lists for keeping track of matrix and weight values
    S = [None] * num_agents
    V = np.zeros((n,k))

    # Calculate number of iterations needed for orthonormalization
    num_itrs = orth_itrs

    for l in range(0, num_itrs):
        # Set up all S values to be K_i initially
        for i in range(0, num_agents):
            rows = rows_for_agent[i]
            all_rows = np.concatenate([rows_for_agent[neighb] for neighb in neighbors[i]])
            all_rows = np.unique(np.append(all_rows, rows))

            K_i = np.zeros((k,k))

            for r in rows:
                V_r = np.sum([A[r, j] * Q[j,:] for j in all_rows], axis=0)

                # Reshape into row vector
                V_r = np.reshape(V_r, (1, V_r.shape[0]))

                V[r,:] = V_r

                K_i += np.transpose(V_r) @ V_r

            S[i] = K_i

        # Select weight to initialize to 1
        i_hat = 2
        w = [0] * num_agents
        w[i_hat] = 1  # Set one node to

        # Run push sum algorithm to calculate K matrices at each node
        itrs = 100
        K_hat = push_sum(S, B, w, neighbors, num_agents, itrs)
        R_hat = [None] * num_agents
        # print(Q)
        for i in range(0, num_agents):
            R = np.transpose(np.linalg.cholesky(K_hat[i]))
            R_hat[i] = R.diagonal()

            # print(i, V[i] @ np.linalg.inv(R))
            for r in rows_for_agent[i]:
                Q[r, :] = V[r, :] @ np.linalg.inv(R)
        # print('RTR: \n', R.T @ R)
        # print('compare R: \n', V @ np.linalg.inv(R))
        # # print(Q)
        # print("estimated Q: \n", Q)
    return Q, R_hat


def decentralized_orth(A, rows_for_agent, neighbors, k, orth_itrs):
    num_agents = len(rows_for_agent)

    # Modify B matrix to take into account nodes_for_agent

    n = A.shape[0]
    Q = np.random.rand(n, k)
    B = np.random.rand(len(neighbors), len(neighbors))

    # Specify adjacencies between nodes
    B_mask = np.zeros_like(B)
    for idx, nbrs in enumerate(neighbors):
        B_mask[idx, idx] = 1
        B_mask[idx, nbrs] = 1
    B = np.multiply(B, B_mask)

    # Create a column stochastic matrix
    B = B/B.sum(axis=0)[None,:]
    # print(B)

    # Select weight to initialize to 1
    i_hat = 2
    w = [0] * num_agents
    w[i_hat] = 1  # Set one node to

    # Initialize lists for keeping track of matrix and weight values
    S = [None] * num_agents
    V = np.zeros((n, k))

    # Calculate number of iterations needed for orthonormalization
    num_itrs = orth_itrs

    for l in range(0, num_itrs):
        # Set up all S values to be K_i initially
        for i in range(0, num_agents):
            rows = rows_for_agent[i]
            all_rows = np.concatenate([rows_for_agent[neighb] for neighb in neighbors[i]])
            all_rows = np.unique(np.append(all_rows, rows))

            K_i = np.zeros((k,k))

            for r in rows:
                V_r = np.sum([A[r, j] * Q[j,:] for j in all_rows], axis=0)

                # Reshape into row vector
                V_r = np.reshape(V_r, (1, V_r.shape[0]))

                V[r,:] = V_r

                K_i += np.transpose(V_r) @ V_r

            S[i] = K_i

        # Select weight to initialize to 1
        i_hat = 2
        w = [0] * num_agents
        w[i_hat] = 1  # Set one node to

        # Run push sum algorithm to calculate K matrices at each node
        itrs = 100
        K_hat = push_sum(S, B, w, neighbors, num_agents, itrs)
        R_hat = [None] * num_agents
        # print(Q)
        for i in range(0, num_agents):
            R = np.transpose(np.linalg.cholesky(K_hat[i]))
            R_hat[i] = R.diagonal()

            # print(i, V[i] @ np.linalg.inv(R))
            for r in rows_for_agent[i]:
                Q[r, :] = V[r, :] @ np.linalg.inv(R)
        # print('RTR: \n', R.T @ R)
        # print('compare R: \n', V @ np.linalg.inv(R))
        # # print(Q)
        # print("estimated Q: \n", Q)
    return Q, R_hat


def decentralized_orth_from_data(X, A, DA, rows_for_agent, neighbors, k, orth_itrs):
    num_agents = len(rows_for_agent)

    # Modify B matrix to take into account nodes_for_agent

    N = X.shape[0]
    Q = np.random.rand(N, k)
    B = np.random.rand(len(neighbors), len(neighbors))

    # Specify adjacencies between nodes
    B_mask = np.zeros_like(B)
    for idx, nbrs in enumerate(neighbors):
        B_mask[idx, idx] = 1
        B_mask[idx, nbrs] = 1
    B = np.multiply(B, B_mask)

    # Create a column stochastic matrix
    B = B/B.sum(axis=0)[None,:]
    # print(B)

    # Select weight to initialize to 1
    i_hat = 2
    w = [0] * num_agents
    w[i_hat] = 1  # Set one node to

    # Initialize lists for keeping track of matrix and weight values
    S = [None] * num_agents
    V = np.zeros((N, k))

    # Calculate number of iterations needed for orthonormalization
    num_itrs = orth_itrs
    old_Q = Q
    # Q = DA @ Q

    for l in range(0, num_itrs):
        # Set up all S values to be K_i initially
        for idx in range(0, np.shape(Q)[1]):
            Z_j = np.zeros_like(X)
            for j in range(0, np.shape(Q)[0]):
                # print(np.shape(X[j, :]), np.shape(DA[j, :]))
                Z_j[j, :] = Q[j, idx] * X[j, :]

            # Change to push sum
            Z_hat = (1 / np.shape(X)[1]) * np.sum(Z_j, axis=0)

            # Each agent can do this independently for its own data given Z
            for j in range(0, np.shape(Q)[0]):

                V[j, idx] = Z_hat.T @ X[j, :]
        # print(np.allclose(A@old_Q, V))
        for i in range(0, num_agents):
            rows = rows_for_agent[i]
            # all_rows = np.concatenate([rows_for_agent[neighb] for neighb in neighbors[i]])
            # all_rows = np.unique(np.append(all_rows, rows))
            K_i = np.zeros((k, k))

            for r in rows:
                V_r = V[r, :]

                # Reshape into row vector
                V_r = np.reshape(V_r, (1, V_r.shape[0]))

                K_i += np.transpose(V_r) @ V_r
            #
            # V_r = np.sum([V[j, :] for j in rows], axis=0)
            # V_r = np.reshape(V_r, (1, V_r.shape[0]))
            # # V[r,:] = V_r
            #
            # K_i += np.transpose(V_r) @ V_r

            S[i] = K_i
        # print(S)
        # Select weight to initialize to 1
        i_hat = 2
        w = [0] * num_agents
        w[i_hat] = 1  # Set one node to

        # Run push sum algorithm to calculate K matrices at each node
        itrs = 100
        K_hat = push_sum(S, B, w, neighbors, num_agents, itrs)
        R_hat = [None] * num_agents
        # print(Q)
        # print(K_hat)
        for i in range(0, num_agents):
            # print(np.allclose(K_hat[i], K_hat[i].T, atol=1e-2))
            R = np.transpose(np.linalg.cholesky(K_hat[i]))
            R_hat[i] = R.diagonal()

            # print(i, V[i] @ np.linalg.inv(R))
            for r in rows_for_agent[i]:
                Q[r, :] = V[r, :] @ np.linalg.inv(R)
        # print('RTR: \n', R.T @ R)
        # print('compare R: \n', V @ np.linalg.inv(R))
        # # print(Q)
        # print("estimated Q: \n", Q)
        # print("done")
        # Q = DA @ Q

    return Q, R_hat


def push_sum(S, B, w, neighbors, num_agents, num_itr):
    K_est = list(S)

    for idx in range(0, num_itr):
        new_S = list(S)
        new_w = list(w)

        for i in range(0, num_agents):
            neighbor_idxs = np.unique(np.append(neighbors[i], i)) # Include self in neighbor list
            S_i = np.sum([B[i, j]*S[j] for j in neighbor_idxs], axis=0)
            w_i = np.sum([B[i, j]*w[j] for j in neighbor_idxs])

            new_S[i] = S_i
            new_w[i] = w_i

        S = list(new_S)
        w = list(new_w)

    for i, s in enumerate(S):
        K_est[i] = s/w[i]

    return K_est


# Returns list of neighbors from symmetrix matrix
def get_neighbors(adjacency_matrix):
    return [np.where(adj == 1)[0] for adj in adjacency_matrix]


