from optimal_placement_algorithm_2d import *
from decentralized_orth import *
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import scipy.linalg as la


def distance(p0, p1):
    return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x / length, sum_y / length


def get_centers(mesh_dictionary, region_dictionary):
    sensing_regions = list(set(region_dictionary.values()))
    print(sensing_regions)
    center = [None] * len(sensing_regions)
    # Mesh dictionary {idx: coordinate point}
    # Region dictionary {idx: region value}
    for i in range(0, len(sensing_regions)):
        array_location = [mesh_dictionary[k] for k, v in region_dictionary.items() if v is sensing_regions[i]]
        center[i] = centeroidnp(np.asarray(array_location))

    return center


def create_bipartite_graph(old_robot_region, new_robot_region, sensing_regions, region_dictionary, mesh_dictionary, init):
    B = nx.Graph()
    # Robot nodes
    robot_str = ['r' + str(num) for num in range(0, len(old_robot_region))]
    B.add_nodes_from(robot_str, bipartite=0)

    # Sensing region nodes
    B.add_nodes_from(sensing_regions, bipartite=1)

    center = [None] * len(sensing_regions)
    # Mesh dictionary {idx: coordinate point}
    # Region dictionary {idx: region value}
    for i in range(0, len(sensing_regions)):
        array_location = [mesh_dictionary[k] for k, v in region_dictionary.items() if v is sensing_regions[i]]
        center[i] = centeroidnp(np.asarray(array_location))

    edge_weights = [None] * (len(new_robot_region) * len(sensing_regions))
    for i in range(0, len(new_robot_region)):
        for j in range(0, len(sensing_regions)):
            if sensing_regions[j] in new_robot_region:
                dist = distance(center[old_robot_region[i]], center[sensing_regions[j]])
            else:
                dist = 60000
            edge_weights[i * len(sensing_regions) + j] = (robot_str[i], sensing_regions[j], -1 * dist)

    B.add_weighted_edges_from(edge_weights)
    connected = nx.is_connected(B)
    bottom_nodes, top_nodes = bipartite.sets(B)

    unsorted_matching = nx.max_weight_matching(B, maxcardinality=True)

    matching = {}
    for k, v in unsorted_matching:
        if type(k) == str:
            matching[k] = v
        else:
            matching[v] = k

    matching = [matching[k] for k in sorted(matching)]
    return matching


def calculate_POD_basis_distributed(node_solution, DA, neighbors, rows_for_agent, num_k, num_orth_itrs):
    n_elmts = np.shape(node_solution)[0]
    n_snapshots = np.shape(node_solution)[2]
    n_sensors = np.shape(node_solution)[1]
    sum_UUT = np.matrix(np.zeros((n_elmts, n_elmts)))

    for t in range(0, n_snapshots):
        U = node_solution[:, 0, t]
        U = np.reshape(U, (n_elmts, 1))
        for s in range(1, n_sensors):
            U = np.concatenate(U, np.reshape(node_solution[:,s,t], (n_elmts, 1)), axis = 0)

        UUT = U @ U.T
        sum_UUT = sum_UUT + UUT

    R = 1/n_snapshots * sum_UUT
    # DA = np.identity(np.shape(DA)[0])
    K = R @ DA
    # true_Q = la.orth(K)
    # # print(np.shape(node_solution))
    # oldQ, R_hat = decentralized_orth(K, rows_for_agent, neighbors, num_k, num_orth_itrs)
    # print(oldQ)
    Q, R_hat = decentralized_orth_from_data(node_solution.reshape(np.shape(node_solution)[0], np.shape(node_solution)[2]), K, DA, rows_for_agent, neighbors, num_k, num_orth_itrs)
    # print(Q)
    # dist = np.linalg.norm((Q @ Q.T - true_Q[:,0:num_k] @ true_Q[:,0:num_k].T))
    # print('true and new decentralized: ', dist)
    # dist = np.linalg.norm((oldQ @ oldQ.T - true_Q[:,0:num_k] @ true_Q[:,0:num_k].T))
    # print('true and old decentralized: ', dist)
    # dist = np.linalg.norm((Q @ Q.T - oldQ @ oldQ.T))
    # print('old and new decentralized: ', dist)


    # print(np.allclose(Q, oldQ, atol=1e-3))

    return create_basis_distributed(Q, R_hat)


def create_basis_distributed(V, D_hats):
    """"
        Creates basis functions for POD using solution from MATLAB
        The number of basis functions is determined by reaching a minimum energy captured

        Input: MATLAB file describing the mesh of the region used in simulation

        Output: truncated basis functions for POD

    """

    # print(type(DA[0][:] == DA.T[:][0]))
    D = D_hats[0]
    cum_sum = 0
    total_sum = sum(D).real
    m = 0  # Calculate number of POD bases that will be needed

    for i in range(0, len(D)):
        cum_sum += D[i].real
        percentage_energy = cum_sum/total_sum
        if percentage_energy < 0.99:
            m = i
        else:
            m = i
            break
    # Create new matrix based on POD values at each of the sub domains given by value in regionDictionary
    # phi is (n, m) for n = number of elements in mesh, m = number of elements in POD
    phi = V[:, 0:m+1]
    # print(phi)
    return phi.real


# Calculate time dependent coefficients for POD basis for specific time instant
def distributed_coefficient(sensor_locations, set_of_phi, set_of_da, set_of_y, neighbors):
    m = len(sensor_locations)
    n, k = np.shape(set_of_phi[0])

    pi = np.matrix(np.zeros((k, k)))
    sum_field = np.matrix(np.zeros((k, 1)))

    Pi = [None] * m
    Sum_field = [None] * m
    # Check to see this still works with example
    for idx, s_i in enumerate(sorted(sensor_locations)):
        pi = set_of_phi[s_i].T @ set_of_da[s_i] @ set_of_phi[s_i]
        sum_field = set_of_phi[s_i].T @ set_of_da[s_i] @ set_of_y[s_i]

        Pi[idx] = pi
        Sum_field[idx] = sum_field

    B = np.random.rand(len(neighbors), len(neighbors))

    # Specify adjacencies between nodes
    B_mask = np.zeros_like(B)
    for idx, nbrs in enumerate(neighbors):
        B_mask[idx, idx] = 1
        B_mask[idx, nbrs] = 1
    B = np.multiply(B, B_mask)

    # Create a column stochastic matrix
    B = B/B.sum(axis=0)[None,:]

    # Select weight to initialize to 1
    i_hat = 2
    w = [0] * m
    w[i_hat] = 1  # Set one node to

    itrs = 100

    Pi_hat = push_sum(Pi, B, w, neighbors, m, itrs)
    Sum_field_hat = push_sum(Sum_field, B, w, neighbors, m, itrs)

    C_hat = [None] * m

    # Check to see this still works with example
    for idx, _ in enumerate(sorted(sensor_locations)):
        C_hat[idx] = Pi_hat[idx].I @ Sum_field_hat[idx]

    return C_hat


def state_estimation_2D_distributed(sensor_locations, phi, set_of_phi, set_of_da, set_of_y):
    C_hat = distributed_coefficient(sensor_locations, set_of_phi, set_of_da, set_of_y)

    # print(np.shape(c_hat))
    return [phi @ c_hat for c_hat in C_hat]


def get_regions_for_agent(region_dictionary, ada):
    regions_for_agent = [[ada_val] for ada_val in ada]
    available_regions = list(set(region_dictionary.values()))
    available_regions = np.array([region for region in available_regions if region not in ada])

    split_regions = np.array_split(available_regions, len(ada))

    for idx in range(0, len(ada)):
        regions_for_agent[idx] = regions_for_agent[idx] + list(split_regions[idx])

    return regions_for_agent


def adaptation_scheme_distributed(region_dictionary, mesh_dictionary, node_solution, DA, t0, t_resample, sensors, neighbors, rows_for_agent, num_k, num_orth_itrs, centers, dt):
    node_solution = np.reshape(node_solution, (np.shape(node_solution)[0], 1, np.shape(node_solution)[1]))
    total_time = np.shape(node_solution)[2]
    # Keep start and end times of ada values
    ada_vals = {}
    loc_vals = {}
    path_vals = {}

    # Calculate POD basis for initial set of data for time [0, t0)
    idx = np.round(np.linspace(0, np.shape(node_solution)[2] - 1, t0)).astype(int)
    truncated_data = node_solution[:, :, idx]
    # noise = np.random.normal(128, 10, truncated_data.shape)
    # truncated_data = truncated_data + noise
    # truncated_data = node_solution[:, :, 0:t0]

    # TODO: Change calculation of POD basis to be distributed
    phi = calculate_POD_basis_distributed(truncated_data, DA, neighbors, rows_for_agent, num_k, num_orth_itrs)
    # phi = calculate_POD_basis(truncated_data, DA)
    sigma, set_of_phi, set_of_da = create_sigma(region_dictionary, phi, DA)

    sensing_regions = list(set(region_dictionary.values()))
    # To start, information from whole field are used
    # ada_vals[(0, t0-1)] = sensing_regions

    # Estimate optimal placement information from POD basis
    p, ada = opa.optimal_arrangement(sensors, np.asarray(sigma), 100000000, 10)
    print(ada)
    old_loc = [int(len(sensing_regions)/2)] * len(ada)

    loc = create_bipartite_graph(old_loc, ada, sensing_regions, region_dictionary, mesh_dictionary, True)
    paths = plan_path(old_loc, loc, centers,  (600, 600), (100, 100), 2, dt)
    # loc_vals[(0, t0 - 1)] = loc
    # path_vals[(0, t0 - 1)] = paths

    old_loc = loc

    regions_for_agent = get_regions_for_agent(region_dictionary, ada)
    rows_for_agent = [sorted([k for (k, v) in region_dictionary.items() if v in region]) for region in regions_for_agent]
    # print(ada, rows_for_agent)

    # sample_start_time = t0
    # sample_end_time = t0 + t_resample
    # new_ada = ada
    # new_data = truncated_data

    sample_start_time = 0
    sample_end_time = t_resample
    new_ada = ada
    #new_data = truncated_data
    new_data = np.array([], dtype=np.float64).reshape((np.shape(truncated_data)[0], np.shape(truncated_data)[1], 0))

    while sample_start_time < total_time:
        ada_vals[(sample_start_time, min(sample_end_time, total_time) - 1)] = new_ada
        loc_vals[(sample_start_time, min(sample_end_time, total_time) - 1)] = loc
        path_vals[(sample_start_time, min(sample_end_time, total_time) - 1)] = paths


        # Estimate field from sensor arrangement [t0, t1)
        # Get field measurements from time t0 to t1, apply estimation technique using selected sensors

        estimated_vals = np.array([], dtype=np.float64).reshape(np.shape(node_solution)[0], 0)
        for time in range(sample_start_time, min(sample_end_time, total_time)):
            y = node_solution[:, :, time]   # Get data from original data set
            set_of_y = y_set(region_dictionary, new_ada, y)

            # TODO: Change state estimation to be distributed
            # Y_hat = state_estimation_2D_distributed(new_ada, phi, set_of_phi, set_of_da, set_of_y)  # Estimated data -- add noise?
            # y_hat = np.mean(np.array(Y_hat), axis=0)

            y_hat = state_estimation_2D(new_ada, phi, set_of_phi, set_of_da, set_of_y)  # Estimated data -- add noise?

            # Keep sensor values as measured
            idxs = sorted([k for (k, v) in region_dictionary.items() if v in new_ada])
            y_hat[idxs] = y[idxs]

            estimated_vals = np.concatenate((estimated_vals, y_hat), axis=1)

        # Update POD with information from estimated field
        # Add values to ORIGINAL data matrix -- can also add to POD estimated to get "lower order" estimates of field
        # print(np.shape(new_data), np.shape(estimated_vals[:, np.newaxis, :]))
        estimated_vals = estimated_vals[:, np.newaxis, :]
        new_data = np.concatenate((np.asfarray(new_data), np.asfarray(estimated_vals)), axis=2)

        # TODO: Change calculation of POD basis to be distributed
        phi = calculate_POD_basis_distributed(new_data, DA, neighbors, rows_for_agent, num_k, num_orth_itrs)
        # phi = calculate_POD_basis(new_data, DA)

        new_sigma, set_of_phi, set_of_da = create_sigma(region_dictionary, phi, DA)

        # Calculate new optimal arrangement of sensors
        p, new_ada = opa.optimal_arrangement(sensors, np.asarray(new_sigma), 100000000, 10)
        loc = create_bipartite_graph(old_loc, new_ada, sensing_regions, region_dictionary, mesh_dictionary, False)

        paths = plan_path(old_loc, loc, centers, (600, 600), (100, 100), 2, dt)

        old_loc = loc
        print(new_ada, loc)


        # Next iteration
        sample_start_time = sample_end_time
        sample_end_time = sample_start_time + t_resample

    return new_data, ada_vals, loc_vals, path_vals


def adaptation_scheme_2(region_dictionary, mesh_dictionary, node_solution, DA, t0, t_resample, sensors, neighbors, rows_for_agent, num_k, num_orth_itrs, centers, dt):
    node_solution = np.reshape(node_solution, (np.shape(node_solution)[0], 1, np.shape(node_solution)[1]))
    total_time = np.shape(node_solution)[2]
    # Keep start and end times of ada values
    ada_vals = {}
    loc_vals = {}
    path_vals = {}

    # Calculate POD basis for initial set of data for time [0, t0)
    idx = np.round(np.linspace(0, np.shape(node_solution)[2] - 1, t0)).astype(int)
    truncated_data = node_solution[:, :, idx]
    # noise = np.random.normal(128, 10, truncated_data.shape)
    # truncated_data = truncated_data + noise
    # truncated_data = node_solution[:, :, 0:t0]

    # TODO: Change calculation of POD basis to be distributed
    # phi = calculate_POD_basis_distributed(truncated_data, DA, neighbors, rows_for_agent, num_k, num_orth_itrs)
    phi = calculate_POD_basis(truncated_data, DA)
    sigma, set_of_phi, set_of_da = create_sigma(region_dictionary, phi, DA)

    sensing_regions = list(set(region_dictionary.values()))
    # To start, information from whole field are used
    # ada_vals[(0, t0-1)] = sensing_regions

    # Estimate optimal placement information from POD basis
    p, ada = opa.optimal_arrangement(sensors, np.asarray(sigma), 100000000, 10)
    print(ada)
    old_loc = [int(len(sensing_regions)/2)] * len(ada)

    loc = create_bipartite_graph(old_loc, ada, sensing_regions, region_dictionary, mesh_dictionary, True)
    paths = plan_path(old_loc, loc, centers,  (600,600), (100, 100), 2, dt)
    # loc_vals[(0, t0 - 1)] = loc
    # path_vals[(0, t0 - 1)] = paths

    old_loc = loc

    regions_for_agent = get_regions_for_agent(region_dictionary, ada)
    rows_for_agent = [sorted([k for (k, v) in region_dictionary.items() if v in region]) for region in regions_for_agent]
    # print(ada, rows_for_agent)

    # sample_start_time = t0
    # sample_end_time = t0 + t_resample
    # new_ada = ada
    # new_data = truncated_data

    sample_start_time = 0
    sample_end_time = t_resample
    new_ada = ada
    #new_data = truncated_data
    new_data = np.array([], dtype=np.float64).reshape((np.shape(truncated_data)[0], np.shape(truncated_data)[1], 0))

    while sample_start_time < total_time:
        ada_vals[(sample_start_time, min(sample_end_time, total_time) - 1)] = new_ada
        loc_vals[(sample_start_time, min(sample_end_time, total_time) - 1)] = loc
        path_vals[(sample_start_time, min(sample_end_time, total_time) - 1)] = paths


        # Estimate field from sensor arrangement [t0, t1)
        # Get field measurements from time t0 to t1, apply estimation technique using selected sensors

        estimated_vals = np.array([], dtype=np.float64).reshape(np.shape(node_solution)[0], 0)
        for time in range(sample_start_time, min(sample_end_time, total_time)):
            y = node_solution[:, :, time]   # Get data from original data set
            set_of_y = y_set(region_dictionary, new_ada, y)

            # TODO: Change state estimation to be distributed
            # Y_hat = state_estimation_2D_distributed(new_ada, phi, set_of_phi, set_of_da, set_of_y)  # Estimated data -- add noise?
            # y_hat = np.mean(np.array(Y_hat), axis=0)

            y_hat = state_estimation_2D(new_ada, phi, set_of_phi, set_of_da, set_of_y)  # Estimated data -- add noise?

            # Keep sensor values as measured
            idxs = sorted([k for (k, v) in region_dictionary.items() if v in new_ada])
            y_hat[idxs] = y[idxs]

            estimated_vals = np.concatenate((estimated_vals, y_hat), axis=1)

        # Update POD with information from estimated field
        # Add values to ORIGINAL data matrix -- can also add to POD estimated to get "lower order" estimates of field
        # print(np.shape(new_data), np.shape(estimated_vals[:, np.newaxis, :]))
        estimated_vals = estimated_vals[:, np.newaxis, :]
        new_data = np.concatenate((np.asfarray(new_data), np.asfarray(estimated_vals)), axis=2)

        # TODO: Change calculation of POD basis to be distributed
        # phi = calculate_POD_basis_distributed(new_data, DA, neighbors, rows_for_agent, num_k, num_orth_itrs)
        phi = calculate_POD_basis(new_data, DA)

        new_sigma, set_of_phi, set_of_da = create_sigma(region_dictionary, phi, DA)

        # Calculate new optimal arrangement of sensors
        p, new_ada = opa.optimal_arrangement(sensors, np.asarray(new_sigma), 100000000, 10)
        loc = create_bipartite_graph(old_loc, new_ada, sensing_regions, region_dictionary, mesh_dictionary, False)

        paths = plan_path(old_loc, loc, centers, (600, 600), (100, 100), 2, dt)

        old_loc = loc
        print(new_ada, loc)


        # Next iteration
        sample_start_time = sample_end_time
        sample_end_time = sample_start_time + t_resample

    return new_data, ada_vals, loc_vals, path_vals


def adaptation_scheme_2_no_mesh(region_dictionary, mesh_dictionary, node_solution, t0, t_resample, sensors, neighbors, rows_for_agent, num_k, num_orth_itrs, centers, dt, itr_num, epsilon):
    print(np.shape(node_solution))
    node_solution = np.reshape(node_solution, (np.shape(node_solution)[0], 1, np.shape(node_solution)[2]))
    total_time = np.shape(node_solution)[2]
    # Keep start and end times of ada values
    ada_vals = {}
    # loc_vals = {}
    # path_vals = {}

    # Calculate POD basis for initial set of data for time [0, t0)
    idx = np.round(np.linspace(0, np.shape(node_solution)[2] - 1, t0)).astype(int)
    print(idx)
    truncated_data = node_solution[:, :, idx]
    # noise = np.random.normal(128, 10, truncated_data.shape)
    # truncated_data = truncated_data + noise
    # truncated_data = node_solution[:, :, 0:t0]

    # TODO: Change calculation of POD basis to be distributed
    # phi = calculate_POD_basis_distributed(truncated_data, DA, neighbors, rows_for_agent, num_k, num_orth_itrs)
    phi = calculate_POD_basis(truncated_data, None)
    sigma, set_of_phi = create_sigma_no_mesh(region_dictionary, phi)

    sensing_regions = list(set(region_dictionary.values()))
    # To start, information from whole field are used
    # ada_vals[(0, t0-1)] = sensing_regions

    # Estimate optimal placement information from POD basis
    p, ada = opa.optimal_arrangement(sensors, np.asarray(sigma), itr_num, epsilon)
    print(ada)
    # old_loc = [int(len(sensing_regions)/2)] * len(ada)

    # loc = create_bipartite_graph(old_loc, ada, sensing_regions, region_dictionary, mesh_dictionary, True)
    # dim_plot = (11.5125, 5.7525)
    # dim_workspace = (11.5125, 5.7525)
    # paths = plan_path(old_loc, loc, centers,  dim_plot, dim_workspace, 2, dt)
    # path = None
    # loc_vals[(0, t0 - 1)] = loc
    # path_vals[(0, t0 - 1)] = paths

    # old_loc = loc

    regions_for_agent = get_regions_for_agent(region_dictionary, ada)
    rows_for_agent = [sorted([k for (k, v) in region_dictionary.items() if v in region]) for region in regions_for_agent]
    # print(ada, rows_for_agent)

    # sample_start_time = t0
    # sample_end_time = t0 + t_resample
    # new_ada = ada
    # new_data = truncated_data

    sample_start_time = 0
    sample_end_time = t_resample
    new_ada = ada
    #new_data = truncated_data
    new_data = np.array([], dtype=np.int8).reshape((np.shape(truncated_data)[0], np.shape(truncated_data)[1], 0))

    while sample_start_time < total_time:
        ada_vals[(sample_start_time, min(sample_end_time, total_time) - 1)] = new_ada
        # loc_vals[(sample_start_time, min(sample_end_time, total_time) - 1)] = loc
        # path_vals[(sample_start_time, min(sample_end_time, total_time) - 1)] = paths


        # Estimate field from sensor arrangement [t0, t1)
        # Get field measurements from time t0 to t1, apply estimation technique using selected sensors

        estimated_vals = np.array([], dtype=np.int8).reshape(np.shape(node_solution)[0], 0)
        for time in range(sample_start_time, min(sample_end_time, total_time)):
            y = node_solution[:, :, time]   # Get data from original data set
            set_of_y = y_set(region_dictionary, new_ada, y)

            # TODO: Change state estimation to be distributed
            # Y_hat = state_estimation_2D_distributed(new_ada, phi, set_of_phi, set_of_da, set_of_y)  # Estimated data -- add noise?
            # y_hat = np.mean(np.array(Y_hat), axis=0)

            y_hat = state_estimation_2D_no_mesh(new_ada, phi, set_of_phi, set_of_y)  # Estimated data -- add noise?

            # Keep sensor values as measured
            idxs = sorted([k for (k, v) in region_dictionary.items() if v in new_ada])
            y_hat[idxs] = y[idxs]

            estimated_vals = np.concatenate((estimated_vals, y_hat), axis=1)

        # Update POD with information from estimated field
        # Add values to ORIGINAL data matrix -- can also add to POD estimated to get "lower order" estimates of field
        # print(np.shape(new_data), np.shape(estimated_vals[:, np.newaxis, :]))
        estimated_vals = estimated_vals[:, np.newaxis, :]
        new_data = np.concatenate((np.asfarray(new_data), np.asfarray(estimated_vals)), axis=2)

        # TODO: Change calculation of POD basis to be distributed
        # phi = calculate_POD_basis_distributed(new_data, DA, neighbors, rows_for_agent, num_k, num_orth_itrs)
        phi = calculate_POD_basis(new_data, None)

        new_sigma, set_of_phi = create_sigma_no_mesh(region_dictionary, phi)

        # Calculate new optimal arrangement of sensors
        p, new_ada = opa.optimal_arrangement(sensors, np.asarray(new_sigma), itr_num, epsilon)
        # loc = create_bipartite_graph(old_loc, new_ada, sensing_regions, region_dictionary, mesh_dictionary, False)
        #
        # paths = plan_path(old_loc, loc, centers, dim_plot, dim_workspace, 2, dt)
        #
        # old_loc = loc
        # print(new_ada, loc)


        # Next iteration
        sample_start_time = sample_end_time
        sample_end_time = sample_start_time + t_resample

    return new_data, ada_vals, None, None
