import numpy as np
import scipy.io as sio
import scipy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import transforms

import matplotlib.patches as mpatches
import matplotlib.lines as lines
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist as AA
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import optimal_placement_algorithm as opa
from plotting_functions import *

def distance(p0, p1):
    return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def discretize_fem_mesh(shape, matlab_fem_mesh, size, indices=None):
    """"
        Creates dictionary that associates each candidate location to a subdomain

        Input: desired shape and size (currently only supports square)
                MATLAB file describing the mesh of the region used in simulation

        Output: dictionary with locations and numbered regions

    """
    mat_contents = sio.loadmat(matlab_fem_mesh)
    # Make dictionary out of mesh nodes and nodal solution
    meshNodes = mat_contents['MeshNodes']

    if indices is not None:
        meshNodes = meshNodes[:,indices]
        # meshNodes = np.squeeze(meshNodes, axis=1) # Eliminate new dimension creates

    meshDictionary = {}

    # Create dictionary from mesh nodes where key is location and value is number
    for elmt in range(0, len(meshNodes[0])):
        meshDictionary[(meshNodes[0][elmt], meshNodes[1][elmt])] = elmt

    locations = meshDictionary.keys()
    x_elements = [x for (x,y) in locations]
    y_elements = [y for (x,y) in locations]

    # Create a dictionary assigning each node of the mesh to a region
    regionDictionary = {}

    if shape == "square":
        x_range = np.linspace(min(x_elements), max(x_elements), size+1)
        y_range = np.linspace(min(y_elements), max(y_elements), size+1)

        region_num = 0
        for i in range(0, len(x_range)-1):
            for j in range(0, len(y_range)-1):
                for location in locations:
                    if location not in regionDictionary:
                        if x_range[i] <= location[0] and y_range[j] <= location[1]:  # Check to see if within bounds
                            # Assign conditions to the end of the segments
                            if location[0] < x_range[i+1] or (i+1 == len(x_range)-1 and location[0] <= x_range[i+1]):
                                if location[1] < y_range[j+1] or (j+1 == len(y_range)-1 and location[1] <= y_range[j+1]):
                                    regionDictionary[meshDictionary[location]] = region_num
                region_num += 1

    meshDictionary = {y: x for x, y in meshDictionary.items()}
    # Returns dictionary where each location is now associated with a number for its region
    return meshDictionary, regionDictionary


def discretize_mesh_plot(shape, size):
    import itertools

    coords = list(itertools.product(np.linspace(0, 599), np.linspace(0, 599)))
    mesh0, mesh1 = zip(*coords)
    meshNodes = [mesh0, mesh1]

    meshDictionary = {}

    # Create dictionary from mesh nodes where key is location and value is number
    for elmt in range(0, len(meshNodes[0])):
        meshDictionary[(meshNodes[0][elmt], meshNodes[1][elmt])] = elmt

    locations = meshDictionary.keys()
    x_elements = [x for (x,y) in locations]
    y_elements = [y for (x,y) in locations]

    # Create a dictionary assigning each node of the mesh to a region
    regionDictionary = {}

    if shape == "square":
        x_range = np.linspace(min(x_elements), max(x_elements), size+1)
        y_range = np.linspace(min(y_elements), max(y_elements), size+1)

        region_num = 0
        for i in range(0, len(x_range)-1):
            for j in range(0, len(y_range)-1):
                for location in locations:
                    if location not in regionDictionary:
                        if x_range[i] <= location[0] and y_range[j] <= location[1]:  # Check to see if within bounds
                            # Assign conditions to the end of the segments
                            if location[0] < x_range[i+1] or (i+1 == len(x_range)-1 and location[0] <= x_range[i+1]):
                                if location[1] < y_range[j+1] or (j+1 == len(y_range)-1 and location[1] <= y_range[j+1]):
                                    regionDictionary[meshDictionary[location]] = region_num
                region_num += 1

    meshDictionary = {y: x for x, y in meshDictionary.items()}
    # Returns dictionary where each location is now associated with a number for its region
    return meshDictionary, regionDictionary


def phi_j(phi, j, regionDictionary):
    # Get indices of all elements in regionDictionary in region j
    idxs = sorted([k for (k, v) in regionDictionary.items() if v == j])

    # Extract rows of phi that are within region j
    phi_j = phi[idxs, :]
    return phi_j


def DA_j(DA, j, regionDictionary):
    # Get indices of all elements in regionDictionary in region j
    idxs = sorted([k for (k, v) in regionDictionary.items() if v == j])

    # Extract rows and columns of DA that are within region j
    DA_j = DA[idxs, :]
    DA_j = DA_j[:, idxs]

    if isinstance(DA_j, sp.sparse.csc.csc_matrix):
        return np.array(DA_j.todense())
    else:
        return np.array(DA_j)


def phi_set(phi, regionDictionary):
    # Create new phi and DA matrices for each region
    phi_set = {}
    for i in list(set(regionDictionary.values())):
        phi_set[i] = phi_j(phi, i, regionDictionary)
    return phi_set


def DA_set(DA, regionDictionary):
    DA_set = {}
    for i in list(set(regionDictionary.values())):
        DA_set[i] = DA_j(DA, i, regionDictionary)
    return DA_set


def create_basis(V, D):
    """"
        Creates basis functions for POD using solution from MATLAB
        The number of basis functions is determined by reaching a minimum energy captured

        Input: MATLAB file describing the mesh of the region used in simulation

        Output: truncated basis functions for POD

    """

    # print(type(DA[0][:] == DA.T[:][0]))

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


def create_sigma(region_dictionary, phi, DA):
    """"
        Creates basis functions for POD using solution from MATLAB
        The number of basis functions is determined by reaching a minimum energy captured

        Input: MATLAB file describing the mesh of the region used in simulation

        Output: truncated basis functions for POD

    """
    m = np.shape(phi)[1]

    # Create new phi and DA matrices for each region
    set_of_phi = phi_set(phi, region_dictionary)
    set_of_DA = DA_set(DA, region_dictionary)

    # Tolerance may need to be adjusted depending on the data (truncation)
    # print(np.allclose(W.dot(V).dot(W.H), set_of_DA[0].real, atol=1e-19))
    # print(np.allclose(W * V * W.T, set_of_DA[0].real, atol=1e-19))

    # Construct Z elements
    set_of_Z = {}
    for i in list(set(set_of_DA.keys())):
        # Diagonalize each DA to get Z matrix
        W, V = ldl_decomp(set_of_DA[i].real)
        # Calculate each Z
        set_of_Z[i] = np.sqrt(V) * W.T * set_of_phi[i]

    # Construct sigma elements
    sigma = []

    for i in range(0, m):
        sigma_i = [np.asarray(set_of_Z[idx][:,i].T * set_of_Z[idx][:,i]).tolist()[0][0] for idx in sorted(list(set_of_Z.keys()))]
        sigma.append(sigma_i)
    print('sigma dim: ', m, len(sigma), len(sigma[0]))
    return sigma, set_of_phi, set_of_DA


def create_sigma_no_mesh(region_dictionary, phi):
    """"
        Creates basis functions for POD using solution from MATLAB
        The number of basis functions is determined by reaching a minimum energy captured

        Input: MATLAB file describing the mesh of the region used in simulation

        Output: truncated basis functions for POD

    """
    m = np.shape(phi)[1]

    # Create new phi and DA matrices for each region
    set_of_phi = phi_set(phi, region_dictionary)
    print(len(set_of_phi))

    # Tolerance may need to be adjusted depending on the data (truncation)
    # print(np.allclose(W.dot(V).dot(W.H), set_of_DA[0].real, atol=1e-19))
    # print(np.allclose(W * V * W.T, set_of_DA[0].real, atol=1e-19))

    # Construct Z elements
    set_of_Z = {}
    for i in list(set(set_of_phi.keys())):
        # Diagonalize each DA to get Z matrix
        # Calculate each Z
        set_of_Z[i] = set_of_phi[i]

    # Construct sigma elements
    sigma = []

    for i in range(0, m):
        sigma_i = [np.asarray(set_of_Z[idx][:,i].T * set_of_Z[idx][:,i])[0] for idx in sorted(list(set_of_Z.keys()))]
        sigma.append(sigma_i)
    print('sigma dim: ', m, len(sigma), len(sigma[0]))
    return sigma, set_of_phi


def ldl_decomp(A):
    A = np.matrix(A)
    if not check_symmetry(A):
        print("A must be Hermitian!")
        return None, None
    else:
        S = np.diag(np.diag(A))
        Sinv = np.diag(1/np.diag(A))
        D = np.matrix(S.dot(S))
        Lch = np.linalg.cholesky(A)
        L = np.matrix(Lch.dot(Sinv))
        return L, D


def check_symmetry(matrix):
    mat = np.asarray(matrix)
    mat_T = mat.T
    return np.allclose(mat, mat_T.T, atol=1e-40)


def test_discretize_fem_mesh(pathname_mesh):
    mesh_dictionary, region_dictionary = discretize_fem_mesh('square', pathname_mesh, 5)
    plot_discretization(mesh_dictionary, region_dictionary)


def plan_path(old_robot_locations, new_robot_locations, centers, dim_plot, dim_workspace, boat_speed, dt):
    if old_robot_locations == new_robot_locations:
        return []
    else:
        num_bots = len(old_robot_locations)
        paths = [None] * num_bots

        plot_x_lim, plot_y_lim = dim_plot
        workspace_x_lim, workspace_y_lim = dim_workspace

        pixels_per_second = (boat_speed * plot_x_lim) / workspace_x_lim
        # print(pixels_per_second)
        Kp_rho = 1
        Kp_alpha = 15
        Kp_beta = -3

        dist_threshold = 1
        for i in range(0, num_bots):
            if old_robot_locations[i] is not new_robot_locations[i]:
                x_traj = []
                y_traj = []
                x_pos = centers[old_robot_locations[i]][0]
                y_pos = centers[old_robot_locations[i]][1]
                # print(x_pos, y_pos)
                theta = 0
                theta_goal = 0
                while distance((x_pos, y_pos), centers[new_robot_locations[i]]) > dist_threshold:
                    x_traj.append(x_pos)
                    y_traj.append(y_pos)

                    x_diff = centers[new_robot_locations[i]][0] - x_pos
                    y_diff = centers[new_robot_locations[i]][1] - y_pos

                    alpha = (np.arctan2(y_diff, x_diff) - theta + np.pi) % (2 * np.pi) - np.pi
                    beta = (theta_goal - theta - alpha + np.pi) % (2 * np.pi) - np.pi

                    v = Kp_rho * np.sqrt(x_diff**2 + y_diff**2)
                    w = Kp_alpha * alpha + Kp_beta * beta

                    if alpha > np.pi / 2 or alpha < -np.pi / 2:
                        v = -v

                    theta = theta + w * dt

                    x_pos = x_pos + v * np.cos(theta) * dt
                    y_pos = y_pos + v * np.sin(theta) * dt
                paths[i] = list(zip(x_traj, y_traj))

    return paths


def state_estimation_2D(sensor_locations, phi, set_of_phi, set_of_da, set_of_y):
    c_hat = coefficient(sensor_locations, set_of_phi, set_of_da, set_of_y)

    # print(np.shape(c_hat))
    return phi @ c_hat


def state_estimation_2D_no_mesh(sensor_locations, phi, set_of_phi, set_of_y):
    c_hat = coefficient_no_mesh(sensor_locations, set_of_phi,set_of_y)

    # print(np.shape(c_hat))
    return phi @ c_hat

def test_create_basis(V, D, DA, pathname_mesh, sqrt_num_boxes, sensors):
    mesh_dictionary, region_dictionary = discretize_fem_mesh('square', pathname_mesh, sqrt_num_boxes)
    phi = create_basis(V, D)

    sigma, set_of_phi, set_of_da = create_sigma(region_dictionary, phi, DA)

    # TODO: check why this is sensitive to initial epsilon value
    p, ada = opa.optimal_arrangement(sensors, np.asarray(sigma), 100000, 10)
    print(p, ada)

    return phi, ada, set_of_phi, set_of_da
    # plot_sensing_regions(mesh_dictionary, region_dictionary, ada)


def test_create_basis_no_mesh(V, D, pathname_mesh, sqrt_num_boxes, sensors, indices, itr_num, epsilon):
    mesh_dictionary, region_dictionary = discretize_fem_mesh('square', pathname_mesh, sqrt_num_boxes, indices)
    phi = create_basis(V, D)

    sigma, set_of_phi = create_sigma_no_mesh(region_dictionary, phi)

    # TODO: check why this is sensitive to initial epsilon value

    p, ada = opa.optimal_arrangement(sensors, np.asarray(sigma), itr_num, epsilon)
    print(p, ada)

    return phi, ada, set_of_phi

def y_j(region_dictionary, j, data):
    # Get indices of all elements in regionDictionary in region j
    idxs = sorted([k for (k, v) in region_dictionary.items() if v == j])

    # Extract rows of phi that are within region j
    y = data[idxs, :]
    return y


def y_set(region_dictionary, sensor_locations, data):
    y_set = {}
    for i in sensor_locations:
        y_set[i] = y_j(region_dictionary, i, data)

    return y_set


def test_state_estimation(ada, phi, data, set_of_phi, set_of_da, regionDictionary):
    # Compute estimate for each vector from snapshot l, assuming data is ordered in time
    time = [None] * np.shape(data)[1]

    for l in range(0, np.shape(data)[1]):
        v = data[:, l]  # True data
        # Get all corresponding points from FE discretization over selected sensors
        set_of_y = y_set(regionDictionary, regionDictionary, v)
        v_hat = state_estimation_2D(ada, phi, set_of_phi, set_of_da, set_of_y)  # Estimated data
        time[l] = (v, v_hat)

    return time


def test_state_estimation_no_mesh(ada, phi, data, set_of_phi, regionDictionary):
    # Compute estimate for each vector from snapshot l, assuming data is ordered in time
    time = [None] * np.shape(data)[1]

    for l in range(0, np.shape(data)[1]):
        v = data[:, l]  # True data
        # Get all corresponding points from FE discretization over selected sensors
        set_of_y = y_set(regionDictionary, regionDictionary, v)
        v_hat = state_estimation_2D_no_mesh(ada, phi, set_of_phi, set_of_y)  # Estimated data
        time[l] = (v, v_hat)

    return time


def calculate_POD_basis(node_solution, DA):
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
    K = R # @ DA

    D, V = np.linalg.eig(K)
    print('size of eigenvalues: ', np.shape(V))

    V = V.real
    D = D.real
    I = np.argsort(D)[::-1]

    D = D[I]
    V = V[:,I]

    # plot_energy(D)
    return create_basis(V, D)


# Calculate time dependent coefficients for POD basis for specific time instant
def coefficient(sensor_locations, set_of_phi, set_of_da, set_of_y):
    m = len(sensor_locations)
    n, k = np.shape(set_of_phi[0])

    pi = np.matrix(np.zeros((k, k)))
    sum_field = np.matrix(np.zeros((k, 1)))

    # Check to see this still works with example
    for s_i in sorted(sensor_locations):
        pi += set_of_phi[s_i].T @ set_of_da[s_i] @ set_of_phi[s_i]

        sum_field += set_of_phi[s_i].T @ set_of_da[s_i] @ set_of_y[s_i]

    c_hat = pi.I @ sum_field

    return c_hat


def coefficient_no_mesh(sensor_locations, set_of_phi, set_of_y):
    m = len(sensor_locations)
    n, k = np.shape(set_of_phi[0])

    pi = np.matrix(np.zeros((k, k)))
    sum_field = np.matrix(np.zeros((k, 1)))

    # Check to see this still works with example
    for s_i in sorted(sensor_locations):
        pi += set_of_phi[s_i].T @ set_of_phi[s_i]

        sum_field += set_of_phi[s_i].T @ set_of_y[s_i]

    c_hat = pi.I @ sum_field

    return c_hat


# TODO: Revise to handle more than one sensing type !
def adaptation_scheme(region_dictionary, node_solution, DA, t0, t_resample, sensors):
    node_solution = np.reshape(node_solution, (np.shape(node_solution)[0], 1, np.shape(node_solution)[1]))
    total_time = np.shape(node_solution)[2]
    # Keep start and end times of ada values
    ada_vals = {}

    # Calculate POD basis for initial set of data for time [0, t0)
    truncated_data = node_solution[:, :, 0:t0]
    idx = np.round(np.linspace(0, np.shape(node_solution)[2] - 1, t0)).astype(int)
    truncated_data = node_solution[:, :, idx]
    phi = calculate_POD_basis(truncated_data, DA)
    sigma, set_of_phi, set_of_da = create_sigma(region_dictionary, phi, DA)

    # To start, information from whole field are used
    # ada_vals[(0, t0-1)] = list(set(region_dictionary.values()))

    # Estimate optimal placement information from POD basis
    p, ada = opa.optimal_arrangement(sensors, np.asarray(sigma), 100000000, 10)

    # sample_start_time = t0
    # sample_end_time = t0 + t_resample
    # new_ada = ada
    # new_data = truncated_data
    sample_start_time = 0
    sample_end_time = t_resample
    new_ada = ada
    #new_data = truncated_data
    new_data = np.array([], dtype=np.int32).reshape((np.shape(truncated_data)[0], np.shape(truncated_data)[1], 0))


    # TODO: check to make sure all the updated values are being updated correctly

    while sample_start_time < total_time:
        ada_vals[(sample_start_time, min(sample_end_time, total_time) - 1)] = new_ada

        # Estimate field from sensor arrangement [t0, t1)
        # Get field measurements from time t0 to t1, apply estimation technique using selected sensors
        print(new_ada)
        estimated_vals = np.array([], dtype=np.int32).reshape(np.shape(node_solution)[0], 0)
        for time in range(sample_start_time, min(sample_end_time, total_time)):
            y = node_solution[:, :, time]   # Get data from original data set
            set_of_y = y_set(region_dictionary, new_ada, y)
            y_hat = state_estimation_2D(new_ada, phi, set_of_phi, set_of_da, set_of_y)  # Estimated data -- add noise?
            estimated_vals = np.concatenate((estimated_vals, y_hat), axis=1)

        # Update POD with information from estimated field
        # Add values to ORIGINAL data matrix -- can also add to POD estimated to get "lower order" estimates of field
        # print(np.shape(new_data), np.shape(estimated_vals[:, np.newaxis, :]))
        new_data = np.concatenate((new_data[:, 0, :], estimated_vals), axis=1)
        new_data = new_data[:, np.newaxis, :]
        phi = calculate_POD_basis(new_data, DA)
        new_sigma, set_of_phi, set_of_da = create_sigma(region_dictionary, phi, DA)

        # Calculate new optimal arrangement of sensors
        p, new_ada = opa.optimal_arrangement(sensors, np.asarray(new_sigma), 100000000, 10)

        # Next iteration
        sample_start_time = sample_end_time
        sample_end_time = sample_start_time + t_resample

    return new_data, ada_vals


def adaptation_scheme_keep_sensing_vals(region_dictionary, node_solution, DA, t0, t_resample, sensors):
    node_solution = np.reshape(node_solution, (np.shape(node_solution)[0], 1, np.shape(node_solution)[1]))
    total_time = np.shape(node_solution)[2]
    # Keep start and end times of ada values
    ada_vals = {}

    # Calculate POD basis for initial set of data for time [0, t0)
    truncated_data = node_solution[:, :, 0:t0]
    phi = calculate_POD_basis(truncated_data, DA)
    sigma, set_of_phi, set_of_da = create_sigma(region_dictionary, phi, DA)

    # To start, information from whole field are used
    ada_vals[(0, t0-1)] = list(set(region_dictionary.values()))

    # Estimate optimal placement information from POD basis
    p, ada = opa.optimal_arrangement(sensors, np.asarray(sigma), 1000, 10)

    sample_start_time = t0
    sample_end_time = t0 + t_resample
    new_ada = ada
    new_data = truncated_data

    # TODO: check to make sure all the updated values are being updated correctly

    while sample_start_time < total_time:
        ada_vals[(sample_start_time, min(sample_end_time, total_time) - 1)] = new_ada

        # Estimate field from sensor arrangement [t0, t1)
        # Get field measurements from time t0 to t1, apply estimation technique using selected sensors
        print(new_ada)
        estimated_vals = np.array([], dtype=np.float64).reshape(np.shape(node_solution)[0], 0)
        for time in range(sample_start_time, min(sample_end_time, total_time)):
            y = node_solution[:, :, time]   # Get data from original data set
            set_of_y = y_set(region_dictionary, new_ada, y)
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
        phi = calculate_POD_basis(new_data, DA)
        new_sigma, set_of_phi, set_of_da = create_sigma(region_dictionary, phi, DA)

        # Calculate new optimal arrangement of sensors
        p, new_ada = opa.optimal_arrangement(sensors, np.asarray(new_sigma), 1000, 10)

        # Next iteration
        sample_start_time = sample_end_time
        sample_end_time = sample_start_time + t_resample

    return new_data, ada_vals


def adaptation_scheme_replacement(region_dictionary, node_solution, DA, t0, t_resample, sensors):
    node_solution = np.reshape(node_solution, (np.shape(node_solution)[0], 1, np.shape(node_solution)[1]))
    total_time = np.shape(node_solution)[2]
    # Keep start and end times of ada values
    ada_vals = {}

    # Calculate POD basis for initial set of data for time [0, t0)
    truncated_data = node_solution[:, :, 0:t0]
    phi = calculate_POD_basis(truncated_data, DA)
    sigma, set_of_phi, set_of_da = create_sigma(region_dictionary, phi, DA)

    # To start, information from whole field are used
    ada_vals[(0, t0-1)] = list(set(region_dictionary.values()))

    # Estimate optimal placement information from POD basis
    p, ada = opa.optimal_arrangement(sensors, np.asarray(sigma), 1000, 10)

    sample_start_time = t0
    sample_end_time = t0 + t_resample
    new_ada = ada
    new_data = truncated_data
    sigma_data = truncated_data
    # TODO: check to make sure all the updated values are being updated correctly

    while sample_start_time < total_time:
        ada_vals[(sample_start_time, min(sample_end_time, total_time) - 1)] = new_ada

        # Estimate field from sensor arrangement [t0, t1)
        # Get field measurements from time t0 to t1, apply estimation technique using selected sensors
        print(new_ada)
        estimated_vals = np.array([], dtype=np.float64).reshape(np.shape(node_solution)[0], 0)
        for time in range(sample_start_time, min(sample_end_time, total_time)):
            y = node_solution[:, :, time]   # Get data from original data set
            set_of_y = y_set(region_dictionary, new_ada, y)
            y_hat = state_estimation_2D(new_ada, phi, set_of_phi, set_of_da, set_of_y)  # Estimated data -- add noise?
            estimated_vals = np.concatenate((estimated_vals, y_hat), axis=1)

        # Update POD with information from estimated field
        # Add values to ORIGINAL data matrix -- can also add to POD estimated to get "lower order" estimates of field

        estimated_vals = estimated_vals[:, np.newaxis, :]

        if np.shape(sigma_data)[2] > np.shape(estimated_vals[:, np.newaxis, :])[2]:
            sigma_data = new_data[:, :, np.shape(estimated_vals[:, np.newaxis, :])[2]:np.shape(new_data)[2]]
            sigma_data = np.concatenate((np.asfarray(sigma_data), np.asfarray(estimated_vals)), axis=2)
        else:
            sigma_data = estimated_vals

        new_data = np.concatenate((np.asfarray(new_data), np.asfarray(estimated_vals)), axis=2)
        phi = calculate_POD_basis(sigma_data, DA)
        new_sigma, set_of_phi, set_of_da = create_sigma(region_dictionary, phi, DA)

        # Calculate new optimal arrangement of sensors
        p, new_ada = opa.optimal_arrangement(sensors, np.asarray(new_sigma), 1000, 10)

        # Next iteration
        sample_start_time = sample_end_time
        sample_end_time = sample_start_time + t_resample

    return new_data, ada_vals

# pathname_mesh = '/home/tahiya/Documents/MATLAB/2D_PDE/2eq_2d_5sec_pde_example_soln_mesh.mat'
# pathname_eig = '/home/tahiya/Documents/MATLAB/2D_PDE/2eq_2d_5sec_pde_example_eigens.mat'
# test_discretize_fem_mesh(pathname_mesh)
# mat_contents = sio.loadmat(pathname_mesh)
# nodeSoln = mat_contents['NodalSolution']
# eig_contents = sio.loadmat(pathname_eig)
# D = eig_contents['D']
# plot_energy(D)
# test_create_basis(pathname_eig)


# pathname_mesh = '/home/tahiya/Documents/MATLAB/2DUnsteadyConvection-diffusion/2d_2pisec_pde_example.mat'
# pathname_eig = '/home/tahiya/Documents/MATLAB/2DUnsteadyConvection-diffusion/2d_2pisec_pde_example_eigens.mat'
# pathname_mesh = '/home/tahiya/Documents/MATLAB/2DUnsteadyConvection-diffusion/2eq_2d_500sec_pde_example.mat'
# pathname_eig = '/home/tahiya/Documents/MATLAB/2DUnsteadyConvection-diffusion/2eq_2d_500sec_pde_example_eigens.mat'
# pathname_mesh = '/home/tahiya/Documents/MATLAB/2DUnsteadyConvection-diffusion/2d_500sec_pde_example.mat'
# pathname_eig = '/home/tahiya/Documents/MATLAB/2DUnsteadyConvection-diffusion/2d_500sec_pde_example_eigens.mat'
#
# # test_discretize_fem_mesh(pathname_mesh)
# mat_contents = sio.loadmat(pathname_mesh)
# nodeSoln = mat_contents['NodalSolution']
# meshNodes = mat_contents['MeshNodes']
# time_values = mat_contents['T'].T.tolist()
#
#
# eig_contents = sio.loadmat(pathname_eig)
# D = eig_contents['D']
# V = eig_contents['V']
# DA = eig_contents['DA'].real
#
# calculate_POD_basis(nodeSoln, DA)
#
# plot_energy(D)
# discretization_num = 5
# num_sens = 10
# mesh_dictionary, region_dictionary = discretize_fem_mesh('square', pathname_mesh, 5)
# phi, ada, set_of_phi, set_of_da = test_create_basis(V, D, DA, pathname_mesh, discretization_num, num_sens)
# test_discretize_fem_mesh(pathname_mesh)

# plot_sensing_regions(mesh_dictionary, region_dictionary, ada)

# fig, ax = plt.subplots()
# arts = plot_current_sensing_regions(mesh_dictionary, region_dictionary, ada)
# plt.show()


# for elmt in data_estimates:
#     print(np.sum((elmt[0] - elmt[1])), np.max(elmt[0]), np.max(elmt[1]), np.min(elmt[0]), np.min(elmt[1]))
# granularity = 5
# data_estimates = test_state_estimation(ada, phi, np.matrix(nodeSoln), set_of_phi, set_of_da, region_dictionary)
# plot_field_fe(data_estimates, meshNodes, time_values, mesh_dictionary, region_dictionary, granularity)
# plot_field_opt_est(data_estimates, meshNodes, time_values, mesh_dictionary, region_dictionary, ada, granularity)

# # t0_val = 500 t_resample_val = 1000; t0_val = 250 t_resample_val = 2000; t0_val = 1000 t_resample_val = 500; t0_val = 2000 t_resample_val = 250
# t0_val = 2000
# t_resample_val = 250
# #
# # plot_field_changing_ada(updated_data, meshNodes, time_values, mesh_dictionary, region_dictionary, adas, t0_val, t_resample_val, granularity)
# data = [x[0] for x in data_estimates]
# optimal_data = [x[1] for x in data_estimates]
#
# # # TODO: redo all tests for projection error
# t_0_t_resample = [(500, 1000), (250, 2000), (1000, 500), (2000, 250)]
# for (t0_val, t_resample_val) in t_0_t_resample:
#     updated_data, adas = adaptation_scheme(region_dictionary, nodeSoln, DA, t0_val, t_resample_val, num_sens)
#     plot_projection_relative_error_fe_adpt(data, updated_data, time_values, adas, t0_val, t_resample_val)
#     plot_projection_relative_error_fe_opt(data, optimal_data, time_values, adas, t0_val, t_resample_val)

# TODO: test other PDEs

