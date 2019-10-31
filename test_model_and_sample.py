import numpy as np
import scipy.io as sio
import scipy as sp

from adaptive_modeling_and_sampling import *

# pathname_mesh = '/home/tahiya/Documents/MATLAB/lore_tank/600x600_mesh_fine.mat'
# pathname_node = '/home/tahiya/code/video_dataset/600x600_node_soln_fine.mat'
# pathname_eig = '/home/tahiya/code/video_dataset/600x600_node_soln_fine_eigens.mat'
# pathname_times = '/home/tahiya/code/video_dataset/600x600_node_soln_fine_times.mat'

pathname_mesh = '/Users/tahiyasalam/Documents/Research2018/video_dataset/600x600_mesh_fine.mat'
pathname_node = '/Users/tahiyasalam/Documents/Research2018/video_dataset/600x600_node_soln_fine.mat'
pathname_eig = '/Users/tahiyasalam/Documents/Research2018/video_dataset/600x600_node_soln_fine_eigens.mat'
pathname_times = '/Users/tahiyasalam/Documents/Research2018/video_dataset/600x600_node_soln_fine_times.mat'

t = 1805
# Retrieve these values from MATLAB mesh
mat_contents = sio.loadmat(pathname_mesh)
meshNodes = mat_contents['MeshNodes']
meshNodes = np.matrix.round(meshNodes)
DA = mat_contents['DA'].real

mat_contents = sio.loadmat(pathname_node)
nodeSoln = mat_contents['NodalSolution']

mat_contents = sio.loadmat(pathname_times)
time_values = mat_contents['T'].T.tolist()

eig_contents = sio.loadmat(pathname_eig)
D = eig_contents['D']
V = eig_contents['V']

n = np.shape(DA)[0]

num_squares = 3
num_sens = 4

# num_squares = 3
# num_sens = 4
mesh_dictionary, region_dictionary = discretize_fem_mesh('square', pathname_mesh, num_squares)
phi, ada, set_of_phi, set_of_da = test_create_basis(V, D, DA, pathname_mesh, num_squares, num_sens)
centers = get_centers(mesh_dictionary, region_dictionary)
# plot_discretization(mesh_dictionary, region_dictionary)
# plot_sensing_regions(mesh_dictionary, region_dictionary, ada)

granularity = 1
# data_estimates = test_state_estimation(ada, phi, np.matrix(nodeSoln), set_of_phi, set_of_da, region_dictionary)

G = nx.complete_graph(num_sens)
G_adjacency = nx.to_numpy_array(G)
# Keep list of neighbors for each node
G_neighbors = get_neighbors(G_adjacency)

T = nx.trees.random_tree(num_sens)
T_adjacency = nx.to_numpy_array(T)
T_neighbors = get_neighbors(T_adjacency)

# Cycle graph
Cyc = nx.cycle_graph(num_sens)
Cyc_adjacency = nx.to_numpy_array(Cyc)
Cyc_neighbors = get_neighbors(Cyc_adjacency)

# Wheel graph
Wheel = nx.wheel_graph(num_sens)
Wheel_adjacency = nx.to_numpy_array(Wheel)
Wheel_neighbors = get_neighbors(Wheel_adjacency)

rows_for_agent = np.array_split(np.arange(n-1), num_sens)
# print(rows_for_agent)
k = 50
num_orth_itrs = 20

t_0_t_resample = [(100, 100), (100, 500), (250, 500), (500, 250), (500, 100)]
t_0_t_resample = [(100, 100)]

time_steps = np.arange(50, 1001, 50)

# print(time_values)
dt = time_values[1][0] - time_values[0][0]

create_distr_npz_file = False
create_npz_files = False
plot_from_npz = False
plot_panels = False
plot_snapshots = False
plot_panels_w_baseline = False
plot_mean = False
plot_disc_train = True
plot_interpolate = False
# Create graph data based on computation scheme
types = ['rbf', 'rbf_random', 'centr', 'fixed']


if create_distr_npz_file:
    for (t0_val, t_resample_val) in t_0_t_resample:
        try:
            print("TRIAL WITH: ", (t0_val, t_resample_val))
            updated_data, adas, locs, paths = adaptation_scheme_distributed(region_dictionary, mesh_dictionary, nodeSoln, DA, t0_val, t_resample_val, num_sens, G_neighbors, rows_for_agent, k, num_orth_itrs, centers, dt)
            filename = str(num_sens) + "_sensors_" + str(t0_val) + "_train_" + str(t_resample_val) + "_resample.npz"
            np.savez_compressed(filename, updated_data=updated_data, adas=adas, locs=locs, paths=paths, t0_val=t0_val, t_resample_val=t_resample_val)
            plot_changing_lore_opt_est_with_robot(updated_data, meshNodes, time_values, mesh_dictionary, region_dictionary, adas, locs, paths, centers, t0_val, t_resample_val, granularity)
        except np.linalg.LinAlgError as err:
             print("Error with: ", (t0_val, t_resample_val))
