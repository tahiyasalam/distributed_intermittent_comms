import numpy as np
import scipy.io as sio
import scipy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import transforms
from scipy.interpolate import Rbf
import random


import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist as AA
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import optimal_placement_algorithm as opa



def setup_axes1(fig, rect, xlims, ylims, scaling):
    """
    Setting up axes for floating axes rotation

    Returns new rotated figured depicting sensor locations to embed into the plot
    """
    tr = Affine2D().scale(1, 1).rotate_deg(45)

    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(xlims[0]-scaling, xlims[1]+scaling, ylims[0]-scaling, ylims[1]+scaling))

    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    plswork = fig.add_subplot(ax1)

    aux_ax = ax1.get_aux_axes(tr)

    grid_helper.grid_finder.grid_locator1._nbins = 2
    grid_helper.grid_finder.grid_locator2._nbins = 2

    return ax1, aux_ax, plswork


def setup_axes2(fig, rect, xlims, ylims, scaling):
    """
    Setting up axes for floating axes rotation

    Returns new rotated figured depicting sensor locations to embed into the plot
    """
    tr = Affine2D().scale(1, 1)

    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(xlims[0]-scaling, xlims[1]+scaling, ylims[0]-scaling, ylims[1]+scaling))

    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    plswork = fig.add_subplot(ax1)

    aux_ax = ax1.get_aux_axes(tr)

    grid_helper.grid_finder.grid_locator1._nbins = 2
    grid_helper.grid_finder.grid_locator2._nbins = 2

    return ax1, aux_ax, plswork


def plot_current_sensing_regions(ax, aux, mesh_dictionary, region_dictionary, ada):
    """"
        This returns a plot of the current sensing region on the xy-plane
        Plot will be embedded into plot of field

        Input: MATLAB file describing the mesh of the region used in simulation

        Output: figure, axes

    """
    im2 = []
    print(ada)
    # Get x and y limits from mesh dictionary
    x_i = min([xval for (xval, yval) in mesh_dictionary.values()])
    x_f = max([xval for (xval, yval) in mesh_dictionary.values()])

    y_i = min([yval for (xval, yval) in mesh_dictionary.values()])
    y_f = max([yval for (xval, yval) in mesh_dictionary.values()])

    aux.set_xlim([x_i, x_f])
    aux.set_ylim([y_i, y_f])

    # Set up region dictionary
    num_regions = max(region_dictionary.values())

    cmap = opa.get_cmap(num_regions + 1, 'tab20b')

    for location in region_dictionary.keys():
        if region_dictionary[location] in ada:
            im2, = aux.plot(mesh_dictionary[location][0], mesh_dictionary[location][1],  'o', c=cmap(region_dictionary[location]), markersize=2)
    return ax, aux, im2


def plot_energy(D):
    cum_sum = 0
    total_sum = sum(D).real

    # Plot energy of POD basis
    for i in range(0, 10):
        cum_sum += D[i].real
        percentage_energy = cum_sum/total_sum
        plt.plot(i, percentage_energy, 'o', color='black', markersize='4')
        if percentage_energy <= 0.99:
            plt.annotate(str('%.2f'%(percentage_energy*100) + '%'), (i, percentage_energy))

    plt.xlabel('# of POD basis')
    plt.ylabel('% total energy captured ')
    plt.title('Percentage of total energy captured by POD basis')
    plt.show()
    plt.gcf().clear()


def plot_discretization(mesh_dictionary, region_dictionary):
    plt.figure(1)
    num_regions = max(region_dictionary.values())

    cmap = opa.get_cmap(num_regions + 1, 'tab20b')

    for location in region_dictionary.keys():
        plt.plot(mesh_dictionary[location][0], mesh_dictionary[location][1],  'p', c=cmap(region_dictionary[location]), markersize=8)

    plt.show()
    plt.gcf().clear()


def plot_discretization_figure(mesh_dictionary, region_dictionary):
    fig = plt.figure(1)
    num_regions = max(region_dictionary.values())

    ax = fig.add_subplot(111)
    # ax.set_axisbelow(True)
    # ax.grid(linestyle='-', linewidth='0.5', color='gray')

    ms = 3

    #RED BLUE
    interval = np.hstack([np.linspace(0, 0.35), np.linspace(0.65, 1)])
    colors = plt.cm.RdBu(interval)
    new_cmap = LinearSegmentedColormap.from_list('name', colors)

    for location in region_dictionary.keys():
        plt.plot(mesh_dictionary[location][0], mesh_dictionary[location][1], 'o',
                 c=new_cmap(region_dictionary[location]/num_regions), markersize=ms)
    fig.savefig("redblue_disc.png")
    fig.clf()

    #PURPLE ORANGE
    interval = np.hstack([np.linspace(0, 0.35), np.linspace(0.65, 1)])
    colors = plt.cm.PuOr(interval)
    new_cmap = LinearSegmentedColormap.from_list('name', colors)

    for location in region_dictionary.keys():
        plt.plot(mesh_dictionary[location][0], mesh_dictionary[location][1], 'o',
                 c=new_cmap(region_dictionary[location]/num_regions), markersize=ms)
    fig.savefig("purporange_disc.png")
    fig.clf()

    # GREENS
    interval = np.linspace(0.25, 1.0)
    colors = plt.cm.Greens(interval)
    new_cmap = LinearSegmentedColormap.from_list('name', colors)

    for location in region_dictionary.keys():
        plt.plot(mesh_dictionary[location][0], mesh_dictionary[location][1], 'o',
                 c=new_cmap(region_dictionary[location]/num_regions), markersize=ms)
    fig.savefig("greens_disc.png")
    fig.clf()

    # GREEN + BLUE COLOR MAP
    interval = np.linspace(0.25, 1.0)
    colors = plt.cm.GnBu(interval)
    new_cmap = LinearSegmentedColormap.from_list('name', colors)

    for location in region_dictionary.keys():
        plt.plot(mesh_dictionary[location][0], mesh_dictionary[location][1], 'o',
                 c=new_cmap(region_dictionary[location]/num_regions), markersize=ms)
    fig.savefig("greenblue_disc.png")
    fig.clf()


def plot_sensing_regions(mesh_dictionary, region_dictionary, ada):
    plt.figure(1)
    num_regions = max(region_dictionary.values())

    cmap = opa.get_cmap(num_regions + 1, 'tab20b')

    for location in region_dictionary.keys():
        plt.plot(mesh_dictionary[location][0], mesh_dictionary[location][1],  'p', c=cmap(region_dictionary[location]), markersize=8)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    plt.figure(2)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    num_regions = max(region_dictionary.values())

    cmap = opa.get_cmap(num_regions + 1, 'tab20b')

    for location in region_dictionary.keys():
        if region_dictionary[location] in ada:
            plt.plot(mesh_dictionary[location][0], mesh_dictionary[location][1],  'p', c=cmap(region_dictionary[location]), markersize=8)
    plt.show()
    plt.gcf().clear()


# Plot field and estimate of field as animation
def plot_field_fe(data_and_estimates, mesh_nodes, time_values, mesh_dictionary, region_dictionary, granularity):
    def updateData(idx, n_xmax, n_ymax, z_i, z_f):
        surface_plot[0].remove()

        ground_truth, _ = data_and_estimates[idx]
        Z_gt = np.reshape(np.array(ground_truth), (n_xmax, n_ymax))

        surface_plot[0] = gt_ax.plot_surface(X.T, Y.T, np.flip(Z_gt, 0), cmap=plt.cm.coolwarm, antialiased='False')
        gt_ax.set_zlim([z_i, z_f])

        txt = 't = ' + str.strip(str(round(time_values[idx][0], 2))) + ' secs'
        gt_txt.set_text(txt)

        return surface_plot,

    plt.gcf().clear()
    gt = plt.figure(1)
    gt_ax = plt.axes(projection='3d')

    gt_txt = gt_ax.text2D(0.45, -0.1, '', transform=gt_ax.transAxes, size=14)

    # Does it plot better to pass in limits instead?
    # Limit values
    u_min = float("inf")
    u_max = -float("inf")

    max_time = round(time_values[len(time_values) - 1][0])

    for ground_truth, estimate in data_and_estimates:
        estimate = [0, 0]
        u_gt_min = min(ground_truth)
        u_gt_max = max(ground_truth)

        u_est_min = min(estimate)
        u_est_max = max(estimate)

        if min(u_gt_min, u_est_min) < u_min:
            u_min = min(u_gt_min, u_est_min)

        if max(u_gt_max, u_est_max) > u_max:
            u_max = max(u_gt_max, u_est_max)

    # Set up writer
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    # Find grid dimensions
    n_xmax = len(set(mesh_nodes[0]))
    n_ymax = len(set(mesh_nodes[1]))

    x_i = min(mesh_nodes[0])
    x_f = max(mesh_nodes[0])

    y_i = min(mesh_nodes[1])
    y_f = max(mesh_nodes[0])

    scale = 0.10
    ax_gt, aux_ax_gt, work = setup_axes1(gt, 341, [x_i, x_f], [y_i, y_f], scale)

    num_regions = max(region_dictionary.values())
    cmap = opa.get_cmap(num_regions + 1, 'tab20b')

    for location in region_dictionary.keys():
        small_plot, = aux_ax_gt.plot(mesh_dictionary[location][0], mesh_dictionary[location][1], 'o',
                                               c=cmap(region_dictionary[location]), markersize=2)

    # Reshape from array to grid for plotting
    X = np.reshape(np.array(mesh_nodes[0]), (n_xmax, n_ymax))
    Y = np.reshape(np.array(mesh_nodes[1]), (n_xmax, n_ymax))

    ground_truth, _ = data_and_estimates[0]
    Z_gt = np.reshape(np.array(ground_truth), (n_xmax, n_ymax))

    txt = 't = ' + str.strip(str(round(time_values[0][0], 2))) + ' secs'
    gt_txt.set_text(txt)

    num_regions = max(region_dictionary.values())

    surface_plot = [gt_ax.plot_surface(X.T, Y.T, np.flip(Z_gt, 0), cmap=plt.cm.coolwarm, antialiased='False')]
    gt_ax.set_xlim([x_i, x_f])
    gt_ax.set_ylim([y_i, y_f])

    z_i, z_f = gt_ax.get_zlim()
    gt_ax.set_zlim([u_min, u_max])

    gt_ax.invert_xaxis()

    gt_ax.set_xlabel('x', size=14)
    gt_ax.set_ylabel('y', size=14)
    gt_ax.set_zlabel('u(x,y)', size=14)
    gt.suptitle('FE Simulation for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center',
                verticalalignment='top')

    print("GT animations for ", str(max_time), " secs")
    # Save animations
    gt_ani = animation.FuncAnimation(gt, updateData, frames = [i for i in range(0, len(data_and_estimates))[::granularity]], fargs = (n_xmax, n_ymax, u_min, u_max), interval=50, repeat_delay=3000, blit=False)
    gt_ani.save('gt_' + str(max_time) + 'sec.mp4', writer=writer)


# Plot changing ada values estimate of field as animation
def plot_field_opt_est(data_and_estimates, mesh_nodes, time_values, mesh_dictionary, region_dictionary, ada, granularity):
    def updateData(idx, n_xmax, n_ymax, z_i, z_f):
        surface_plot[0].remove()

        _, estimate = data_and_estimates[idx]
        Z_est = np.reshape(np.array(estimate), (n_xmax, n_ymax))

        surface_plot[0] = est_ax.plot_surface(X.T, Y.T, np.flip(Z_est, 0), cmap=plt.cm.coolwarm, antialiased='False')
        est_ax.set_zlim([z_i, z_f])

        txt = 't = ' + str.strip(str(round(time_values[idx][0], 2))) + ' secs'
        est_txt.set_text(txt)

        return surface_plot,

    plt.gcf().clear()
    est = plt.figure(1)
    est_ax = plt.axes(projection='3d')

    est_txt = est_ax.text2D(0.45, -0.1, '', transform=est_ax.transAxes, size=14)

    # Does it plot better to pass in limits instead?
    # Limit values
    u_min = float("inf")
    u_max = -float("inf")

    max_time = round(time_values[len(time_values) - 1][0])

    for ground_truth, estimate in data_and_estimates:
        estimate = [0, 0]
        u_gt_min = min(ground_truth)
        u_gt_max = max(ground_truth)

        u_est_min = min(estimate)
        u_est_max = max(estimate)

        if min(u_gt_min, u_est_min) < u_min:
            u_min = min(u_gt_min, u_est_min)

        if max(u_gt_max, u_est_max) > u_max:
            u_max = max(u_gt_max, u_est_max)

    # Set up writer
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    # Find grid dimensions
    n_xmax = len(set(mesh_nodes[0]))
    n_ymax = len(set(mesh_nodes[1]))

    x_i = min(mesh_nodes[0])
    x_f = max(mesh_nodes[0])

    y_i = min(mesh_nodes[1])
    y_f = max(mesh_nodes[0])

    scale = 0.10
    ax_est, aux_ax_est, work = setup_axes1(est, 341, [x_i, x_f], [y_i, y_f], scale)

    num_regions = max(region_dictionary.values())
    cmap = opa.get_cmap(num_regions + 1, 'tab20b')

    for location in region_dictionary.keys():
        if region_dictionary[location] in ada:
            small_plot, = aux_ax_est.plot(mesh_dictionary[location][0], mesh_dictionary[location][1], 'o',
                                                   c=cmap(region_dictionary[location]), markersize=2)

    # Reshape from array to grid for plotting
    X = np.reshape(np.array(mesh_nodes[0]), (n_xmax, n_ymax))
    Y = np.reshape(np.array(mesh_nodes[1]), (n_xmax, n_ymax))

    _, estimate = data_and_estimates[0]
    Z_est = np.reshape(np.array(estimate), (n_xmax, n_ymax))

    txt = 't = ' + str.strip(str(round(time_values[0][0], 2))) + ' secs'
    est_txt.set_text(txt)

    surface_plot = [est_ax.plot_surface(X.T, Y.T, np.flip(Z_est, 0), cmap=plt.cm.coolwarm, antialiased='False')]

    est_ax.set_xlim([x_i, x_f])
    est_ax.set_ylim([y_i, y_f])
    # est_ax.set_zlim([u_min, u_max])  # Check what this value should be
    est_ax.invert_xaxis()

    est_ax.set_xlabel('x', size=14)
    est_ax.set_ylabel('y', size=14)
    est_ax.set_zlabel('u(x,y)', size=14)
    est.suptitle('Estimate from Sensors for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center',
                 verticalalignment='top')

    print("EST animations for ", str(max_time), " secs")
    # Save animations
    est_ani = animation.FuncAnimation(est, updateData, frames = [i for i in range(0, len(data_and_estimates))[::granularity]], fargs = (n_xmax, n_ymax, u_min, u_max), interval=50, repeat_delay=3000, blit=False)
    est_ani.save('opt_est_' + str(max_time) + 'sec.mp4', writer=writer)


# Plot field and estimate of field as animation
def plot_lore(data_and_estimates, mesh_nodes, time_values, mesh_dictionary, region_dictionary, granularity):
    def updateData(idx, n_xmax, n_ymax, z_i, z_f):
        surface_plot[0].remove()

        ground_truth, _ = data_and_estimates[idx]
        ground_truth = np.reshape(np.asarray(ground_truth), (np.shape(ground_truth)[0],))

        surface_plot[0] = gt_ax.scatter(np.array(mesh_nodes[1]), np.array(mesh_nodes[0]), c=ground_truth, cmap=new_cmap, antialiased='False', vmin=0.001, vmax=255)

        txt = 't = ' + str.strip(str(round(time_values[idx][0], 2))) + ' secs'
        gt_txt.set_text(txt)

        return surface_plot,

    plt.gcf().clear()
    gt = plt.figure(1)
    gt_ax = plt.axes()

    # Remove the middle 10% of the PRGn_r colormap
    interval = np.hstack([np.linspace(0, 0.35), np.linspace(0.65, 1)])
    colors = plt.cm.PRGn_r(interval)
    new_cmap = LinearSegmentedColormap.from_list('name', colors)
    new_cmap.set_under('white')

    gt_txt = gt_ax.text(0.45, -0.1, '', transform=gt_ax.transAxes, size=14)

    # Does it plot better to pass in limits instead?
    # Limit values
    u_min = float("inf")
    u_max = -float("inf")

    max_time = round(time_values[len(time_values) - 1][0])

    for ground_truth, estimate in data_and_estimates:
        estimate = [0, 0]
        u_gt_min = min(ground_truth)
        u_gt_max = max(ground_truth)

        u_est_min = min(estimate)
        u_est_max = max(estimate)

        if min(u_gt_min, u_est_min) < u_min:
            u_min = min(u_gt_min, u_est_min)

        if max(u_gt_max, u_est_max) > u_max:
            u_max = max(u_gt_max, u_est_max)

    # Set up writer
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    # Find grid dimensions
    n_xmax = len(set(mesh_nodes[0]))
    n_ymax = len(set(mesh_nodes[1]))

    x_i = min(mesh_nodes[0])
    x_f = max(mesh_nodes[0])

    y_i = min(mesh_nodes[1])
    y_f = max(mesh_nodes[0])

    plt.subplots_adjust(right=0.75)

    scale = 0.10
    ax_gt, aux_ax_gt, work = setup_axes2(gt, 341, [x_i, x_f], [y_i, y_f], scale)

    num_regions = max(region_dictionary.values())
    cmap = opa.get_cmap(num_regions + 1, 'tab20b')

    for location in region_dictionary.keys():
        small_plot, = aux_ax_gt.plot(mesh_dictionary[location][1], 599-mesh_dictionary[location][0], 'o',
                                               c=cmap(region_dictionary[location]), markersize=2)

    ground_truth, _ = data_and_estimates[0]
    ground_truth = np.reshape(np.asarray(ground_truth), (np.shape(ground_truth)[0], ))

    txt = 't = ' + str.strip(str(round(time_values[0][0], 2))) + ' secs'
    gt_txt.set_text(txt)

    num_regions = max(region_dictionary.values())

    surface_plot = [gt_ax.scatter(np.array(mesh_nodes[1]), np.array(mesh_nodes[0]), c=ground_truth,
                                    cmap=new_cmap, antialiased='False', vmin=0.001, vmax=255)]

    # surface_plot = [gt_ax.plot_trisurf(np.array(mesh_nodes[0]), np.array(mesh_nodes[1]), ground_truth, cmap=plt.cm.coolwarm, antialiased='False')]
    gt_ax.set_xlim([x_i, x_f])
    gt_ax.set_ylim([y_i, y_f])

    gt_ax.invert_yaxis()
    y_axis_labels = np.arange(100, 601, 100)
    y_axis_labels = [str(label) for label in y_axis_labels][::-1]  # Flipping labels for conversion from rc matrix to xy coordinates
    gt_ax.set_yticklabels(y_axis_labels)

    gt.suptitle('LoRe Tank Video for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center',
                verticalalignment='top')

    plt.colorbar(surface_plot[0], orientation='vertical', ax=gt_ax, cmap=plt.cm.PRGn_r)

    pos1 = (gt.get_axes()[0]).get_position()
    pos2 = (gt.get_axes()[1]).get_position()
    pos3 = (gt.get_axes()[2]).get_position()

    offset_x = 0.05
    pos2.x1 = pos2.x1 - pos2.x0 + offset_x
    pos2.x0 = offset_x

    pos1.x1 = pos1.x1 - pos1.x0 + pos2.x1 + offset_x*3.25
    pos1.x0 = pos2.x1 + offset_x*1.25

    pos3.x1 = pos3.x1 - pos3.x0 + pos1.x1 + offset_x*2.25
    pos3.x0 = pos1.x1 + offset_x*0.25

    (gt.get_axes()[0]).set_position(pos1)
    (gt.get_axes()[1]).set_position(pos2)
    (gt.get_axes()[2]).set_position(pos3)

    print("GT animations for ", str(max_time), " secs")
    # Save animations
    gt_ani = animation.FuncAnimation(gt, updateData, frames = [i for i in range(0, len(data_and_estimates))[::granularity]], fargs = (n_xmax, n_ymax, u_min, u_max), interval=50, repeat_delay=3000, blit=False)
    gt_ani.save('gt_' + str(max_time) + 'sec.mp4', writer=writer)


# Plot changing ada values estimate of field as animation
def plot_lore_opt_est(data_and_estimates, mesh_nodes, time_values, mesh_dictionary, region_dictionary, ada, granularity):
    def updateData(idx, n_xmax, n_ymax, z_i, z_f):
        surface_plot[0].remove()

        _, estimate = data_and_estimates[idx]
        estimate = np.reshape(np.asarray(estimate), (np.shape(estimate)[0],))

        surface_plot[0] = est_ax.scatter(np.array(mesh_nodes[1]), np.array(mesh_nodes[0]), c=estimate, cmap=new_cmap, antialiased='False', vmin=30, vmax=255)

        txt = 't = ' + str.strip(str(round(time_values[idx][0], 2))) + ' secs'
        est_txt.set_text(txt)

        return surface_plot,

    plt.gcf().clear()
    est = plt.figure(1)
    est_ax = plt.axes()

    interval = np.hstack([np.linspace(0, 0.35), np.linspace(0.65, 1)])
    colors = plt.cm.PRGn_r(interval)
    new_cmap = LinearSegmentedColormap.from_list('name', colors)
    new_cmap.set_under('white')

    est_txt = est_ax.text(0.45, -0.1, '', transform=est_ax.transAxes, size=14)

    # Does it plot better to pass in limits instead?
    # Limit values
    u_min = float("inf")
    u_max = -float("inf")

    max_time = round(time_values[len(time_values) - 1][0])

    for ground_truth, estimate in data_and_estimates:
        estimate = [0, 0]
        u_gt_min = min(ground_truth)
        u_gt_max = max(ground_truth)

        u_est_min = min(estimate)
        u_est_max = max(estimate)

        if min(u_gt_min, u_est_min) < u_min:
            u_min = min(u_gt_min, u_est_min)

        if max(u_gt_max, u_est_max) > u_max:
            u_max = max(u_gt_max, u_est_max)

    # Set up writer
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    # Find grid dimensions
    n_xmax = len(set(mesh_nodes[0]))
    n_ymax = len(set(mesh_nodes[1]))

    x_i = min(mesh_nodes[0])
    x_f = max(mesh_nodes[0])

    y_i = min(mesh_nodes[1])
    y_f = max(mesh_nodes[0])

    plt.subplots_adjust(right=0.75)

    scale = 0.10
    ax_est, aux_ax_est, work = setup_axes2(est, 341, [x_i, x_f], [y_i, y_f], scale)

    num_regions = max(region_dictionary.values())
    cmap = opa.get_cmap(num_regions + 1, 'tab20b')

    for location in region_dictionary.keys():
        if region_dictionary[location] in ada:
            small_plot, = aux_ax_est.plot(mesh_dictionary[location][1], 599-mesh_dictionary[location][0], 'o',
                                                   c=cmap(region_dictionary[location]), markersize=2)
    # Reshape from array to grid for plotting

    _, estimate = data_and_estimates[0]
    estimate = np.reshape(np.asarray(estimate), (np.shape(estimate)[0], ))

    txt = 't = ' + str.strip(str(round(time_values[0][0], 2))) + ' secs'
    est_txt.set_text(txt)

    surface_plot = [est_ax.scatter(np.array(mesh_nodes[1]), np.array(mesh_nodes[0]), c=estimate,
                                    cmap=new_cmap, antialiased='False', vmin=30, vmax=255)]

    est_ax.set_xlim([x_i, x_f])
    est_ax.set_ylim([y_i, y_f])

    est_ax.invert_yaxis()
    y_axis_labels = np.arange(100, 601, 100)
    y_axis_labels = [str(label) for label in y_axis_labels][::-1]  # Flipping labels for conversion from rc matrix to xy coordinates
    est_ax.set_yticklabels(y_axis_labels)

    est.suptitle('Estimate from Sensors for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center',
                 verticalalignment='top')

    plt.colorbar(surface_plot[0], orientation='vertical', ax=est_ax, cmap=plt.cm.PRGn_r)

    pos1 = (est.get_axes()[0]).get_position()
    pos2 = (est.get_axes()[1]).get_position()
    pos3 = (est.get_axes()[2]).get_position()

    offset_x = 0.05
    pos2.x1 = pos2.x1 - pos2.x0 + offset_x
    pos2.x0 = offset_x

    pos1.x1 = pos1.x1 - pos1.x0 + pos2.x1 + offset_x*3.25
    pos1.x0 = pos2.x1 + offset_x*1.25

    pos3.x1 = pos3.x1 - pos3.x0 + pos1.x1 + offset_x*2.25
    pos3.x0 = pos1.x1 + offset_x*0.25

    (est.get_axes()[0]).set_position(pos1)
    (est.get_axes()[1]).set_position(pos2)
    (est.get_axes()[2]).set_position(pos3)

    print("EST animations for ", str(max_time), " secs")
    # Save animations
    est_ani = animation.FuncAnimation(est, updateData, frames = [i for i in range(0, len(data_and_estimates))[::granularity]], fargs = (n_xmax, n_ymax, u_min, u_max), interval=50, repeat_delay=3000, blit=False)
    est_ani.save('opt_est_' + str(max_time) + 'sec.mp4', writer=writer)
# TODO make plots comparing FE to POD basis


# Plot changing ada values estimate of field as animation
def plot_changing_lore_opt_est(data, mesh_nodes, time_values, mesh_dictionary, region_dictionary, ada_vals, t0, t_resample, granularity):
    def updateData(idx, small_plot, n_xmax, n_ymax, z_i, z_f, cmap):
        surface_plot[0].remove()
        aux_ax_adpt_est.cla()

        for (times, ada_val) in ada_vals.items():
            if times[0] <= idx <= times[1]:
                locations = ada_val
                break

        for location in region_dictionary.keys():
            if region_dictionary[location] in locations:
                small_plot, = aux_ax_adpt_est.plot(mesh_dictionary[location][1], 599-mesh_dictionary[location][0], 'o',
                                c=cmap(region_dictionary[location]), markersize=2)

        adpt_estimate = data[:, :, idx]
        adpt_estimate = np.reshape(np.asarray(adpt_estimate), (np.shape(adpt_estimate)[0],))

        surface_plot[0] = adpt_est_ax.scatter(np.array(mesh_nodes[1]), np.array(mesh_nodes[0]), c=adpt_estimate, cmap=new_cmap, antialiased='False', vmin=0.001, vmax=255)

        txt = 't = ' + str.strip(str(round(time_values[idx][0], 2))) + ' secs'
        adpt_est_txt.set_text(txt)

        return surface_plot,

    plt.gcf().clear()
    adpt_est = plt.figure(1)
    adpt_est_ax = plt.axes()

    interval = np.hstack([np.linspace(0, 0.35), np.linspace(0.65, 1)])
    colors = plt.cm.PRGn_r(interval)
    new_cmap = LinearSegmentedColormap.from_list('name', colors)
    new_cmap.set_under('white')

    adpt_est_txt = adpt_est_ax.text(0.45, -0.1, '', transform=adpt_est_ax.transAxes, size=14)

    # Limit values
    u_min = float("inf")
    u_max = -float("inf")

    max_time = round(time_values[len(time_values) - 1][0])

    # Does it plot better to pass in limits instead?
    for adpt_estimate in data:
        u_est_min = adpt_estimate.min()
        u_est_max = adpt_estimate.max()

        if u_est_min < u_min:
            u_min = u_est_min

        if u_est_max > u_max:
            u_max = u_est_max

    # Set up writer
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    # Find grid dimensions
    n_xmax = len(set(mesh_nodes[0]))
    n_ymax = len(set(mesh_nodes[1]))

    x_i = min(mesh_nodes[0])
    x_f = max(mesh_nodes[0])

    y_i = min(mesh_nodes[1])
    y_f = max(mesh_nodes[0])

    plt.subplots_adjust(right=0.75)

    scale = 0.10
    ax_adpt_est, aux_ax_adpt_est, work = setup_axes2(adpt_est, 341, [x_i, x_f], [y_i, y_f], scale)

    num_regions = max(region_dictionary.values())
    cmap = opa.get_cmap(num_regions + 1, 'tab20b')

    x_data = []
    y_data = []
    for location in region_dictionary.keys():
        x_data.append(mesh_dictionary[location][1])
        y_data.append(599-mesh_dictionary[location][0])

    small_plot = aux_ax_adpt_est.plot(x_data, y_data, 'o', markersize=2)

    adpt_estimate = data[:,:,0]
    estimate = np.reshape(np.asarray(adpt_estimate), (np.shape(adpt_estimate)[0], ))

    txt = 't = ' + str.strip(str(round(time_values[0][0], 2))) + ' secs'
    adpt_est_txt.set_text(txt)

    num_regions = max(region_dictionary.values())

    cmap = opa.get_cmap(num_regions + 1, 'tab20b')

    surface_plot = [adpt_est_ax.scatter(np.array(mesh_nodes[1]), np.array(mesh_nodes[0]), c=estimate,
                                    cmap=new_cmap, antialiased='False', vmin=0.001, vmax=255)]

    adpt_est_ax.set_xlim([x_i, x_f])
    adpt_est_ax.set_ylim([y_i, y_f])

    adpt_est_ax.invert_yaxis()
    y_axis_labels = np.arange(100, 601, 100)
    y_axis_labels = [str(label) for label in y_axis_labels][::-1]  # Flipping labels for conversion from rc matrix to xy coordinates
    adpt_est_ax.set_yticklabels(y_axis_labels)

    adpt_est.suptitle('Estimate from Sensors for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center',
                 verticalalignment='top')

    plt.colorbar(surface_plot[0], orientation='vertical', ax=adpt_est_ax, cmap=plt.cm.PRGn_r)

    pos1 = (adpt_est.get_axes()[0]).get_position()
    pos2 = (adpt_est.get_axes()[1]).get_position()
    pos3 = (adpt_est.get_axes()[2]).get_position()

    offset_x = 0.05
    pos2.x1 = pos2.x1 - pos2.x0 + offset_x
    pos2.x0 = offset_x

    pos1.x1 = pos1.x1 - pos1.x0 + pos2.x1 + offset_x*3.25
    pos1.x0 = pos2.x1 + offset_x*1.25

    pos3.x1 = pos3.x1 - pos3.x0 + pos1.x1 + offset_x*2.25
    pos3.x0 = pos1.x1 + offset_x*0.25

    (adpt_est.get_axes()[0]).set_position(pos1)
    (adpt_est.get_axes()[1]).set_position(pos2)
    (adpt_est.get_axes()[2]).set_position(pos3)

    t0_time = round(time_values[t0][0])
    t_resample_val_time = round(time_values[t_resample][0])

    print("ADPT EST animations  ", str(max_time), " secs")
    est_ani = animation.FuncAnimation(adpt_est, updateData, frames = [i for i in range(0, np.shape(data)[2])[::granularity]], fargs = (small_plot, n_xmax, n_ymax, u_min, u_max, cmap), interval=50, repeat_delay=3000, blit=False)
    est_ani.save('adpt_est_'  + 'all_sensors_' + str(t0_time) + '_recompute_' +
                str(t_resample_val_time) + '_total_' + str(max_time) + 'sec.mp4', writer=writer)


# Plot changing ada values estimate of field as animation
def plot_changing_lore_opt_est_with_robot(data, mesh_nodes, time_values, mesh_dictionary, region_dictionary, ada_vals, loc_vals, path_vals, centers, t0, t_resample, granularity):
    def updateData(idx, small_plot, n_xmax, n_ymax, z_i, z_f, cmap):
        surface_plot[0].remove()

        change = False
        for (times, ada_val) in ada_vals.items():
            if times[0] <= idx <= times[1]:
                if times[0] == idx:
                    change = True
                start_time = times[0]
                locations = ada_val
                robot_locations = loc_vals[times]
                robot_paths = path_vals[times]
                break

        # print(start_time)
        # for (times, loc_val) in loc_vals.items():
        #     if times[0] <= idx <= times[1]:
        #         if times[0] == idx:
        #             change = True
        #         robot_locations = loc_val
        #         break
        aux_ax_adpt_est.cla()

        # if change:
        #     for location in region_dictionary.keys():
        #         if region_dictionary[location] in locations:
        #             small_plot, = aux_ax_adpt_est.plot(mesh_dictionary[location][1], 599-mesh_dictionary[location][0],
        #                                                marker='o', c=cmap(region_dictionary[location]), markersize=1)

            # for rob_loc in robot_locations:
            #     small_plot, = aux_ax_adpt_est.plot(centers[rob_loc][1], 599 - centers[rob_loc][0], marker='^',
            #                                        c='black', markersize=8, zorder=3)
        for i, path in enumerate(robot_paths):
            if path:
                if idx - start_time < len(path):
                    x, y = zip(*path)
                    small_plot, = aux_ax_adpt_est.plot(np.asarray(np.asarray(y[0:(idx-start_time)])), 599 - np.asarray(x[0:(idx-start_time)]), marker='_',
                                                       c='black', markersize=2, zorder=3)
                    # ln_idx[i].append(aux_ax_adpt_est.lines[len(aux_ax_adpt_est.lines)-1])
                    small_plot, = aux_ax_adpt_est.plot(y[(idx-start_time)], 599 - x[(idx-start_time)], marker='^',
                                                       c='black', markersize=2, zorder=3)
                    # ln.remove()
                    # rob_marks.remove()
                # elif idx - start_time == len(path):
                else:

                    # print(idx, i, ln_idx[i])
                    # for j in range(0, len(ln_idx[i])):
                    #     aux_ax_adpt_est.lines.remove(ln_idx[i][j])
                    rob_loc = robot_locations[i]

                    for location in region_dictionary.keys():
                        if region_dictionary[location] == rob_loc:
                            small_plot, = aux_ax_adpt_est.plot(mesh_dictionary[location][1], 599 - mesh_dictionary[location][0],
                                                           marker='o', c=cmap(region_dictionary[location]), markersize=1)

                    small_plot, = aux_ax_adpt_est.plot(centers[rob_loc][1], 599 - centers[rob_loc][0], marker='^',
                                                       c='black', markersize=2, zorder=3)
            else:
                rob_loc = robot_locations[i]
                # print(rob_loc)
                for location in region_dictionary.keys():
                    if region_dictionary[location] == rob_loc:
                        small_plot, = aux_ax_adpt_est.plot(mesh_dictionary[location][1],
                                                           599 - mesh_dictionary[location][0],
                                                           marker='o', c=cmap(region_dictionary[location]),
                                                           markersize=1)

                small_plot, = aux_ax_adpt_est.plot(centers[rob_loc][1], 599 - centers[rob_loc][0], marker='^',
                                                   c='black', markersize=2, zorder=3)

                    # ln_idx[i] = []
        if not robot_paths:

            for location in region_dictionary.keys():
                if region_dictionary[location] in robot_locations:
                    small_plot, = aux_ax_adpt_est.plot(mesh_dictionary[location][1], 599-mesh_dictionary[location][0],
                                                       marker='o', c=cmap(region_dictionary[location]), markersize=1)

            for rob_loc in robot_locations:
                small_plot, = aux_ax_adpt_est.plot(centers[rob_loc][1], 599 - centers[rob_loc][0], marker='^',
                                                   c='black', markersize=2, zorder=3)
            # rob_loc = robot_locations[i]
            # # print(rob_loc)
            # for location in region_dictionary.keys():
            #     if region_dictionary[location] == rob_loc:
            #         small_plot, = aux_ax_adpt_est.plot(mesh_dictionary[location][1],
            #                                            599 - mesh_dictionary[location][0],
            #                                            marker='o', c=cmap(region_dictionary[location]),
            #                                            markersize=1)
            #
            # small_plot, = aux_ax_adpt_est.plot(centers[rob_loc][1], 599 - centers[rob_loc][0], marker='^',
            #                                    c='black', markersize=2, zorder=3)

        adpt_estimate = data[:, :, idx]
        adpt_estimate = np.reshape(np.asarray(adpt_estimate), (np.shape(adpt_estimate)[0],))

        surface_plot[0] = adpt_est_ax.scatter(np.array(mesh_nodes[1]), np.array(mesh_nodes[0]), c=adpt_estimate, cmap=new_cmap, antialiased='False', vmin=0.001, vmax=255)

        txt = 't = ' + str.strip(str(round(time_values[idx][0], 2))) + ' secs'
        adpt_est_txt.set_text(txt)

        return surface_plot,

    ln_idx = []
    for i in range(0, 4):
        ln_idx.append([])

    plt.gcf().clear()
    adpt_est = plt.figure(1)
    adpt_est_ax = plt.axes()

    interval = np.hstack([np.linspace(0, 0.35), np.linspace(0.65, 1)])
    colors = plt.cm.PRGn_r(interval)
    new_cmap = LinearSegmentedColormap.from_list('name', colors)
    new_cmap.set_under('white')

    adpt_est_txt = adpt_est_ax.text(0.45, -0.1, '', transform=adpt_est_ax.transAxes, size=14)

    # Limit values
    u_min = float("inf")
    u_max = -float("inf")

    max_time = round(time_values[len(time_values) - 1][0])

    # Does it plot better to pass in limits instead?
    for adpt_estimate in data:
        u_est_min = adpt_estimate.min()
        u_est_max = adpt_estimate.max()

        if u_est_min < u_min:
            u_min = u_est_min

        if u_est_max > u_max:
            u_max = u_est_max

    # Set up writer
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    # Find grid dimensions
    n_xmax = len(set(mesh_nodes[0]))
    n_ymax = len(set(mesh_nodes[1]))

    x_i = min(mesh_nodes[0])
    x_f = max(mesh_nodes[0])

    y_i = min(mesh_nodes[1])
    y_f = max(mesh_nodes[0])

    plt.subplots_adjust(right=0.75)

    scale = 0.10
    ax_adpt_est, aux_ax_adpt_est, work = setup_axes2(adpt_est, 341, [x_i, x_f], [y_i, y_f], scale)

    num_regions = max(region_dictionary.values())
    cmap = opa.get_cmap(num_regions + 1, 'tab20b')

    x_data = []
    y_data = []
    for location in region_dictionary.keys():
        x_data.append(mesh_dictionary[location][1])
        y_data.append(599-mesh_dictionary[location][0])

    small_plot = aux_ax_adpt_est.plot(x_data, y_data, '1', markersize=2)

    adpt_estimate = data[:,:,0]
    estimate = np.reshape(np.asarray(adpt_estimate), (np.shape(adpt_estimate)[0], ))

    txt = 't = ' + str.strip(str(round(time_values[0][0], 2))) + ' secs'
    adpt_est_txt.set_text(txt)

    num_regions = max(region_dictionary.values())

    cmap = opa.get_cmap(num_regions + 1, 'tab20b')

    surface_plot = [adpt_est_ax.scatter(np.array(mesh_nodes[1]), np.array(mesh_nodes[0]), c=estimate,
                                    cmap=new_cmap, antialiased='False', vmin=0.001, vmax=255)]

    adpt_est_ax.set_xlim([x_i, x_f])
    adpt_est_ax.set_ylim([y_i, y_f])

    adpt_est_ax.invert_yaxis()
    y_axis_labels = np.arange(100, 601, 100)
    y_axis_labels = [str(label) for label in y_axis_labels][::-1]  # Flipping labels for conversion from rc matrix to xy coordinates
    adpt_est_ax.set_yticklabels(y_axis_labels)

    adpt_est.suptitle('Estimate from Sensors for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center',
                 verticalalignment='top')

    plt.colorbar(surface_plot[0], orientation='vertical', ax=adpt_est_ax, cmap=plt.cm.PRGn_r)

    pos1 = (adpt_est.get_axes()[0]).get_position()
    pos2 = (adpt_est.get_axes()[1]).get_position()
    pos3 = (adpt_est.get_axes()[2]).get_position()

    offset_x = 0.05
    pos2.x1 = pos2.x1 - pos2.x0 + offset_x
    pos2.x0 = offset_x

    pos1.x1 = pos1.x1 - pos1.x0 + pos2.x1 + offset_x*3.25
    pos1.x0 = pos2.x1 + offset_x*1.25

    pos3.x1 = pos3.x1 - pos3.x0 + pos1.x1 + offset_x*2.25
    pos3.x0 = pos1.x1 + offset_x*0.25

    (adpt_est.get_axes()[0]).set_position(pos1)
    (adpt_est.get_axes()[1]).set_position(pos2)
    (adpt_est.get_axes()[2]).set_position(pos3)

    t0_time = round(time_values[t0][0])
    t_resample_val_time = round(time_values[t_resample][0])

    print("ADPT EST animations  ", str(max_time), " secs")
    est_ani = animation.FuncAnimation(adpt_est, updateData, frames = [i for i in range(0, np.shape(data)[2])[::granularity]], fargs = (small_plot, n_xmax, n_ymax, u_min, u_max, cmap), interval=50, repeat_delay=3000, blit=False)
    est_ani.save('adpt_est_'  + 'all_sensors_' + str(t0_time) + '_recompute_' +
                str(t_resample_val_time) + '_total_' + str(max_time) + 'sec.mp4', writer=writer)

    plt.gcf().clear()

# Plot changing ada values estimate of field as animation
def plot_changing_lore_opt_est_with_robot2(data, mesh_nodes, time_values, mesh_dictionary, region_dictionary, ada_vals, loc_vals, path_vals, centers, t0, t_resample, granularity):
    def updateData(idx, small_plot, n_xmax, n_ymax, z_i, z_f, cmap):
        surface_plot[0].remove()

        change = False
        for (times, ada_val) in ada_vals.items():
            if times[0] <= idx <= times[1]:
                if times[0] == idx:
                    change = True
                start_time = times[0]
                locations = ada_val
                robot_locations = loc_vals[times]
                robot_paths = path_vals[times]
                break

        aux_ax_adpt_est.cla()

        for i, path in enumerate(robot_paths):
            if path:
                if idx - start_time < len(path):
                    x, y = zip(*path)
                    small_plot, = aux_ax_adpt_est.plot(599 - np.asarray(np.asarray(y[0:(idx-start_time)])), 599 - np.asarray(x[0:(idx-start_time)]), marker='_',
                                                       c='black', markersize=2, zorder=3)
                    # ln_idx[i].append(aux_ax_adpt_est.lines[len(aux_ax_adpt_est.lines)-1])
                    small_plot, = aux_ax_adpt_est.plot(599-y[(idx-start_time)], 599 - x[(idx-start_time)], marker='^',
                                                       c='black', markersize=2, zorder=3)
                    # ln.remove()
                    # rob_marks.remove()
                # elif idx - start_time == len(path):
                else:

                    # print(idx, i, ln_idx[i])
                    # for j in range(0, len(ln_idx[i])):
                    #     aux_ax_adpt_est.lines.remove(ln_idx[i][j])
                    rob_loc = robot_locations[i]

                    for location in region_dictionary.keys():
                        if region_dictionary[location] == rob_loc:
                            small_plot, = aux_ax_adpt_est.plot(599 - mesh_dictionary[location][1], 599 - mesh_dictionary[location][0],
                                                           marker='o', c=cmap(region_dictionary[location]), markersize=1)

                    small_plot, = aux_ax_adpt_est.plot(599 - centers[rob_loc][1], 599 - centers[rob_loc][0], marker='^',
                                                       c='black', markersize=2, zorder=3)
            else:
                rob_loc = robot_locations[i]
                # print(rob_loc)
                for location in region_dictionary.keys():
                    if region_dictionary[location] == rob_loc:
                        small_plot, = aux_ax_adpt_est.plot(599-mesh_dictionary[location][1],
                                                           599 - mesh_dictionary[location][0],
                                                           marker='o', c=cmap(region_dictionary[location]),
                                                           markersize=1)

                small_plot, = aux_ax_adpt_est.plot(599-centers[rob_loc][1], 599 - centers[rob_loc][0], marker='^',
                                                   c='black', markersize=2, zorder=3)

                    # ln_idx[i] = []
        if not robot_paths:

            for location in region_dictionary.keys():
                if region_dictionary[location] in robot_locations:
                    small_plot, = aux_ax_adpt_est.plot(599-mesh_dictionary[location][1], 599-mesh_dictionary[location][0],
                                                       marker='o', c=cmap(region_dictionary[location]), markersize=1)

            for rob_loc in robot_locations:
                small_plot, = aux_ax_adpt_est.plot(599-centers[rob_loc][1], 599 - centers[rob_loc][0], marker='^',
                                                   c='black', markersize=2, zorder=3)
            # rob_loc = robot_locations[i]
            # # print(rob_loc)
            # for location in region_dictionary.keys():
            #     if region_dictionary[location] == rob_loc:
            #         small_plot, = aux_ax_adpt_est.plot(mesh_dictionary[location][1],
            #                                            599 - mesh_dictionary[location][0],
            #                                            marker='o', c=cmap(region_dictionary[location]),
            #                                            markersize=1)
            #
            # small_plot, = aux_ax_adpt_est.plot(centers[rob_loc][1], 599 - centers[rob_loc][0], marker='^',
            #                                    c='black', markersize=2, zorder=3)

        adpt_estimate = data[:, :, idx]
        adpt_estimate = np.reshape(np.asarray(adpt_estimate), (np.shape(adpt_estimate)[0],))

        surface_plot[0] = adpt_est_ax.scatter(np.array(mesh_nodes[1]), np.array(mesh_nodes[0]), c=adpt_estimate, cmap=new_cmap, antialiased='False', vmin=0.001, vmax=255)

        txt = 't = ' + str.strip(str(round(time_values[idx][0], 2))) + ' secs'
        adpt_est_txt.set_text(txt)

        return surface_plot,

    ln_idx = []
    for i in range(0, 4):
        ln_idx.append([])

    plt.gcf().clear()
    adpt_est = plt.figure(1)
    adpt_est_ax = plt.axes()

    interval = np.hstack([np.linspace(0, 0.35), np.linspace(0.65, 1)])
    colors = plt.cm.PRGn_r(interval)
    new_cmap = LinearSegmentedColormap.from_list('name', colors)
    new_cmap.set_under('white')

    adpt_est_txt = adpt_est_ax.text(0.45, -0.1, '', transform=adpt_est_ax.transAxes, size=14)

    # Limit values
    u_min = float("inf")
    u_max = -float("inf")

    max_time = round(time_values[len(time_values) - 1][0])

    # Does it plot better to pass in limits instead?
    for adpt_estimate in data:
        u_est_min = adpt_estimate.min()
        u_est_max = adpt_estimate.max()

        if u_est_min < u_min:
            u_min = u_est_min

        if u_est_max > u_max:
            u_max = u_est_max

    # Set up writer
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    # Find grid dimensions
    n_xmax = len(set(mesh_nodes[0]))
    n_ymax = len(set(mesh_nodes[1]))

    x_i = min(mesh_nodes[0])
    x_f = max(mesh_nodes[0])

    y_i = min(mesh_nodes[1])
    y_f = max(mesh_nodes[0])

    plt.subplots_adjust(right=0.75)

    scale = 0.10
    ax_adpt_est, aux_ax_adpt_est, work = setup_axes2(adpt_est, 341, [x_i, x_f], [y_i, y_f], scale)

    num_regions = max(region_dictionary.values())
    cmap = opa.get_cmap(num_regions + 1, 'tab20b')

    x_data = []
    y_data = []
    for location in region_dictionary.keys():
        x_data.append(mesh_dictionary[location][1])
        y_data.append(599-mesh_dictionary[location][0])

    small_plot = aux_ax_adpt_est.plot(x_data, y_data, '1', markersize=2)

    adpt_estimate = data[:,:,0]
    estimate = np.reshape(np.asarray(adpt_estimate), (np.shape(adpt_estimate)[0], ))

    txt = 't = ' + str.strip(str(round(time_values[0][0], 2))) + ' secs'
    adpt_est_txt.set_text(txt)

    num_regions = max(region_dictionary.values())

    cmap = opa.get_cmap(num_regions + 1, 'tab20b')

    surface_plot = [adpt_est_ax.scatter(np.array(mesh_nodes[1]), np.array(mesh_nodes[0]), c=estimate,
                                    cmap=new_cmap, antialiased='False', vmin=0.001, vmax=255)]

    adpt_est_ax.set_xlim([x_i, x_f])
    adpt_est_ax.set_ylim([y_i, y_f])

    # adpt_est_ax.invert_yaxis()
    y_axis_labels = np.arange(100, 601, 100)
    # y_axis_labels = [str(label) for label in y_axis_labels][::-1]  # Flipping labels for conversion from rc matrix to xy coordinates
    # adpt_est_ax.set_yticklabels(y_axis_labels)

    adpt_est.suptitle('Estimate from Sensors for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center',
                 verticalalignment='top')

    plt.colorbar(surface_plot[0], orientation='vertical', ax=adpt_est_ax, cmap=plt.cm.PRGn_r)

    pos1 = (adpt_est.get_axes()[0]).get_position()
    pos2 = (adpt_est.get_axes()[1]).get_position()
    pos3 = (adpt_est.get_axes()[2]).get_position()

    offset_x = 0.05
    pos2.x1 = pos2.x1 - pos2.x0 + offset_x
    pos2.x0 = offset_x

    pos1.x1 = pos1.x1 - pos1.x0 + pos2.x1 + offset_x*3.25
    pos1.x0 = pos2.x1 + offset_x*1.25

    pos3.x1 = pos3.x1 - pos3.x0 + pos1.x1 + offset_x*2.25
    pos3.x0 = pos1.x1 + offset_x*0.25

    (adpt_est.get_axes()[0]).set_position(pos1)
    (adpt_est.get_axes()[1]).set_position(pos2)
    (adpt_est.get_axes()[2]).set_position(pos3)

    t0_time = round(time_values[t0][0])
    t_resample_val_time = round(time_values[t_resample][0])

    print("ADPT EST animations  ", str(max_time), " secs")
    est_ani = animation.FuncAnimation(adpt_est, updateData, frames = [i for i in range(0, np.shape(data)[2])[::granularity]], fargs = (small_plot, n_xmax, n_ymax, u_min, u_max, cmap), interval=50, repeat_delay=3000, blit=False)
    est_ani.save('adpt_est_'  + 'all_sensors_' + str(t0_time) + '_recompute_' +
                str(t_resample_val_time) + '_total_' + str(max_time) + 'sec.mp4', writer=writer)

    plt.gcf().clear()

# Plot changing ada values estimate of field as animation
def plot_field_changing_ada(data, mesh_nodes, time_values, mesh_dictionary, region_dictionary, ada_vals, t0, t_resample, granularity):
    def updateData(idx, small_plot, n_xmax, n_ymax, z_i, z_f, cmap):
        surface_plot[0].remove()
        aux_ax_adpt_est.cla()

        for (times, ada_val) in ada_vals.items():
            if times[0] <= idx <= times[1]:
                locations = ada_val
                break

        for location in region_dictionary.keys():
            if region_dictionary[location] in locations:
                small_plot, = aux_ax_adpt_est.plot(mesh_dictionary[location][0], mesh_dictionary[location][1], 'o',
                                c=cmap(region_dictionary[location]), markersize=2)

        adpt_estimate = data[:, :, idx]
        Z_est = np.reshape(np.array(adpt_estimate), (n_xmax, n_ymax))

        surface_plot[0] = adpt_est_ax.plot_surface(X.T, Y.T, np.flip(Z_est, 0), cmap=plt.cm.coolwarm, antialiased='False')
        adpt_est_ax.set_zlim([z_i, z_f])

        txt = 't = ' + str.strip(str(round(time_values[idx][0], 2))) + ' secs'
        adpt_est_txt.set_text(txt)

        return small_plot, surface_plot,

    plt.gcf().clear()
    adpt_est = plt.figure(1)
    adpt_est_ax = plt.axes(projection='3d')

    adpt_est_txt = adpt_est_ax.text2D(0.45, -0.1, '', transform=adpt_est_ax.transAxes, size=14)

    # Limit values
    u_min = float("inf")
    u_max = -float("inf")

    max_time = round(time_values[len(time_values) - 1][0])

    # Does it plot better to pass in limits instead?
    for adpt_estimate in data:
        u_est_min = adpt_estimate.min()
        u_est_max = adpt_estimate.max()

        if u_est_min < u_min:
            u_min = u_est_min

        if u_est_max > u_max:
            u_max = u_est_max

    # Set up writer
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    print(np.shape(data))
    # Find grid dimensions
    n_xmax = len(set(mesh_nodes[0]))
    n_ymax = len(set(mesh_nodes[1]))

    x_i = min(mesh_nodes[0])
    x_f = max(mesh_nodes[0])

    y_i = min(mesh_nodes[1])
    y_f = max(mesh_nodes[0])

    scale = 0.10
    ax_adpt_est, aux_ax_adpt_est, work = setup_axes1(adpt_est, 341, [x_i, x_f], [y_i, y_f], scale)

    x_data = []
    y_data = []
    for location in region_dictionary.keys():
        x_data.append(mesh_dictionary[location][0])
        y_data.append(mesh_dictionary[location][1])

    small_plot = aux_ax_adpt_est.plot(x_data, y_data, 'o', markersize=2)

    # Reshape from array to grid for plotting
    X = np.reshape(np.array(mesh_nodes[0]), (n_xmax, n_ymax))
    Y = np.reshape(np.array(mesh_nodes[1]), (n_xmax, n_ymax))

    adpt_estimate = data[:,:,0]

    Z_est = np.reshape(np.array(adpt_estimate), (n_xmax, n_ymax))
    txt = 't = ' + str.strip(str(round(time_values[0][0], 2))) + ' secs'
    adpt_est_txt.set_text(txt)

    num_regions = max(region_dictionary.values())

    cmap = opa.get_cmap(num_regions + 1, 'tab20b')

    surface_plot = [adpt_est_ax.plot_surface(X.T, Y.T, np.flip(Z_est, 0), cmap=plt.cm.coolwarm, antialiased='False')]
    adpt_est_ax.set_xlim([x_i, x_f])
    adpt_est_ax.set_ylim([y_i, y_f])

    z_i, z_f = adpt_est_ax.get_zlim()
    adpt_est_ax.set_zlim([u_min, u_max])

    adpt_est_ax.invert_xaxis()

    adpt_est_ax.set_xlabel('x', size=14)
    adpt_est_ax.set_ylabel('y', size=14)
    adpt_est_ax.set_zlabel('u(x,y)', size=14)
    adpt_est.suptitle('Adaptive Estimate from Sensors for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center', verticalalignment='top')


    t0_time = round(time_values[t0][0])
    t_resample_val_time = round(time_values[t_resample][0])

    print("ADPT EST animations  ", str(max_time), " secs")
    est_ani = animation.FuncAnimation(adpt_est, updateData, frames = [i for i in range(0, np.shape(data)[2])[::granularity]], fargs = (small_plot, n_xmax, n_ymax, u_min, u_max, cmap), interval=50, repeat_delay=3000, blit=False)
    est_ani.save('adpt_est_'  + 'all_sensors_' + str(t0_time) + '_recompute_' +
                str(t_resample_val_time) + '_total_' + str(max_time) + 'sec.mp4', writer=writer)


# Plot relative error between FE discretized field and adaptive
def plot_relative_error_fe_adpt(fe_data, adpt_data, time_values, ada_vals, t0, t_resample):
    plt.gcf().clear()
    total_time = np.shape(fe_data)[0]
    ada_colors = []
    abs_error = []
    relative_error = []

    sorted_by_time = sorted(ada_vals.keys(), key=lambda tup: tup[0])
    cmap = opa.get_cmap(len(ada_vals), 'viridis')

    for i in range(0, total_time):
        reshaped_fe = fe_data[i]
        reshaped_adpt = adpt_data[:, :, i]

        diff = np.linalg.norm(reshaped_fe - reshaped_adpt)
        percent_diff = diff / np.linalg.norm(reshaped_fe)

        for idx, times in enumerate(sorted_by_time):
            if times[0] <= i <= times[1]:
                ada_colors.append(idx)
                break

        abs_error.append(diff)
        relative_error.append(percent_diff)

    t0_time = round(time_values[t0][0])
    t_resample_val_time = round(time_values[t_resample][0])

    max_time = round(time_values[len(time_values) - 1][0])

    plt.scatter(time_values, relative_error, c=ada_colors, marker='.',  cmap=cmap)
    plt.xlabel('Time (secs)', size=11)
    plt.ylabel('Relative error using Frobenius norm', size=11)
    plt.annotate('Global knowledge until ' + str(t0_time) + ' secs with \n recomputation of placement every '
              + str(t_resample_val_time) + ' secs', (0.5, 0), (0, -40), xycoords='axes fraction', textcoords='offset points', ha='center', va='top', size=12)
    plt.suptitle('Normwise Relative Error between FE Simulation \n and Adaptive Placement Algorithm for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center', verticalalignment='top')
    plt.savefig('relative_error_of_norms_fe_adpt_' + 'all_sensors_' + str(t0_time) + '_recompute_' +
                str(t_resample_val_time) + '_total_' + str(max_time) + 'secs.png', bbox_inches="tight")
    plt.gcf().clear()

    plt.scatter(time_values, abs_error, c=ada_colors, marker='.',  cmap=cmap)
    plt.xlabel('Time (secs)', size=11)
    plt.ylabel('Absolute error using Frobenius norm', size=11)
    plt.annotate('Global knowledge until ' + str(t0_time) + ' secs with \n recomputation of placement every '
              + str(t_resample_val_time) + ' secs', (0.5, 0), (0, -40), xycoords='axes fraction', textcoords='offset points', ha='center', va='top', size=12)
    plt.suptitle('Normwise Absolute Error between FE Simulation \n and Adaptive Placement Algorithm for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center', verticalalignment='top')
    plt.savefig('abs_error_of_norms_fe_adpt_' + 'all_sensors_' + str(t0_time) + '_recompute_' +
                str(t_resample_val_time) + '_total_' + str(max_time) + 'secs.png', bbox_inches="tight")
    plt.gcf().clear()


# Plot relative error between FE discretized field and adaptive
def plot_relative_error_fe_opt(fe_data, opt_data, time_values, ada_vals, t0=None, t_resample=None):
    plt.gcf().clear()
    total_time = np.shape(fe_data)[0]
    ada_colors = []
    abs_error = []
    relative_error = []

    sorted_by_time = sorted(ada_vals.keys(), key=lambda tup: tup[0])
    cmap = opa.get_cmap(len(ada_vals), 'viridis')

    for i in range(0, total_time):
        reshaped_fe = fe_data[i]
        reshaped_opt = opt_data[i]

        diff = np.linalg.norm(reshaped_fe - reshaped_opt)
        percent_diff = diff / np.linalg.norm(reshaped_fe)

        for idx, times in enumerate(sorted_by_time):
            if times[0] <= i <= times[1]:
                ada_colors.append(idx)
                break

        abs_error.append(diff)
        relative_error.append(percent_diff)

    max_time = round(time_values[len(time_values) - 1][0])
    plt.scatter(time_values, relative_error, c=ada_colors, marker='.',  cmap=cmap)
    plt.xlabel('Time (secs)', size=11)
    plt.ylabel('Relative error using Frobenius norm', size=11)
    plt.suptitle('Normwise Relative Error between FE Simulation \n and Optimal Placement Algorithm for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center', verticalalignment='top')
    if t0 and t_resample:
        t0_time = round(time_values[t0][0])
        t_resample_val_time = round(time_values[t_resample][0])
        fig_title = 'relative_error_of_norms_fe_opt_' + 'all_sensors_' + str(t0_time) + '_recompute_' +\
                    str(t_resample_val_time) + '_total_' + str(max_time) + 'secs.png'
    else:
        fig_title = 'relative_error_of_norms_fe_opt_' + str(max_time) + 'secs.png'
    plt.savefig(fig_title)
    plt.gcf().clear()

    plt.scatter(time_values, abs_error, c=ada_colors, marker='.',  cmap=cmap)
    plt.xlabel('Time (secs)', size=11)
    plt.ylabel('Absolute error using Frobenius norm', size=11)
    plt.suptitle('Normwise Absolute Error between FE Simulation \n and Optimal Placement Algorithm for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center', verticalalignment='top')
    if t0 and t_resample:
        fig_title = 'abs_error_of_norms_fe_opt_' + 'all_sensors_' + str(t0_time) + '_recompute_' +\
                    str(t_resample_val_time) + '_total_' + str(max_time) + 'secs.png'
    else:
        fig_title = 'abs_error_of_norms_fe_opt_' + str(max_time) + 'secs.png'
    plt.savefig(fig_title)
    plt.gcf().clear()


# Plot relative error between FE discretized field and adaptive
def plot_projection_relative_error_fe_adpt(fe_data, adpt_data, time_values, ada_vals, t0, t_resample):
    plt.gcf().clear()
    total_time = np.shape(fe_data)[0]
    ada_colors = []
    abs_error = []
    relative_error = []

    sorted_by_time = sorted(ada_vals.keys(), key=lambda tup: tup[0])
    cmap = opa.get_cmap(len(ada_vals), 'viridis')

    for i in range(0, total_time):
        reshaped_fe = fe_data[i]
        reshaped_adpt = adpt_data[:, :, i]

        diff = np.linalg.norm(reshaped_fe @ reshaped_fe.T - reshaped_adpt @ reshaped_adpt.T)
        percent_diff = diff / np.linalg.norm(reshaped_fe @ reshaped_fe.T)

        for idx, times in enumerate(sorted_by_time):
            if times[0] <= i <= times[1]:
                ada_colors.append(idx)
                break

        abs_error.append(diff)
        relative_error.append(percent_diff)

    t0_time = round(time_values[t0][0])
    t_resample_val_time = round(time_values[t_resample][0])

    max_time = round(time_values[len(time_values) - 1][0])

    plt.scatter(time_values, relative_error, c=ada_colors, marker='.',  cmap=cmap)
    plt.xlabel('Time (secs)', size=11)
    plt.ylabel('Relative error using Frobenius norm', size=11)
    plt.annotate('Global knowledge until ' + str(t0_time) + ' secs with \n recomputation of placement every '
              + str(t_resample_val_time) + ' secs', (0.5, 0), (0, -40), xycoords='axes fraction', textcoords='offset points', ha='center', va='top', size=12)
    plt.suptitle('Normwise Relative Error between FE Simulation \n and Adaptive Placement Algorithm for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center', verticalalignment='top')
    plt.savefig('projection_relative_error_of_norms_fe_adpt_' + 'all_sensors_' + str(t0_time) + '_recompute_' +
                str(t_resample_val_time) + '_total_' + str(max_time) + 'secs.png', bbox_inches="tight")
    plt.gcf().clear()

    plt.scatter(time_values, abs_error, c=ada_colors, marker='.',  cmap=cmap)
    plt.xlabel('Time (secs)', size=11)
    plt.ylabel('Absolute error using Frobenius norm', size=11)
    plt.annotate('Global knowledge until ' + str(t0_time) + ' secs with \n recomputation of placement every '
              + str(t_resample_val_time) + ' secs', (0.5, 0), (0, -40), xycoords='axes fraction', textcoords='offset points', ha='center', va='top', size=12)
    plt.suptitle('Normwise Absolute Error between FE Simulation \n and Adaptive Placement Algorithm for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center', verticalalignment='top')
    plt.savefig('projection_abs_error_of_norms_fe_adpt_' + 'all_sensors_' + str(t0_time) + '_recompute_' +
                str(t_resample_val_time) + '_total_' + str(max_time) + 'secs.png', bbox_inches="tight")
    plt.gcf().clear()


# Plot relative error between FE discretized field and adaptive
def plot_projection_relative_error_fe_opt(fe_data, opt_data, time_values, ada_vals, t0=None, t_resample=None):
    plt.gcf().clear()
    total_time = np.shape(fe_data)[0]
    ada_colors = []
    abs_error = []
    relative_error = []

    sorted_by_time = sorted(ada_vals.keys(), key=lambda tup: tup[0])
    cmap = opa.get_cmap(len(ada_vals), 'viridis')

    for i in range(0, total_time):
        reshaped_fe = fe_data[i]
        reshaped_opt = opt_data[i]

        diff = np.linalg.norm(reshaped_fe @ reshaped_fe.T - reshaped_opt @ reshaped_opt.T)
        percent_diff = diff / np.linalg.norm(reshaped_fe @ reshaped_fe.T)

        for idx, times in enumerate(sorted_by_time):
            if times[0] <= i <= times[1]:
                ada_colors.append(idx)
                break

        abs_error.append(diff)
        relative_error.append(percent_diff)

    max_time = round(time_values[len(time_values) - 1][0])
    plt.scatter(time_values, relative_error, c=ada_colors, marker='.',  cmap=cmap)
    plt.xlabel('Time (secs)', size=11)
    plt.ylabel('Relative error using Frobenius norm', size=11)
    plt.suptitle('Normwise Relative Error between FE Simulation \n and Optimal Placement Algorithm for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center', verticalalignment='top')
    if t0 and t_resample:
        t0_time = round(time_values[t0][0])
        t_resample_val_time = round(time_values[t_resample][0])
        fig_title = 'projection_relative_error_of_norms_fe_opt_' + 'all_sensors_' + str(t0_time) + '_recompute_' +\
                    str(t_resample_val_time) + '_total_' + str(max_time) + 'secs.png'
    else:
        fig_title = 'projection_relative_error_of_norms_fe_opt_' + str(max_time) + 'secs.png'
    plt.savefig(fig_title)
    plt.gcf().clear()

    plt.scatter(time_values, abs_error, c=ada_colors, marker='.',  cmap=cmap)
    plt.xlabel('Time (secs)', size=11)
    plt.ylabel('Absolute error using Frobenius norm', size=11)
    plt.suptitle('Normwise Absolute Error between FE Simulation \n and Optimal Placement Algorithm for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center', verticalalignment='top')
    if t0 and t_resample:
        fig_title = 'projection_abs_error_of_norms_fe_opt_' + 'all_sensors_' + str(t0_time) + '_recompute_' +\
                    str(t_resample_val_time) + '_total_' + str(max_time) + 'secs.png'
    else:
        fig_title = 'projection_abs_error_of_norms_fe_opt_' + str(max_time) + 'secs.png'
    plt.savefig(fig_title)
    plt.gcf().clear()


# Plot relative error between FE discretized field and adaptive
def plot_field_relative_error_fe_opt(fe_data, opt_data, time_values, ada_vals, t0=None, t_resample=None):
    plt.gcf().clear()
    total_time = np.shape(fe_data)[0]
    ada_colors = []
    abs_error = []
    relative_error = []

    sorted_by_time = sorted(ada_vals.keys(), key=lambda tup: tup[0])
    cmap = opa.get_cmap(len(ada_vals), 'viridis')

    for i in range(0, total_time):
        reshaped_fe = fe_data[i]
        reshaped_opt = opt_data[i]

        diff = np.linalg.norm(reshaped_fe @ reshaped_fe.T - reshaped_opt @ reshaped_opt.T)
        percent_diff = diff / np.linalg.norm(reshaped_fe @ reshaped_fe.T)

        for idx, times in enumerate(sorted_by_time):
            if times[0] <= i <= times[1]:
                ada_colors.append(idx)
                break

        abs_error.append(diff)
        relative_error.append(percent_diff)

    max_time = round(time_values[len(time_values) - 1][0])
    plt.scatter(time_values, relative_error, c=ada_colors, marker='.',  cmap=cmap)
    plt.xlabel('Time (secs)', size=11)
    plt.ylabel('Relative error using Frobenius norm', size=11)
    plt.suptitle('Normwise Relative Error between FE Simulation \n and Optimal Placement Algorithm for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center', verticalalignment='top')
    if t0 and t_resample:
        t0_time = round(time_values[t0][0])
        t_resample_val_time = round(time_values[t_resample][0])
        fig_title = 'projection_relative_error_of_norms_fe_opt_' + 'all_sensors_' + str(t0_time) + '_recompute_' +\
                    str(t_resample_val_time) + '_total_' + str(max_time) + 'secs.png'
    else:
        fig_title = 'projection_relative_error_of_norms_fe_opt_' + str(max_time) + 'secs.png'
    plt.savefig(fig_title)
    plt.gcf().clear()

    plt.scatter(time_values, abs_error, c=ada_colors, marker='.',  cmap=cmap)
    plt.xlabel('Time (secs)', size=11)
    plt.ylabel('Absolute error using Frobenius norm', size=11)
    plt.suptitle('Normwise Absolute Error between FE Simulation \n and Optimal Placement Algorithm for ' + str(max_time) + ' seconds', size=14, horizontalalignment='center', verticalalignment='top')
    if t0 and t_resample:
        fig_title = 'projection_abs_error_of_norms_fe_opt_' + 'all_sensors_' + str(t0_time) + '_recompute_' +\
                    str(t_resample_val_time) + '_total_' + str(max_time) + 'secs.png'
    else:
        fig_title = 'projection_abs_error_of_norms_fe_opt_' + str(max_time) + 'secs.png'
    plt.savefig(fig_title)
    plt.gcf().clear()

def create_rbf(real_data, region_dictionary, adas, t0_val, t_resample_val):
    total_time = np.shape(real_data)[1]
    relative_error = np.zeros_like(real_data)

    rbf_data = np.zeros_like(real_data)
    # CALCULATE RBF DATA
    for t in range(0, total_time):
        # Find corresponding robot location in time range
        for (times, ada_val) in adas.items():
            if times[0] <= t <= times[1]:
                locations = ada_val

        # Find indices of locations that correspond to robot locations
        indices = [k for k, v in region_dictionary.items() if v in locations]

        # Get data from real data
        rbf_vals = real_data[indices, t]

        # Interpolate missing values
        rbf = Rbf(indices, rbf_vals)
        di = rbf(np.array(list(region_dictionary.keys())))
        rbf_data[:, t] = di
    return rbf_data


def create_rbf_random(real_data, region_dictionary, adas, t0_val, t_resample_val):
    total_time = np.shape(real_data)[1]
    relative_error = np.zeros_like(real_data)

    rbf_data = np.zeros_like(real_data)
    # CALCULATE RBF DATA
    for t in range(0, total_time):
        # Find corresponding robot location in time range
        for (times, ada_val) in adas.items():
            if times[0] <= t <= times[1]:
                locations = ada_val

        # Find indices of locations that correspond to robot locations
        shuffle_indices = np.array(list(region_dictionary.keys()))
        random.shuffle(shuffle_indices)

        indices = [k for k, v in region_dictionary.items() if v in locations]
        shuffle_indices = shuffle_indices[0: len(indices)]
        # Get data from real data
        rbf_vals = real_data[shuffle_indices, t]

        # Interpolate missing values
        rbf = Rbf(shuffle_indices, rbf_vals)
        di = rbf(np.array(list(region_dictionary.keys())))
        rbf_data[:, t] = di
    return rbf_data


def create_centralized(region_dictionary, mesh_dictionary, nodeSoln, DA, t0_val, t_resample_val, num_sens, G_neighbors, rows_for_agent, k, num_orth_itrs, centers, dt):
    from optimal_placement_distributed import adaptation_scheme_2
    centr_data, adas, locs, paths = adaptation_scheme_2(region_dictionary, mesh_dictionary, nodeSoln, DA, t0_val, t_resample_val, num_sens, G_neighbors, rows_for_agent, k, num_orth_itrs, centers, dt)
    centr_data = np.reshape(centr_data, np.shape(nodeSoln))
    return centr_data, adas


def created_fixed(nodeSoln, ada, phi, set_of_phi, set_of_da, region_dictionary):
    from optimal_placement_algorithm_2d import test_state_estimation
    data_estimates = test_state_estimation(ada, phi, np.matrix(nodeSoln), set_of_phi, set_of_da, region_dictionary)
    optimal_data = np.zeros_like(nodeSoln)
    for i, x in enumerate(data_estimates):
        optimal_data[:, i] = np.asarray(x)[1,:,0]
    return optimal_data


def create_centralized_no_mesh(region_dictionary, mesh_dictionary, nodeSoln, t0_val, t_resample_val, num_sens, G_neighbors, rows_for_agent, k, num_orth_itrs, centers, dt, itr_num, epsilon):
    from optimal_placement_distributed import adaptation_scheme_2_no_mesh
    centr_data, adas, locs, paths = adaptation_scheme_2_no_mesh(region_dictionary, mesh_dictionary, nodeSoln, t0_val, t_resample_val, num_sens, G_neighbors, rows_for_agent, k, num_orth_itrs, centers, dt, itr_num, epsilon)
    centr_data = np.reshape(centr_data, np.shape(nodeSoln))
    return centr_data, adas


def created_fixed_no_mesh(nodeSoln, ada, phi, set_of_phi, region_dictionary):
    from optimal_placement_algorithm_2d import test_state_estimation_no_mesh
    itr_num = int(1e7)
    epsilon = 1e-7
    data_estimates = test_state_estimation_no_mesh(ada, phi, np.matrix(nodeSoln), set_of_phi, region_dictionary)
    optimal_data = np.zeros_like(nodeSoln)
    for i, x in enumerate(data_estimates):
        optimal_data[:, i] = np.asarray(x)[1,:,0]
    return optimal_data


def plot_error_bt_fixed_pos_and_real_field(fixed_pos_data, real_data, time_values, mesh_dictionary, region_dictionary, centers, t0_val, t_resample_val, ylim, markersize):
    total_time = np.shape(real_data)[1]
    relative_error = [0] * total_time

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - fixed_pos_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

    max_time = round(time_values[len(time_values) - 1][0])
    plt.scatter(time_values, relative_error, c='gold', marker='.', s=markersize)

    plt.xlabel('Time (secs)', size=11)
    plt.ylabel('Relative error using Frobenius norm', size=11)
    plt.ylim(top=ylim, bottom=0)
    plt.suptitle('Normwise Relative Error between Field Simulation \n and Field from Fixed Optimal Placement Algorithm', size=14, horizontalalignment='center', verticalalignment='top')

    fig_title = 'relative_error_of_norms_field_fixed_pos_' + 'train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs.png'

    plt.savefig(fig_title)
    plt.gcf().clear()


def plot_error_bt_centralized_and_real_field(centralized_data, real_data, time_values, mesh_dictionary, region_dictionary, adas, centers, t0_val, t_resample_val, ylim, markersize):
    total_time = np.shape(real_data)[1]
    relative_error = [0] * total_time

    switching_times = []
    sorted_times = sorted(adas.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times)-1):
        if not sorted(adas[sorted_times[i]]) == sorted(adas[sorted_times[i+1]]): # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i+1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - centralized_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))

    max_time = round(time_values[len(time_values) - 1][0])
    plt.scatter(time_values, relative_error, c='purple', marker='.', s=markersize)

    if switching_coords:
        switching_coords_x, switching_coords_y = zip(*switching_coords)
        for i, xc in enumerate(switching_coords_x):
            plt.axvline(xc, c='#C0C0C0', linestyle='--')

            if i == (len(switching_coords_x) - 1):
                plt.axvline(xc, c='#C0C0C0', linestyle='--', label = 'switching positions')

        plt.legend(loc='upper right')
    plt.xlabel('Time (secs)', size=11)
    plt.ylabel('Relative error using Frobenius norm', size=11)
    plt.ylim(top=ylim, bottom=0)
    plt.suptitle('Normwise Relative Error between Field Simulation \n and Field from Centralized Optimal Placement Algorithm', size=14, horizontalalignment='center', verticalalignment='top')

    fig_title = 'relative_error_of_norms_field_centr_' + 'train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs.png'

    plt.savefig(fig_title)
    plt.gcf().clear()


def plot_error_bt_distributed_and_real_field(distributed_data, real_data, time_values, mesh_dictionary, region_dictionary, adas, locs, paths, centers, t0_val, t_resample_val, ylim, markersize):
    total_time = np.shape(real_data)[1]
    relative_error = [0] * total_time

    switching_times = []
    sorted_times = sorted(adas.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times)-1):
        if not sorted(adas[sorted_times[i]]) == sorted(adas[sorted_times[i+1]]): # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i+1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distributed_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))

    max_time = round(time_values[len(time_values) - 1][0])
    plt.scatter(time_values, relative_error, c='red', marker='.', s=markersize)

    switching_coords_x, switching_coords_y = zip(*switching_coords)
    for i, xc in enumerate(switching_coords_x):
        plt.axvline(xc, c='#C0C0C0', linestyle='--')

        if i == (len(switching_coords_x) - 1):
            plt.axvline(xc, c='#C0C0C0', linestyle='--', label = 'switching positions')

    plt.legend(loc='upper right')
    plt.xlabel('Time (secs)', size=11)
    plt.ylabel('Relative error using Frobenius norm', size=11)
    plt.ylim(top=ylim, bottom=0)
    plt.suptitle('Normwise Relative Error between Field Simulation \n and Field from Distributed Optimal Placement Algorithm', size=14, horizontalalignment='center', verticalalignment='top')

    fig_title = 'relative_error_of_norms_field_distr_' + 'train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs.png'

    plt.savefig(fig_title)
    plt.gcf().clear()


def plot_error_bt_rbf_and_real_field(real_data, time_values, mesh_dictionary, region_dictionary, adas, locs, paths, centers, t0_val, t_resample_val, ylim, markersize, file_data=None):
    total_time = np.shape(real_data)[1]
    relative_error = [0] * total_time

    if file_data is not None:
        rbf_data = file_data
    else:
        rbf_data = np.zeros_like(real_data)
        # CALCULATE RBF DATA
        for t in range(0, total_time):
            # Find corresponding robot location in time range
            for (times, ada_val) in adas.items():
                if times[0] <= t <= times[1]:
                    locations = ada_val

            # Find indices of locations that correspond to robot locations
            indices = [k for k, v in region_dictionary.items() if v in locations]

            # Get data from real data
            rbf_vals = real_data[indices, t]

            # Interpolate missing values
            rbf = Rbf(indices, rbf_vals)
            di = rbf(np.array(list(region_dictionary.keys())))
            rbf_data[:, t] = di

    switching_times = []
    sorted_times = sorted(adas.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times)-1):
        if not sorted(adas[sorted_times[i]]) == sorted(adas[sorted_times[i+1]]): # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i+1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - rbf_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))

    max_time = round(time_values[len(time_values) - 1][0])
    plt.scatter(time_values, relative_error, c='blue', marker='.', s=markersize)

    switching_coords_x, switching_coords_y = zip(*switching_coords)
    for i, xc in enumerate(switching_coords_x):
        plt.axvline(xc, c='#C0C0C0', linestyle='--')

        if i == (len(switching_coords_x) - 1):
            plt.axvline(xc, c='#C0C0C0', linestyle='--', label = 'switching positions')

    plt.legend(loc='upper right')
    plt.xlabel('Time (secs)', size=11)
    plt.ylabel('Relative error using Frobenius norm', size=11)
    plt.ylim(top=ylim, bottom=0)
    plt.suptitle('Normwise Relative Error between Field Simulation \n and Field from Radial Basis Function with Optimal Locations', size=14, horizontalalignment='center', verticalalignment='top')

    fig_title = 'relative_error_of_norms_field_rbf_' + 'train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs.png'

    plt.savefig(fig_title)
    plt.gcf().clear()


def plot_error_bt_random_rbf_and_real_field(real_data, time_values, mesh_dictionary, region_dictionary, adas, locs, paths, centers, t0_val, t_resample_val, ylim, markersize, file_data=None):
    total_time = np.shape(real_data)[1]
    relative_error = [0] * total_time
    if file_data is not None:
        rbf_data = file_data
    else:
        rbf_data = np.zeros_like(real_data)
        # CALCULATE RBF DATA
        for t in range(0, total_time):
            # Find corresponding robot location in time range
            for (times, ada_val) in adas.items():
                if times[0] <= t <= times[1]:
                    locations = ada_val

            # Find indices of locations that correspond to robot locations
            shuffle_indices = np.array(list(region_dictionary.keys()))
            random.shuffle(shuffle_indices)

            indices = [k for k, v in region_dictionary.items() if v in locations]
            shuffle_indices = shuffle_indices[0: len(indices)]
            # Get data from real data
            rbf_vals = real_data[shuffle_indices, t]

            # Interpolate missing values
            rbf = Rbf(shuffle_indices, rbf_vals)
            di = rbf(np.array(list(region_dictionary.keys())))
            rbf_data[:, t] = di

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - rbf_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

    max_time = round(time_values[len(time_values) - 1][0])
    plt.scatter(time_values, relative_error, c='green', marker='.', s=markersize)

    plt.xlabel('Time (secs)', size=11)
    plt.ylabel('Relative error using Frobenius norm', size=11)
    plt.ylim(top=ylim, bottom=0)
    plt.suptitle('Normwise Relative Error between Field Simulation \n and Field from Radial Basis Function with Random Locations', size=14, horizontalalignment='center', verticalalignment='top')

    fig_title = 'relative_error_of_norms_field_rbf_random_' + 'train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs.png'

    plt.savefig(fig_title)
    plt.gcf().clear()


def plots_bt_estimates_and_real_field_single_plot(real_data, distr_data, rbf_data, rbf_random_data, centr_data, fixed_data, adas_distr, adas_centr, ada_fixed, time_values, t0_val, t_resample_val):
    markersize = 2
    f, (ax2, ax3, ax1, ax4) = plt.subplots(1, 4)
    total_time = np.shape(real_data)[1]

    relative_error = [0] * total_time

    switching_times = []
    sorted_times = sorted(adas_distr.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times)-1):
        if not sorted(adas_distr[sorted_times[i]]) == sorted(adas_distr[sorted_times[i+1]]): # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i+1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))

    max_time = round(time_values[len(time_values) - 1][0])
    ax1.scatter(time_values, relative_error, c='red', marker='.', s=markersize)

    switching_coords_x, switching_coords_y = zip(*switching_coords)
    for i, xc in enumerate(switching_coords_x):
        ax1.axvline(xc, c='#C0C0C0', linestyle='--')

    ax1.set_title('Distributed optimal placement', size=11)

    switching_times = []
    sorted_times = sorted(adas_distr.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times) - 1):
        if not sorted(adas_distr[sorted_times[i]]) == sorted(
                adas_distr[sorted_times[i + 1]]):  # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i + 1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - rbf_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))

    ax2.scatter(time_values, relative_error, c='blue', marker='.', s=markersize)

    switching_coords_x, switching_coords_y = zip(*switching_coords)
    for i, xc in enumerate(switching_coords_x):
        ax2.axvline(xc, c='#C0C0C0', linestyle='--')

    ax2.set_title('RBF with pre-selected location', size=11)

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - rbf_random_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

    max_time = round(time_values[len(time_values) - 1][0])
    ax3.scatter(time_values, relative_error, c='green', marker='.', s=markersize)
    ax3.set_title('RBF with random points', size=11)

    switching_times = []
    sorted_times = sorted(adas_centr.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times) - 1):
        if not sorted(adas_centr[sorted_times[i]]) == sorted(
                adas_centr[sorted_times[i + 1]]):  # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i + 1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - centr_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))

    max_time = round(time_values[len(time_values) - 1][0])
    ax4.scatter(time_values, relative_error, c='purple', marker='.', s=markersize)

    if switching_coords:
        switching_coords_x, switching_coords_y = zip(*switching_coords)
        for i, xc in enumerate(switching_coords_x):
            ax4.axvline(xc, c='#C0C0C0', linestyle='--')

    custom_label = mlines.Line2D([], [], color='#C0C0C0', linestyle='--', label='switching positions')
    ax4.set_title('Centralized optimal placement', size=11)
    ax4.legend(handles=[custom_label], bbox_to_anchor=(0.45, -0.25, 1., .102), loc=2,
           ncol=2, borderaxespad=0.)

    f.text(0.5, 0.96, 'Normwise Relative Error between Field Simulation \n and Various Field Estimation Algorithms', size=14, horizontalalignment='center', verticalalignment='top')
    f.subplots_adjust(top=0.83)
    f.text(0.5, 0.02, 'Time (secs)', ha='center', size=11)
    f.text(0.04, 0.5, 'Relative error using Frobenius norm', va='center', rotation='vertical', size=11)

    fig_title = 'single_plot_abs_error_of_regions_field_' + 'train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs.png'

    plt.savefig(fig_title)
    plt.gcf().clear()


def plots_bt_estimates_and_real_field_panel(real_data, distr_data, rbf_data, rbf_random_data, centr_data, fixed_data, adas_distr, adas_centr, ada_fixed, time_values, t0_val, t_resample_val):
    markersize = 2
    f, ((ax2, ax3), (ax1, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    total_time = np.shape(real_data)[1]

    relative_error = [0] * total_time

    switching_times = []
    sorted_times = sorted(adas_distr.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times)-1):
        if not sorted(adas_distr[sorted_times[i]]) == sorted(adas_distr[sorted_times[i+1]]): # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i+1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))

    max_time = round(time_values[len(time_values) - 1][0])
    ax1.scatter(time_values, relative_error, c='red', marker='.', s=markersize)

    switching_coords_x, switching_coords_y = zip(*switching_coords)
    for i, xc in enumerate(switching_coords_x):
        ax1.axvline(xc, c='#C0C0C0', linestyle='--')

    ax1.set_title('Distributed optimal placement', size=11)

    switching_times = []
    sorted_times = sorted(adas_distr.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times) - 1):
        if not sorted(adas_distr[sorted_times[i]]) == sorted(
                adas_distr[sorted_times[i + 1]]):  # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i + 1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - rbf_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))

    ax2.scatter(time_values, relative_error, c='blue', marker='.', s=markersize)

    switching_coords_x, switching_coords_y = zip(*switching_coords)
    for i, xc in enumerate(switching_coords_x):
        ax2.axvline(xc, c='#C0C0C0', linestyle='--')

    ax2.set_title('RBF with pre-selected location', size=11)

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - rbf_random_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

    max_time = round(time_values[len(time_values) - 1][0])
    ax3.scatter(time_values, relative_error, c='green', marker='.', s=markersize)
    ax3.set_title('RBF with random points', size=11)

    switching_times = []
    sorted_times = sorted(adas_centr.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times) - 1):
        if not sorted(adas_centr[sorted_times[i]]) == sorted(
                adas_centr[sorted_times[i + 1]]):  # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i + 1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - centr_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))

    max_time = round(time_values[len(time_values) - 1][0])
    ax4.scatter(time_values, relative_error, c='purple', marker='.', s=markersize)

    if switching_coords:
        switching_coords_x, switching_coords_y = zip(*switching_coords)
        for i, xc in enumerate(switching_coords_x):
            ax4.axvline(xc, c='#C0C0C0', linestyle='--')

    custom_label = mlines.Line2D([], [], color='#C0C0C0', linestyle='--', label='switching positions')
    ax4.set_title('Centralized optimal placement', size=11)
    ax4.legend(handles=[custom_label], bbox_to_anchor=(0.45, -0.25, 1., .102), loc=2,
           ncol=2, borderaxespad=0.)

    f.text(0.5, 0.96, 'Normwise Relative Error between Field Simulation \n and Various Field Estimation Algorithms', size=14, horizontalalignment='center', verticalalignment='top')
    f.subplots_adjust(top=0.83)
    f.text(0.5, 0.02, 'Time (secs)', ha='center', size=11)
    f.text(0.04, 0.5, 'Relative error using Frobenius norm', va='center', rotation='vertical', size=11)

    fig_title = 'panel_abs_error_of_regions_field_' + 'train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs.png'

    plt.savefig(fig_title)
    plt.gcf().clear()


def plots_bt_estimates_and_real_field_panel_with_baseline(real_data, distr_data, rbf_data, rbf_random_data, centr_data, fixed_data, adas_distr, adas_centr, ada_fixed, time_values, t0_val, t_resample_val):
    markersize = 2
    f, ((ax2, ax3), (ax1, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    total_time = np.shape(real_data)[1]

    relative_error = [0] * total_time

    fixed_plcmt_relative_error = [0] * total_time
    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - fixed_data[:, t]) / np.linalg.norm(real_data[:, t])
        fixed_plcmt_relative_error[t] = e

    switching_times = []
    sorted_times = sorted(adas_distr.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times)-1):
        if not sorted(adas_distr[sorted_times[i]]) == sorted(adas_distr[sorted_times[i+1]]): # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i+1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))

    max_time = round(time_values[len(time_values) - 1][0])
    ax1.scatter(time_values, fixed_plcmt_relative_error, c='orange', marker='.', s=markersize)
    ax1.scatter(time_values, relative_error, c='red', marker='.', s=markersize)

    switching_coords_x, switching_coords_y = zip(*switching_coords)
    for i, xc in enumerate(switching_coords_x):
        ax1.axvline(xc, c='#C0C0C0', linestyle='--')

    ax1.set_title('Distributed optimal placement', size=11)

    switching_times = []
    sorted_times = sorted(adas_distr.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times) - 1):
        if not sorted(adas_distr[sorted_times[i]]) == sorted(
                adas_distr[sorted_times[i + 1]]):  # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i + 1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - rbf_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))

    ax2.scatter(time_values, fixed_plcmt_relative_error, c='orange', marker='.', s=markersize)
    ax2.scatter(time_values, relative_error, c='blue', marker='.', s=markersize)
    ax2.xaxis.set_tick_params(labelbottom=True)

    switching_coords_x, switching_coords_y = zip(*switching_coords)
    for i, xc in enumerate(switching_coords_x):
        ax2.axvline(xc, c='#C0C0C0', linestyle='--')

    ax2.set_title('RBF with pre-selected location', size=11)

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - rbf_random_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

    max_time = round(time_values[len(time_values) - 1][0])
    baseline_label = ax3.scatter(time_values, fixed_plcmt_relative_error, c='orange', marker='.', s=markersize, label='Fixed optimal placement')
    ax3.scatter(time_values, relative_error, c='green', marker='.', s=markersize)
    ax3.xaxis.set_tick_params(labelbottom=True)
    ax3.yaxis.set_tick_params(labelleft=True)

    ax3.set_title('RBF with random points', size=11)

    switching_times = []
    sorted_times = sorted(adas_centr.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times) - 1):
        if not sorted(adas_centr[sorted_times[i]]) == sorted(
                adas_centr[sorted_times[i + 1]]):  # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i + 1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - centr_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))

    max_time = round(time_values[len(time_values) - 1][0])
    ax4.scatter(time_values, fixed_plcmt_relative_error, c='orange', marker='.', s=markersize, label='Fixed optimal placement')
    ax4.scatter(time_values, relative_error, c='purple', marker='.', s=markersize)
    ax4.xaxis.set_tick_params(labelbottom=True)

    if switching_coords:
        switching_coords_x, switching_coords_y = zip(*switching_coords)
        for i, xc in enumerate(switching_coords_x):
            ax4.axvline(xc, c='#C0C0C0', linestyle='--')

    custom_label = mlines.Line2D([], [], color='#C0C0C0', linestyle='--', label='switching positions')
    # baseline_label = mpatches.Circle((0.5, 0.5), 0.25, facecolor='orange', label='baseline')

    ax4.set_title('Centralized optimal placement', size=11)
    lgnd = ax4.legend(handles=[baseline_label, custom_label], bbox_to_anchor=(0.25, -0.27, 1., .102), loc=2,
           ncol=1, borderaxespad=0.)

    lgnd.legendHandles[0]._sizes = [30]

    f.text(0.5, 0.96, 'Normwise Relative Error between Field Simulation \n and Various Field Estimation Algorithms', size=14, horizontalalignment='center', verticalalignment='top')
    f.subplots_adjust(top=0.83, bottom=0.15, hspace=0.3)
    f.text(0.5, 0.06, 'Time (secs)', ha='center', size=11)
    f.text(0.04, 0.5, 'Relative error using Frobenius norm', va='center', rotation='vertical', size=11)

    fig_title = 'panel_abs_error_of_regions_field_' + 'train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs.png'

    plt.savefig(fig_title)
    plt.gcf().clear()


def plot_error_by_region_distr_and_real_field(distributed_data, real_data, time_values, mesh_dictionary, region_dictionary, adas, locs, paths, centers, t0_val, t_resample_val):
    total_time = np.shape(real_data)[1]
    relative_error = np.zeros_like(distributed_data)

    switching_times = []
    sorted_times = sorted(adas.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times)-1):
        if not sorted(adas[sorted_times[i]]) == sorted(adas[sorted_times[i+1]]): # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i+1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = abs(real_data[:, t] - distributed_data[:, t])
        relative_error[:, t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))
    relative_error_time_avg = np.average(relative_error, axis=1)

    x_vals = [0] * np.shape(relative_error_time_avg)[0]
    y_vals = [0] * np.shape(relative_error_time_avg)[0]
    for i in range(0, np.shape(relative_error_time_avg)[0]):
        x_vals[i] = mesh_dictionary[i][0]
        y_vals[i] = mesh_dictionary[i][1]


    max_time = round(time_values[len(time_values) - 1][0])
    sct = plt.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap='Blues', antialiased='False', vmin=0.001, vmax=255)
    cb = plt.colorbar(sct, extend='neither', orientation='vertical')
    cb.set_label('Mean-absolute error at spatial points')

    # plt.legend(loc='lower right')
    # plt.xlabel('Time (secs)', size=11)
    # plt.ylabel('Relative error a', size=11)
    plt.suptitle('Point-wise Error between Field Simulation \n and Field from Distributed Optimal Placement Algorithm', size=14, horizontalalignment='center', verticalalignment='top')

    fig_title = 'abs_error_of_regions_field_distr_' + 'train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs.png'

    plt.savefig(fig_title)
    plt.gcf().clear()


def plot_error_by_region_rbf_and_real_field(real_data, time_values, mesh_dictionary, region_dictionary, adas, locs, paths, centers, t0_val, t_resample_val, file_data=None):
    total_time = np.shape(real_data)[1]
    relative_error = np.zeros_like(real_data)

    total_time = np.shape(real_data)[1]
    relative_error = np.zeros_like(real_data)
    if file_data is not None:
        rbf_data = file_data
    else:
        rbf_data = np.zeros_like(real_data)
        # CALCULATE RBF DATA
        for t in range(0, total_time):
            # Find corresponding robot location in time range
            for (times, ada_val) in adas.items():
                if times[0] <= t <= times[1]:
                    locations = ada_val

            # Find indices of locations that correspond to robot locations
            indices = [k for k, v in region_dictionary.items() if v in locations]

            # Get data from real data
            rbf_vals = real_data[indices, t]

            # Interpolate missing values
            rbf = Rbf(indices, rbf_vals)
            di = rbf(np.array(list(region_dictionary.keys())))
            rbf_data[:, t] = di

    switching_times = []
    sorted_times = sorted(adas.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times)-1):
        if not sorted(adas[sorted_times[i]]) == sorted(adas[sorted_times[i+1]]): # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i+1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = abs(real_data[:, t] - rbf_data[:, t])
        relative_error[:, t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))
    relative_error_time_avg = np.average(relative_error, axis=1)

    x_vals = [0] * np.shape(relative_error_time_avg)[0]
    y_vals = [0] * np.shape(relative_error_time_avg)[0]
    for i in range(0, np.shape(relative_error_time_avg)[0]):
        x_vals[i] = mesh_dictionary[i][0]
        y_vals[i] = mesh_dictionary[i][1]


    max_time = round(time_values[len(time_values) - 1][0])
    sct = plt.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap='Blues', antialiased='False', vmin=0.001, vmax=255)
    cb = plt.colorbar(sct, extend='neither', orientation='vertical')
    cb.set_label('Mean-absolute error at spatial points')

    # plt.legend(loc='lower right')
    # plt.xlabel('Time (secs)', size=11)
    # plt.ylabel('Relative error a', size=11)
    plt.suptitle('Point-wise Error between Field Simulation \n and Field from Radial Basis Function with Optimal Locations', size=14, horizontalalignment='center', verticalalignment='top')

    fig_title = 'abs_error_of_regions_field_rbf_' + 'train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs.png'

    plt.savefig(fig_title)
    plt.gcf().clear()


def plot_error_by_region_rbf_random_and_real_field(real_data, time_values, mesh_dictionary, region_dictionary, adas, locs, paths, centers, t0_val, t_resample_val, file_data=None):
    total_time = np.shape(real_data)[1]
    relative_error = np.zeros_like(real_data)
    if file_data is not None:
        rbf_data = file_data
    else:
        rbf_data = np.zeros_like(real_data)
        # CALCULATE RBF DATA
        for t in range(0, total_time):
            # Find corresponding robot location in time range
            for (times, ada_val) in adas.items():
                if times[0] <= t <= times[1]:
                    locations = ada_val

            # Find indices of locations that correspond to robot locations
            shuffle_indices = np.array(list(region_dictionary.keys()))
            random.shuffle(shuffle_indices)

            indices = [k for k, v in region_dictionary.items() if v in locations]
            shuffle_indices = shuffle_indices[0: len(indices)]
            # Get data from real data
            rbf_vals = real_data[shuffle_indices, t]

            # Interpolate missing values
            rbf = Rbf(shuffle_indices, rbf_vals)
            di = rbf(np.array(list(region_dictionary.keys())))
            rbf_data[:, t] = di

    switching_times = []
    sorted_times = sorted(adas.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times)-1):
        if not sorted(adas[sorted_times[i]]) == sorted(adas[sorted_times[i+1]]): # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i+1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = abs(real_data[:, t] - rbf_data[:, t])
        relative_error[:, t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))
    relative_error_time_avg = np.average(relative_error, axis=1)

    x_vals = [0] * np.shape(relative_error_time_avg)[0]
    y_vals = [0] * np.shape(relative_error_time_avg)[0]
    for i in range(0, np.shape(relative_error_time_avg)[0]):
        x_vals[i] = mesh_dictionary[i][0]
        y_vals[i] = mesh_dictionary[i][1]


    max_time = round(time_values[len(time_values) - 1][0])
    sct = plt.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap='Blues', antialiased='False', vmin=0.001, vmax=255)
    cb = plt.colorbar(sct, extend='neither', orientation='vertical')
    cb.set_label('Mean-absolute error at spatial points')

    # plt.legend(loc='lower right')
    # plt.xlabel('Time (secs)', size=11)
    # plt.ylabel('Relative error a', size=11)
    plt.suptitle('Point-wise Error between Field Simulation \n and Field from Radial Basis Function with Random Locations', size=14, horizontalalignment='center', verticalalignment='top')

    fig_title = 'abs_error_of_regions_field_rbf_random_' + 'train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs.png'

    plt.savefig(fig_title)
    plt.gcf().clear()


def plot_snapshot_error_by_region_distr_and_real_field(distributed_data, real_data, time_values, mesh_dictionary, region_dictionary, adas, locs, paths, centers, t0_val, t_resample_val):
    total_time = np.shape(real_data)[1]
    relative_error = np.zeros_like(distributed_data)

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distributed_data[:, t])
        relative_error[:, t] = e

    x_vals = [0] * np.shape(relative_error)[0]
    y_vals = [0] * np.shape(relative_error)[0]
    for i in range(0, np.shape(relative_error)[0]):
        x_vals[i] = mesh_dictionary[i][0]
        y_vals[i] = mesh_dictionary[i][1]

    for t in range(0, total_time):
        max_time = round(time_values[len(time_values) - 1][0])
        sct = plt.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error[:,t], cmap='Blues', antialiased='False', vmin=0.001, vmax=255)
        cb = plt.colorbar(sct, extend='neither', orientation='vertical')
        cb.set_label('Absolute error at spatial points')

        txt = 't = ' + str.strip(str(round(time_values[t][0], 2))) + ' secs'
        plt.text(0.45, -0.1, txt, size=14)

        # plt.legend(loc='lower right')
        # plt.xlabel('Time (secs)', size=11)
        # plt.ylabel('Relative error a', size=11)
        plt.suptitle('Point-wise Error between Field Simulation \n and Field from Distributed Optimal Placement Algorithm', size=14, horizontalalignment='center', verticalalignment='top')

        fig_title = 'abs_snapshot_error_of_regions_field_distr_' + 'train_' + str(t0_val) + '_resample_' +\
                        str(t_resample_val) + 'secs.png'

        plt.savefig(fig_title)
        plt.gcf().clear()


def plot_snapshot_error_comparison_by_region_distr_and_real_field(real_data, distr_data, rbf_data, rbf_random_data, centr_data, fixed_data, adas_distr, adas_centr, ada_fixed, time_values, t0_val, t_resample_val, mesh_dictionary, t_ss):
    markersize = 1
    f, ((ax5, ax2, ax3), (ax1, ax4, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')
    total_time = np.shape(real_data)[1]

    x_vals = [0] * np.shape(real_data)[0]
    y_vals = [0] * np.shape(real_data)[0]
    for i in range(0, np.shape(real_data)[0]):
        x_vals[i] = mesh_dictionary[i][0]
        y_vals[i] = mesh_dictionary[i][1]

    interval = np.hstack([np.linspace(0, 0.35), np.linspace(0.65, 1)])
    colors = plt.cm.PRGn_r(interval)
    new_cmap = LinearSegmentedColormap.from_list('name', colors)
    new_cmap.set_under('white')

    ax5.set_title('Concentration values \nof dye', size=11)
    ax5.scatter(np.array(y_vals), 599-np.array(x_vals), c=real_data[:,t_ss], cmap=new_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)

    interval = np.linspace(0.25, 1.0)
    colors = plt.cm.Blues(interval)
    e_cmap = LinearSegmentedColormap.from_list('name', colors)
    e_cmap.set_under('white')

    e = abs(real_data[:, t_ss] - distr_data[:, t_ss])
    dist_data_error = e

    ax1.set_title('Distributed optimal \n placement', size=11)
    ax1.scatter(np.array(y_vals), 599-np.array(x_vals), c=dist_data_error, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)

    e = abs(real_data[:, t_ss] - rbf_data[:, t_ss])
    rbf_data_error = e

    ax2.scatter(np.array(y_vals), 599-np.array(x_vals), c=rbf_data_error, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    ax2.set_title('RBF with pre-\nselected location', size=11)

    e = abs(real_data[:, t_ss] - rbf_random_data[:, t_ss])
    rbf_random_data_error = e

    ax3.scatter(np.array(y_vals), 599-np.array(x_vals), c=rbf_random_data_error, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    ax3.set_title('RBF with random \npoints', size=11)

    e = abs(real_data[:, t_ss] - centr_data[:, t_ss])
    centr_data_error = e

    sct = ax4.scatter(np.array(y_vals), 599-np.array(x_vals), c=centr_data_error, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    ax4.set_title('Centralized optimal \n placement', size=11)

    e = abs(real_data[:, t_ss] - fixed_data[:, t_ss])
    fixed_data_error = e

    sct = ax6.scatter(np.array(y_vals), 599-np.array(x_vals), c=fixed_data_error, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    ax6.set_title('Fixed optimal \n placement', size=11)

    title = f.text(0.5, 0.96, 'Comparison of Point-wise Error between Field Simulation \n and Various Field Estimation Algorithm',
                 size=14, horizontalalignment='center', verticalalignment='top')

    f.subplots_adjust(top=0.78, right=0.85, hspace=0.3)
    txt = 't = ' + str.strip(str(round(time_values[t_ss][0], 2))) + ' secs'
    time_subtitle = f.text(0.5, 0.02, txt, ha='center', size=11)
    cbar_ax = f.add_axes([0.87, 0.10, 0.02, 0.7])
    cb = f.colorbar(sct, extend='neither', orientation='vertical', cax=cbar_ax, pad=0.25)
    cb.set_label('Absolute error at spatial points', )

    fig_title = 'abs_snapshot_error_of_regions_field_distr_time_' + str(t_ss) + '_train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs.png'

    plt.savefig(fig_title, bbox_extra_artists=(cbar_ax,title,time_subtitle,))
    plt.gcf().clear()


def plot_error_comparison_by_region_distr_and_real_field(real_data, distr_data, rbf_data, rbf_random_data, centr_data, fixed_data, adas_distr, adas_centr, ada_fixed, time_values, t0_val, t_resample_val, mesh_dictionary, t_ss):
    markersize = 1
    f, ((ax2, ax3, ax5), (ax1, ax4, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')
    # f, ((ax2, ax3), (ax1, ax6)) = plt.subplots(2, 2, sharex='col', sharey='row')

    total_time = np.shape(real_data)[1]
    relative_error = np.zeros_like(real_data)

    x_vals = [0] * np.shape(real_data)[0]
    y_vals = [0] * np.shape(real_data)[0]
    for i in range(0, np.shape(real_data)[0]):
        x_vals[i] = mesh_dictionary[i][0]
        y_vals[i] = mesh_dictionary[i][1]

    # interval = np.hstack([np.linspace(0, 0.35), np.linspace(0.65, 1)])
    # colors = plt.cm.PRGn_r(interval)
    # new_cmap = LinearSegmentedColormap.from_list('name', colors)
    # new_cmap.set_under('white')
    #
    # ax5.set_title('Real field', size=11)
    # ax5.scatter(np.array(y_vals), 599-np.array(x_vals), c=real_data[:,t_ss], cmap=new_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)

    ax5.axis('off')
    interval = np.linspace(0.25, 1.0)
    colors = plt.cm.Oranges(interval)
    e_cmap = LinearSegmentedColormap.from_list('name', colors)
    e_cmap.set_under('white')

    # e_cmap = 'Purples'

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax1.set_title('Distributed optimal \n placement', size=11)
    ax1.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    ax1.set_aspect(abs(x1 - x0) / abs(y1 - y0))

    for t in range(0, total_time):
        e = abs(real_data[:, t] - rbf_data[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax2.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    ax2.set_title('RBF with pre-\nselected location', size=11)
    x0, x1 = ax2.get_xlim()
    y0, y1 = ax2.get_ylim()
    ax2.set_aspect(abs(x1 - x0) / abs(y1 - y0))

    for t in range(0, total_time):
        e = abs(real_data[:, t_ss] - rbf_random_data[:, t_ss])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax3.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    ax3.set_title('RBF with random \npoints', size=11)
    x0, x1 = ax3.get_xlim()
    y0, y1 = ax3.get_ylim()
    ax3.set_aspect(abs(x1 - x0) / abs(y1 - y0))

    for t in range(0, total_time):
        e = abs(real_data[:, t_ss] - centr_data[:, t_ss])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax4.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    ax4.set_title('Centralized optimal \n placement', size=11)
    x0, x1 = ax4.get_xlim()
    y0, y1 = ax4.get_ylim()
    ax4.set_aspect(abs(x1 - x0) / abs(y1 - y0))

    for t in range(0, total_time):
        e = abs(real_data[:, t_ss] - fixed_data[:, t_ss])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    sct = ax6.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    ax6.set_title('Fixed optimal \n placement', size=11)
    x0, x1 = ax6.get_xlim()
    y0, y1 = ax6.get_ylim()
    ax6.set_aspect(abs(x1 - x0) / abs(y1 - y0))

    title = f.text(0.5, 0.96, 'Comparison of Point-wise Mean Error between Field Simulation \n and Various Field Estimation Algorithm',
                 size=14, horizontalalignment='center', verticalalignment='top')

    f.subplots_adjust(top=0.78, right=0.85, hspace=0.3, wspace=-0.1)
    cbar_ax = f.add_axes([0.87, 0.10, 0.02, 0.7])
    cb = f.colorbar(sct, extend='neither', orientation='vertical', cax=cbar_ax, pad=0.25)
    cb.set_label('Mean absolute error at spatial points', )

    fig_title = 'mean_error_of_regions_field_distr_train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs.png'

    plt.savefig(fig_title, bbox_extra_artists=(cbar_ax,title,))
    plt.gcf().clear()


def paper_plots_bt_estimates_and_real_field_panel_with_baseline(real_data, distr_data, rbf_data, rbf_random_data, centr_data, fixed_data, adas_distr, adas_centr, ada_fixed, time_values, t0_val, t_resample_val):
    markersize = 2
    label_pad = -0.5
    label_size = 10
    f, ((ax2, ax3), (ax1, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    total_time = np.shape(real_data)[1]

    relative_error = [0] * total_time

    fixed_color = 'black'
    fixed_plcmt_relative_error = [0] * total_time
    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - fixed_data[:, t]) / np.linalg.norm(real_data[:, t])
        fixed_plcmt_relative_error[t] = e

    switching_times = []
    sorted_times = sorted(adas_distr.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times)-1):
        if not sorted(adas_distr[sorted_times[i]]) == sorted(adas_distr[sorted_times[i+1]]): # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i+1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))

    max_time = round(time_values[len(time_values) - 1][0])
    ax1.scatter(time_values, fixed_plcmt_relative_error, c=fixed_color, marker='.', s=markersize)
    ax1.scatter(time_values, relative_error, c='red', marker='.', s=markersize)

    switching_coords_x, switching_coords_y = zip(*switching_coords)
    for i, xc in enumerate(switching_coords_x):
        ax1.axvline(xc, c='#C0C0C0', linestyle='--')

    #'Distributed optimal placement'
    ax1.set_xlabel('(c)', size=label_size, labelpad=label_pad)

    switching_times = []
    sorted_times = sorted(adas_distr.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times) - 1):
        if not sorted(adas_distr[sorted_times[i]]) == sorted(
                adas_distr[sorted_times[i + 1]]):  # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i + 1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - rbf_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))

    ax2.scatter(time_values, fixed_plcmt_relative_error, c=fixed_color, marker='.', s=markersize)
    ax2.scatter(time_values, relative_error, c='blue', marker='.', s=markersize)
    ax2.xaxis.set_tick_params(labelbottom=True, labelsize=label_size)

    switching_coords_x, switching_coords_y = zip(*switching_coords)
    for i, xc in enumerate(switching_coords_x):
        ax2.axvline(xc, c='#C0C0C0', linestyle='--')

    # 'RBF with pre-selected location'
    ax2.set_xlabel('(a)', size=label_size, labelpad=label_pad)

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - rbf_random_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

    max_time = round(time_values[len(time_values) - 1][0])
    baseline_label = ax3.scatter(time_values, fixed_plcmt_relative_error, c=fixed_color, marker='.', s=markersize, label='Optimal placement')
    ax3.scatter(time_values, relative_error, c='green', marker='.', s=markersize)
    ax3.xaxis.set_tick_params(labelbottom=True, labelsize=label_size)
    ax3.yaxis.set_tick_params(labelleft=True, labelsize=label_size)

    # 'RBF with random points'
    ax3.set_xlabel('(b)', size=label_size, labelpad=label_pad)

    switching_times = []
    sorted_times = sorted(adas_centr.keys(), key=lambda x: x[0])  # Sort keys of times
    for i in range(0, len(sorted_times) - 1):
        if not sorted(adas_centr[sorted_times[i]]) == sorted(
                adas_centr[sorted_times[i + 1]]):  # See if ada values switch between consecutive time intervals
            switching_times.append(sorted_times[i + 1][0])  # Store time of switch

    switching_coords = []
    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - centr_data[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error[t] = e

        if t in switching_times:
            switching_coords.append((time_values[t][0], e))

    max_time = round(time_values[len(time_values) - 1][0])
    ax4.scatter(time_values, fixed_plcmt_relative_error, c=fixed_color, marker='.', s=markersize, label='Optimal placement')
    ax4.scatter(time_values, relative_error, c='purple', marker='.', s=markersize)
    ax4.xaxis.set_tick_params(labelbottom=True, labelsize=label_size)
    ax4.yaxis.set_tick_params(labelleft=True, labelsize=label_size)

    # 'Centralized optimal placement'
    ax4.set_xlabel('(d)', size=label_size, labelpad=label_pad)

    if switching_coords:
        switching_coords_x, switching_coords_y = zip(*switching_coords)
        for i, xc in enumerate(switching_coords_x):
            ax4.axvline(xc, c='#C0C0C0', linestyle='--')

    custom_label = mlines.Line2D([], [], color='#C0C0C0', linestyle='--', label='switching positions')
    # baseline_label = mpatches.Circle((0.5, 0.5), 0.25, facecolor='orange', label='baseline')

    lgnd = ax3.legend(handles=[baseline_label, custom_label], bbox_to_anchor=(0.25, 1.05, 1., .102), loc=4,
           ncol=1, borderaxespad=0.)

    ax1.set_xlim((0,60))
    ax3.set_xlim((0,60))

    lgnd.legendHandles[0]._sizes = [30]

    # f.text(0.5, 0.96, 'Normwise Relative Error between Field Simulation \n and Various Field Estimation Algorithms', size=14, horizontalalignment='center', verticalalignment='top')
    f.subplots_adjust(bottom=0.15, hspace=0.35)
    f.text(0.5, 0.04, 'Time (secs)', ha='center', size=11)
    f.text(0.04, 0.5, 'Relative error using Frobenius norm', va='center', rotation='vertical', size=11)

    fig_title = 'panel_abs_error_of_regions_field_' + 'train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs_paper.png'

    plt.savefig(fig_title)
    plt.gcf().clear()


def paper_plot_error_comparison_by_region_distr_and_real_field(real_data, distr_data, rbf_data, rbf_random_data, centr_data, fixed_data, adas_distr, adas_centr, ada_fixed, time_values, t0_val, t_resample_val, mesh_dictionary, t_ss):
    markersize = 1
    label_pad = -0.5
    label_size = 10

    f, ((ax2, ax3, ax5), (ax1, ax4, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')
    # f, ((ax2, ax3), (ax1, ax6)) = plt.subplots(2, 2, sharex='col', sharey='row')
    total_time = np.shape(real_data)[1]
    relative_error = np.zeros_like(real_data)

    x_vals = [0] * np.shape(real_data)[0]
    y_vals = [0] * np.shape(real_data)[0]
    for i in range(0, np.shape(real_data)[0]):
        x_vals[i] = mesh_dictionary[i][0]
        y_vals[i] = mesh_dictionary[i][1]

    # interval = np.hstack([np.linspace(0, 0.35), np.linspace(0.65, 1)])
    # colors = plt.cm.PRGn_r(interval)
    # new_cmap = LinearSegmentedColormap.from_list('name', colors)
    # new_cmap.set_under('white')
    #
    # ax5.set_title('Real field', size=11)
    # ax5.scatter(np.array(y_vals), 599-np.array(x_vals), c=real_data[:,t_ss], cmap=new_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)

    ax5.axis('off')
    interval = np.linspace(0.25, 1.0)
    colors = plt.cm.Oranges(interval)
    e_cmap = LinearSegmentedColormap.from_list('name', colors)
    e_cmap.set_under('white')

    # e_cmap = 'Purples'

    for t in range(0, total_time):
        e = abs(real_data[:, t_ss] - fixed_data[:, t_ss])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    # 'Distributed optimal \n placement'
    ax1.set_xlabel('x', labelpad=label_pad-2)
    ax1.set_ylabel('y', labelpad=label_pad)
    ax1.text(0.5, -0.32, "(c)", size=label_size, ha="center",
             transform=ax1.transAxes)

    ax1.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    ax1.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax1.set_xlim((0, 600))
    ax1.set_ylim((0, 600))
    ax1.set_xticks([0, 300, 600])
    ax1.set_yticks([0, 300, 600])

    tick_labels = [item.get_text() for item in ax1.get_xticklabels()]
    new_labels = np.linspace(0, 1, 3, endpoint=True)
    new_labels = [str(round(l, 2)) for l in new_labels]
    ax1.set_xticklabels(new_labels)
    ax1.set_yticklabels(new_labels)

    for t in range(0, total_time):
        e = abs(real_data[:, t] - rbf_data[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax2.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # 'RBF with pre-\nselected location'
    # x0, x1 = ax2.get_xlim()
    # y0, y1 = ax2.get_ylim()
    ax2.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    # ax2.set_xlabel('(a)', size=label_size, labelpad=label_pad+10)
    # ax2.set_xlabel('x', labelpad=label_pad-2)

    ax2.set_ylabel('y', labelpad=label_pad)
    ax2.text(0.5, -0.2, "(a)", size=label_size, ha="center",
             transform=ax2.transAxes)

    ax2.set_xlim((0, 600))
    ax2.set_ylim((0, 600))
    ax2.set_xticks([0, 300, 600])
    ax2.set_yticks([0, 300, 600])
    # ax2.xaxis.set_tick_params(labelbottom=True, labelsize=label_size)
    # ax2.yaxis.set_tick_params(labelleft=True, labelsize=label_size)

    ax2.set_xticklabels(new_labels)
    ax2.set_yticklabels(new_labels)

    for t in range(0, total_time):
        e = abs(real_data[:, t_ss] - rbf_random_data[:, t_ss])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax3.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # 'RBF with random \npoints'
    # x0, x1 = ax3.get_xlim()
    # y0, y1 = ax3.get_ylim()
    ax3.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax3.set_xlim((0, 600))
    ax3.set_ylim((0, 600))
    # ax3.xaxis.set_tick_params(labelbottom=True, labelsize=label_size)
    # ax3.yaxis.set_tick_params(labelleft=True, labelsize=label_size)
    ax3.set_xticks([0, 300, 600])
    ax3.set_yticks([0, 300, 600])
    ax3.set_xticklabels(new_labels)
    ax3.set_yticklabels(new_labels)
    # ax3.set_xlabel('(b)', size=label_size, labelpad=label_pad+10)
    # ax3.set_xlabel('x', labelpad=label_pad-2)
    ax3.text(0.5, -0.2, "(b)", size=label_size, ha="center",
             transform=ax3.transAxes)

    for t in range(0, total_time):
        e = abs(real_data[:, t_ss] - centr_data[:, t_ss])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax4.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # 'Centralized optimal \n placement'
    # x0, x1 = ax4.get_xlim()
    # y0, y1 = ax4.get_ylim()
    ax4.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax4.set_xlim((0, 600))
    ax4.set_ylim((0, 600))
    ax4.set_xticks([0, 300, 600])
    ax4.set_yticks([0, 300, 600])
    ax4.set_xticklabels(new_labels)
    ax4.set_yticklabels(new_labels)

    # ax4.set_xlabel('(d)', size=label_size, labelpad=label_pad)
    ax4.set_xlabel('x', labelpad=label_pad-2)
    ax4.text(0.5, -0.32, "(d)", size=label_size, ha="center",
             transform=ax4.transAxes)

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    sct = ax6.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # set_title('Fixed optimal \n placement', size=11
    ax6.set_xlabel('(f)', size=label_size, labelpad=label_pad)
    ax6.set_xlabel('x', labelpad=label_pad-2)
    ax6.text(0.5, -0.32, "(f)", size=label_size, ha="center",
             transform=ax6.transAxes)

    # x0, x1 = ax6.get_xlim()
    # y0, y1 = ax6.get_ylim()
    ax6.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax6.set_xlim((0, 600))
    ax6.set_ylim((0, 600))
    ax6.set_xticks([0, 300, 600])
    ax6.set_yticks([0, 300, 600])
    ax6.set_xticklabels(new_labels)
    ax6.set_yticklabels(new_labels)

    # title = f.text(0.5, 0.96, 'Comparison of Point-wise Mean Error between Field Simulation \n and Various Field Estimation Algorithm',
    #              size=14, horizontalalignment='center', verticalalignment='top')

    f.subplots_adjust(hspace=0.1, wspace=0.2)
    cbar_ax = f.add_axes([0.65, 0.83, 0.3, 0.02])
    cb = f.colorbar(sct, extend='neither', orientation='horizontal', cax=cbar_ax, pad=0.25)
    cb.set_label('Mean absolute error \n at spatial points', )

    fig_title = 'mean_error_of_regions_field_distr_train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs_paper.png'

    plt.savefig(fig_title, bbox_extra_artists=(cbar_ax,))
    plt.gcf().clear()


def paper_plot_error_comparison_by_region_distr_and_real_field_single_row(real_data, distr_data, rbf_data, rbf_random_data, centr_data, fixed_data, adas_distr, adas_centr, ada_fixed, time_values, t0_val, t_resample_val, mesh_dictionary, t_ss):
    markersize = 1
    label_pad = -0.5
    label_size = 10

    f, (ax2, ax3, ax1, ax4, ax6) = plt.subplots(1, 5, sharey='row')
    # f, ((ax2, ax3), (ax1, ax6)) = plt.subplots(2, 2, sharex='col', sharey='row')
    total_time = np.shape(real_data)[1]
    relative_error = np.zeros_like(real_data)

    x_vals = [0] * np.shape(real_data)[0]
    y_vals = [0] * np.shape(real_data)[0]
    for i in range(0, np.shape(real_data)[0]):
        x_vals[i] = mesh_dictionary[i][0]
        y_vals[i] = mesh_dictionary[i][1]

    # interval = np.hstack([np.linspace(0, 0.35), np.linspace(0.65, 1)])
    # colors = plt.cm.PRGn_r(interval)
    # new_cmap = LinearSegmentedColormap.from_list('name', colors)
    # new_cmap.set_under('white')
    #
    # ax5.set_title('Real field', size=11)
    # ax5.scatter(np.array(y_vals), 599-np.array(x_vals), c=real_data[:,t_ss], cmap=new_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)

    interval = np.linspace(0.25, 1.0)
    colors = plt.cm.Oranges(interval)
    e_cmap = LinearSegmentedColormap.from_list('name', colors)
    e_cmap.set_under('white')

    # e_cmap = 'Purples'

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    # 'Distributed optimal \n placement'
    ax1.set_xlabel('(c)', size=label_size, labelpad=label_pad)

    ax1.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    ax1.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax1.set_xlim((0, 600))
    ax1.set_ylim((0, 600))
    ax1.set_xticks([0, 300, 600])
    ax1.set_yticks([0, 300, 600])

    tick_labels = [item.get_text() for item in ax1.get_xticklabels()]
    new_labels = np.linspace(0, 1, 3, endpoint=True)
    new_labels = [str(round(l, 2)) for l in new_labels]
    ax1.set_xticklabels(new_labels)
    ax1.set_yticklabels(new_labels)

    for t in range(0, total_time):
        e = abs(real_data[:, t] - rbf_data[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax2.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # 'RBF with pre-\nselected location'
    # x0, x1 = ax2.get_xlim()
    # y0, y1 = ax2.get_ylim()
    ax2.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax2.set_xlabel('(a)', size=label_size, labelpad=label_pad)
    ax2.set_xlim((0, 600))
    ax2.set_ylim((0, 600))
    ax2.set_xticks([0, 300, 600])
    ax2.set_yticks([0, 300, 600])
    # ax2.xaxis.set_tick_params(labelbottom=True, labelsize=label_size)
    # ax2.yaxis.set_tick_params(labelleft=True, labelsize=label_size)

    ax2.set_xticklabels(new_labels)
    ax2.set_yticklabels(new_labels)

    for t in range(0, total_time):
        e = abs(real_data[:, t_ss] - rbf_random_data[:, t_ss])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax3.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # 'RBF with random \npoints'
    # x0, x1 = ax3.get_xlim()
    # y0, y1 = ax3.get_ylim()
    ax3.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax3.set_xlim((0, 600))
    ax3.set_ylim((0, 600))
    # ax3.xaxis.set_tick_params(labelbottom=True, labelsize=label_size)
    # ax3.yaxis.set_tick_params(labelleft=True, labelsize=label_size)
    ax3.set_xticks([0, 300, 600])
    ax3.set_yticks([0, 300, 600])
    ax3.set_xticklabels(new_labels)
    ax3.set_yticklabels(new_labels)
    ax3.set_xlabel('(b)', size=label_size, labelpad=label_pad)

    for t in range(0, total_time):
        e = abs(real_data[:, t_ss] - centr_data[:, t_ss])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax4.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # 'Centralized optimal \n placement'
    # x0, x1 = ax4.get_xlim()
    # y0, y1 = ax4.get_ylim()
    ax4.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax4.set_xlim((0, 600))
    ax4.set_ylim((0, 600))
    ax4.set_xticks([0, 300, 600])
    ax4.set_yticks([0, 300, 600])
    ax4.set_xticklabels(new_labels)
    ax4.set_yticklabels(new_labels)

    ax4.set_xlabel('(d)', size=label_size, labelpad=label_pad)

    for t in range(0, total_time):
        e = abs(real_data[:, t_ss] - fixed_data[:, t_ss])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    sct = ax6.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # set_title('Fixed optimal \n placement', size=11
    ax6.set_xlabel('(f)', size=label_size, labelpad=label_pad)

    # x0, x1 = ax6.get_xlim()
    # y0, y1 = ax6.get_ylim()
    ax6.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax6.set_xlim((0, 600))
    ax6.set_ylim((0, 600))
    ax6.set_xticks([0, 300, 600])
    ax6.set_yticks([0, 300, 600])
    ax6.set_xticklabels(new_labels)
    ax6.set_yticklabels(new_labels)

    # title = f.text(0.5, 0.96, 'Comparison of Point-wise Mean Error between Field Simulation \n and Various Field Estimation Algorithm',
    #              size=14, horizontalalignment='center', verticalalignment='top')

    f.subplots_adjust(hspace=0.1, wspace=0.35)
    cbar_ax = f.add_axes([0.35, 0.65, 0.30, 0.02])
    cb = f.colorbar(sct, extend='neither', orientation='horizontal', cax=cbar_ax, pad=0.25)
    cbar_ax.set_title('Mean absolute error at spatial points', size=label_size)

    fig_title = 'mean_error_of_regions_field_distr_train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs_single_row_paper.png'

    plt.savefig(fig_title, bbox_extra_artists=(cbar_ax,))
    plt.gcf().clear()


def paper_plot_snapshot_error_comparison_train_sample_by_region(real_data, distr_data_small_train_disc_0, distr_data_large_train_disc_0, distr_data_small_train_disc_1, distr_data_large_train_disc_1, mesh_dictionary, time_values, t0_val_0, t_resample_val_0, t0_val_1, t_resample_val_1):
    markersize = 1
    label_pad = -0.5
    label_size = 10

    # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)

    total_time = np.shape(real_data)[1]
    relative_error = np.zeros_like(real_data)

    relative_error_with_time = [0] * total_time

    x_vals = [0] * np.shape(real_data)[0]
    y_vals = [0] * np.shape(real_data)[0]
    for i in range(0, np.shape(real_data)[0]):
        x_vals[i] = mesh_dictionary[i][0]
        y_vals[i] = mesh_dictionary[i][1]

    interval = np.linspace(0.25, 1.0)
    colors = plt.cm.Oranges(interval)
    e_cmap = LinearSegmentedColormap.from_list('name', colors)
    e_cmap.set_under('white')

    # e_cmap = 'Purples'

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_small_train_disc_0[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax1.set_xlabel('(a)', size=label_size, labelpad=label_pad)

    cnc = ax1.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    tick_labels = [item.get_text() for item in ax1.get_yticklabels()]
    new_labels = np.linspace(0.0, 1.0, 3, endpoint=True)
    new_labels = [str(round(l, 2)) for l in new_labels]
    ax1.set_xticklabels(new_labels)
    ax1.set_yticklabels(new_labels)

    ax1.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax1.set_xlim((0, 600))
    ax1.set_ylim((0, 600))
    ax1.set_xticks([0, 300, 600])
    ax1.set_yticks([0, 300, 600])

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_small_train_disc_0[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    ax5.scatter(time_values, relative_error_with_time, c='blue', marker='.', s=markersize)


    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_large_train_disc_0[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax2.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # ax2.set_title('Distributed optimal \n placement at t = ' + str(t_ss0), size=11)
    ax2.set_xlabel('(b)', size=label_size, labelpad=label_pad)

    #
    # x0, x1 = ax2.get_xlim()
    # y0, y1 = ax2.get_ylim()
    # tick_labels = [item.get_text() for item in ax2.get_yticklabels()]
    # new_labels = np.linspace(0.0, 1.0, 3, endpoint=True)
    # new_labels = [str(round(l, 2)) for l in new_labels]
    # ax2.set_xticklabels(new_labels)
    # ax2.set_yticklabels(new_labels)

    ax2.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax2.set_xlim((0, 600))
    ax2.set_ylim((0, 600))
    ax2.set_xticks([0, 300, 600])
    ax2.set_xticklabels(new_labels)
    ax2.set_yticklabels([])

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_large_train_disc_0[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    ax6.scatter(time_values, relative_error_with_time, c='blue', marker='.', s=markersize)

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_small_train_disc_1[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax3.set_xlabel('(c)', size=label_size, labelpad=label_pad)
    ax3.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)

    ax3.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax3.set_xlim((0, 600))
    ax3.set_ylim((0, 600))
    ax3.set_xticks([0, 300, 600])
    ax3.set_xticklabels(new_labels)
    ax3.set_yticklabels([])

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_small_train_disc_1[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    ax7.scatter(time_values, relative_error_with_time, c='blue', marker='.', s=markersize)

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_large_train_disc_1[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    sct = ax4.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # ax4.set_title('Distributed optimal \n placement at t = ' + str(t_ss1), size=11)
    ax4.set_xlabel('(d)', size=label_size, labelpad=label_pad)
    #
    ax4.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax4.set_xlim((0, 600))
    ax4.set_ylim((0, 600))
    ax4.set_xticks([0, 300, 600])
    ax4.set_xticklabels(new_labels)
    ax4.set_yticklabels([])

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_large_train_disc_1[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    ax8.scatter(time_values, relative_error_with_time, c='blue', marker='.', s=markersize)

    # title = f.text(0.5, 0.96, 'Comparison of Point-wise Error between Field Simulation \n and Various Field Estimation Algorithm',
    #              size=14, horizontalalignment='center', verticalalignment='top')

    f.subplots_adjust(hspace=0.01)
    # txt = 't = ' + str.strip(str(round(time_values[t_ss0][0], 2))) + ' secs'
    # time_subtitle = f.text(0.30, 0.25, txt, ha='center', size=11)
    #
    # txt1 = 't = ' + str.strip(str(round(time_values[t_ss1][0], 2))) + ' secs'
    # time1_subtitle = f.text(0.70, 0.25, txt1, ha='center', size=11)

    cbar_ax = f.add_axes([0.55, 0.68,  0.3, 0.02])
    cb = f.colorbar(sct, extend='neither', orientation='horizontal', cax=cbar_ax)
    cbar_ax.set_title('Absolute error at spatial points', size=label_size)

    # cbar_ax2 = f.add_axes([0.15, 0.68,  0.3, 0.02])
    # cb2 = f.colorbar(cnc, extend='neither', orientation='horizontal', cax=cbar_ax2)
    # cbar_ax2.set_title('Concentration value at field', size=label_size)

    fig_title = 'paper_compare_train_disc_' + '_train_' + str(t0_val_0) + '_resample_' +\
                    str(t_resample_val_0) + '_train_' + str(t0_val_1) + '_resample_' +\
                    str(t_resample_val_1) + 'secs.png'

    plt.savefig(fig_title, bbox_extra_artists=(cbar_ax,))
    plt.gcf().clear()


def paper_plot_snapshot_error_comparison_train_sample_by_region2(real_data, distr_data_small_train_disc_0, distr_data_large_train_disc_0, distr_data_small_train_disc_1, distr_data_large_train_disc_1, mesh_dictionary, time_values, t0_val_0, t_resample_val_0, t0_val_1, t_resample_val_1):
    markersize = 1
    label_pad = -0.5
    label_size = 10

    # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

    # f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
    f = plt.figure(figsize=(9, 7))

    from matplotlib import gridspec
    gs = gridspec.GridSpec(35, 1)

    # gs1 = gridspec.GridSpec(1, 4)
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 29, subplot_spec=gs[2:12], hspace=0.35, wspace=0.5)

    ax1 = plt.subplot(gs1[0, 1:7])
    ax2 = plt.subplot(gs1[0, 8:14])
    ax3 = plt.subplot(gs1[0, 15:21])
    ax4 = plt.subplot(gs1[0, 22:28])

    # gs2 = gridspec.GridSpec(2, 4)
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[15:35], hspace=0.35, wspace=0.4)

    ax5 = plt.subplot(gs2[0, 0:2])
    ax6 = plt.subplot(gs2[0, 2:4])
    ax7 = plt.subplot(gs2[1, 0:2])
    ax8 = plt.subplot(gs2[1, 2:4])

    total_time = np.shape(real_data)[1]
    relative_error = np.zeros_like(real_data)

    relative_error_with_time = [0] * total_time

    x_vals = [0] * np.shape(real_data)[0]
    y_vals = [0] * np.shape(real_data)[0]
    for i in range(0, np.shape(real_data)[0]):
        x_vals[i] = mesh_dictionary[i][0]
        y_vals[i] = mesh_dictionary[i][1]

    interval = np.linspace(0.25, 1.0)
    colors = plt.cm.Oranges(interval)
    e_cmap = LinearSegmentedColormap.from_list('name', colors)
    e_cmap.set_under('white')

    # e_cmap = 'Purples'

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_small_train_disc_0[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    cnc = ax1.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    tick_labels = [item.get_text() for item in ax1.get_yticklabels()]
    new_labels = np.linspace(0.0, 1.0, 3, endpoint=True)
    new_labels = [str(round(l, 2)) for l in new_labels]
    ax1.set_xticklabels(new_labels)
    ax1.set_yticklabels(new_labels)

    ax1.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax1.set_xlim((0, 600))
    ax1.set_ylim((0, 600))
    ax1.set_xticks([0, 300, 600])
    ax1.set_yticks([0, 300, 600])
    ax1.set_ylabel('y', labelpad=label_pad+1)

    ax1.set_xlabel('x', labelpad=label_pad-2)
    ax1.text(0.5, -0.35, "(a)", size=label_size, ha="center",
             transform=ax1.transAxes)

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_small_train_disc_0[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    ax5.scatter(time_values, relative_error_with_time, c='blue', marker='.', s=markersize)
    ax5.set_xlabel('(e)', size=label_size, labelpad=label_pad)

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_large_train_disc_0[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax2.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # ax2.set_title('Distributed optimal \n placement at t = ' + str(t_ss0), size=11)
    # ax2.set_xlabel('(b)', size=label_size, labelpad=label_pad)
    ax2.set_xlabel('x', labelpad=label_pad-2)
    ax2.text(0.5, -0.35, "(b)", size=label_size, ha="center",
             transform=ax2.transAxes)
    #
    # x0, x1 = ax2.get_xlim()
    # y0, y1 = ax2.get_ylim()
    # tick_labels = [item.get_text() for item in ax2.get_yticklabels()]
    # new_labels = np.linspace(0.0, 1.0, 3, endpoint=True)
    # new_labels = [str(round(l, 2)) for l in new_labels]
    # ax2.set_xticklabels(new_labels)
    # ax2.set_yticklabels(new_labels)

    ax2.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax2.set_xlim((0, 600))
    ax2.set_ylim((0, 600))
    ax2.set_xticks([0, 300, 600])
    ax2.set_xticklabels(new_labels)
    ax2.set_yticklabels([])

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_large_train_disc_0[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    ax6.scatter(time_values, relative_error_with_time, c='blue', marker='.', s=markersize)
    ax6.set_xlabel('(f)', size=label_size, labelpad=label_pad)

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_small_train_disc_1[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax3.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    ax3.set_xlabel('x', labelpad=label_pad-2)
    ax3.text(0.5, -0.35, "(c)", size=label_size, ha="center",
             transform=ax3.transAxes)

    ax3.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax3.set_xlim((0, 600))
    ax3.set_ylim((0, 600))
    ax3.set_xticks([0, 300, 600])
    ax3.set_xticklabels(new_labels)
    ax3.set_yticklabels([])

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_small_train_disc_1[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    ax7.scatter(time_values, relative_error_with_time, c='blue', marker='.', s=markersize)
    ax7.set_xlabel('(g)', size=label_size, labelpad=label_pad)

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_large_train_disc_1[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    sct = ax4.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # ax4.set_title('Distributed optimal \n placement at t = ' + str(t_ss1), size=11)
    ax4.set_xlabel('x', labelpad=label_pad-2)
    ax4.text(0.5, -0.35, "(d)", size=label_size, ha="center",
             transform=ax4.transAxes)
    #
    ax4.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax4.set_xlim((0, 600))
    ax4.set_ylim((0, 600))
    ax4.set_xticks([0, 300, 600])
    ax4.set_xticklabels(new_labels)
    ax4.set_yticklabels([])

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_large_train_disc_1[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    ax8.scatter(time_values, relative_error_with_time, c='blue', marker='.', s=markersize)
    ax8.set_xlabel('(h)', size=label_size, labelpad=label_pad)

    y_min = 0
    y_max = 0

    error_over_time_axes = [ax5, ax6, ax7, ax8]
    for ax in error_over_time_axes:
        _, y_upper = ax.get_ylim()
        if y_upper > y_max:
            y_max = y_upper
    for ax in error_over_time_axes:
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.linspace(0, 1, 6, endpoint=True))

        ax.set_xlim(0, 60)

    f.text(0.5, 0.03, 'Time (secs)', ha='center', size=11)
    f.text(0.06, 0.35, 'Relative error using Frobenius norm', va='center', rotation='vertical', size=11)
    gs.update(top=0.965)

    # title = f.text(0.5, 0.96, 'Comparison of Point-wise Error between Field Simulation \n and Various Field Estimation Algorithm',
    #              size=14, horizontalalignment='center', verticalalignment='top')

    # f.subplots_adjust(hspace=0.01)
    # txt = 't = ' + str.strip(str(round(time_values[t_ss0][0], 2))) + ' secs'
    # time_subtitle = f.text(0.30, 0.25, txt, ha='center', size=11)
    #
    # txt1 = 't = ' + str.strip(str(round(time_values[t_ss1][0], 2))) + ' secs'
    # time1_subtitle = f.text(0.70, 0.25, txt1, ha='center', size=11)

    cbar_ax = f.add_axes([0.35, 0.94, 0.3, 0.02])
    cb = f.colorbar(sct, extend='neither', orientation='horizontal', cax=cbar_ax)
    cbar_ax.set_title('Absolute error at spatial points', size=label_size)

    # cbar_ax2 = f.add_axes([0.15, 0.68,  0.3, 0.02])
    # cb2 = f.colorbar(cnc, extend='neither', orientation='horizontal', cax=cbar_ax2)
    # cbar_ax2.set_title('Concentration value at field', size=label_size)

    fig_title = 'paper_compare_train_disc_' + '_train_' + str(t0_val_0) + '_resample_' +\
                    str(t_resample_val_0) + '_train_' + str(t0_val_1) + '_resample_' +\
                    str(t_resample_val_1) + 'secs.png'

    plt.savefig(fig_title)

    # plt.savefig(fig_title, bbox_extra_artists=(cbar_ax,))
    plt.gcf().clear()


def paper_plot_snapshot_error_comparison_by_region_distr_and_real_field(real_data, distr_data, rbf_data, rbf_random_data, centr_data, fixed_data, adas_distr, adas_centr, ada_fixed, time_values, t0_val, t_resample_val, mesh_dictionary, t_ss0, t_ss1):
    markersize = 1
    label_pad = -0.5
    label_size = 10

    # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    total_time = np.shape(real_data)[1]

    x_vals = [0] * np.shape(real_data)[0]
    y_vals = [0] * np.shape(real_data)[0]
    for i in range(0, np.shape(real_data)[0]):
        x_vals[i] = mesh_dictionary[i][0]
        y_vals[i] = mesh_dictionary[i][1]

    interval = np.hstack([np.linspace(0, 0.35), np.linspace(0.65, 1)])
    colors = plt.cm.PRGn_r(interval)
    new_cmap = LinearSegmentedColormap.from_list('name', colors)
    new_cmap.set_under('white')

    # ax1.set_title('Concentration values \nof dye at t = ' + str(t_ss0), size=11)
    ax1.set_ylabel('y', labelpad=label_pad)
    ax1.set_xlabel('x', labelpad=label_pad-2)
    ax1.text(0.5, -0.43, "(a)", size=label_size, ha="center",
             transform=ax1.transAxes)

    cnc = ax1.scatter(np.array(y_vals), 599-np.array(x_vals), c=real_data[:,t_ss0], cmap=new_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)

    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    ax1.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax1.set_xlim((0, 600))
    ax1.set_ylim((0, 600))
    ax1.set_xticks([0, 300, 600])
    ax1.set_yticks([0, 300, 600])

    #
    tick_labels = [item.get_text() for item in ax1.get_yticklabels()]
    new_labels = np.linspace(0.0, 1.0, 3, endpoint=True)
    new_labels = [str(round(l, 2)) for l in new_labels]
    ax1.set_xticklabels(new_labels)
    ax1.set_yticklabels(new_labels)

    interval = np.linspace(0.25, 1.0)
    colors = plt.cm.Blues(interval)
    e_cmap = LinearSegmentedColormap.from_list('name', colors)
    e_cmap.set_under('white')

    # ax3.set_title('Concentration values \nof dye at t = ' + str(t_ss1), size=11)
    ax3.set_xlabel('x', labelpad=label_pad-2)
    ax3.text(0.5, -0.43, "(c)", size=label_size, ha="center",
             transform=ax3.transAxes)

    ax3.scatter(np.array(y_vals), 599-np.array(x_vals), c=real_data[:,t_ss1], cmap=new_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)

    # tick_labels = [item.get_text() for item in ax3.get_yticklabels()]
    # new_labels = np.linspace(0.0, 1.0, 4, endpoint=True)
    # new_labels = [str(round(l, 2)) for l in new_labels]
    # ax1.set_xticklabels(new_labels)
    # ax1.set_yticklabels(new_labels)

    ax3.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax3.set_xlim((0, 600))
    ax3.set_ylim((0, 600))
    ax3.set_xticks([0, 300, 600])
    ax3.set_xticklabels(new_labels)
    ax3.set_yticklabels([])

    e = abs(real_data[:, t_ss0] - distr_data[:, t_ss0])
    dist_data_error = e

    ax2.scatter(np.array(y_vals), 599-np.array(x_vals), c=dist_data_error, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # ax2.set_title('Distributed optimal \n placement at t = ' + str(t_ss0), size=11)
    ax2.set_xlabel('x', labelpad=label_pad-2)
    ax2.text(0.5, -0.43, "(b)", size=label_size, ha="center",
             transform=ax2.transAxes)

    ax2.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax2.set_xlim((0, 600))
    ax2.set_ylim((0, 600))
    ax2.set_xticks([0, 300, 600])
    ax2.set_xticklabels(new_labels)
    ax2.set_yticklabels([])

    e = abs(real_data[:, t_ss1] - distr_data[:, t_ss1])
    dist_data_error = e

    sct = ax4.scatter(np.array(y_vals), 599-np.array(x_vals), c=dist_data_error, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # ax4.set_title('Distributed optimal \n placement at t = ' + str(t_ss1), size=11)
    ax4.set_xlabel('x', labelpad=label_pad-2)
    ax4.text(0.5, -0.43, "(d)", size=label_size, ha="center",
             transform=ax4.transAxes)
    #
    ax4.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax4.set_xlim((0, 600))
    ax4.set_ylim((0, 600))
    ax4.set_xticks([0, 300, 600])
    ax4.set_xticklabels(new_labels)
    ax4.set_yticklabels([])

    # title = f.text(0.5, 0.96, 'Comparison of Point-wise Error between Field Simulation \n and Various Field Estimation Algorithm',
    #              size=14, horizontalalignment='center', verticalalignment='top')

    f.subplots_adjust(hspace=0.01)
    txt = 't = ' + str.strip(str(round(time_values[t_ss0][0], 2))) + ' secs'
    time_subtitle = f.text(0.30, 0.25, txt, ha='center', size=11)

    txt1 = 't = ' + str.strip(str(round(time_values[t_ss1][0], 2))) + ' secs'
    time1_subtitle = f.text(0.70, 0.25, txt1, ha='center', size=11)

    cbar_ax = f.add_axes([0.55, 0.68,  0.3, 0.02])
    cb = f.colorbar(sct, extend='neither', orientation='horizontal', cax=cbar_ax)
    cbar_ax.set_title('Absolute error at spatial points', size=label_size)

    cbar_ax2 = f.add_axes([0.15, 0.68,  0.3, 0.02])
    cb2 = f.colorbar(cnc, extend='neither', orientation='horizontal', cax=cbar_ax2)
    cbar_ax2.set_title('Concentration value at field', size=label_size)

    fig_title = 'paper_abs_snapshot_error_of_regions_field_distr_time_' + str(t_ss1) + '_train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs.png'

    plt.savefig(fig_title, bbox_extra_artists=(cbar_ax,time_subtitle, time1_subtitle))
    plt.gcf().clear()


def paper_plot_snapshot_error_comparison_by_region_distr_and_real_field2(real_data, distr_data, rbf_data, rbf_random_data, centr_data, fixed_data, adas_distr, adas_centr, ada_fixed, time_values, t0_val, t_resample_val, mesh_dictionary, t_ss0, t_ss1):
    markersize = 1
    label_pad = -0.5
    label_size = 10

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    total_time = np.shape(real_data)[1]

    x_vals = [0] * np.shape(real_data)[0]
    y_vals = [0] * np.shape(real_data)[0]
    for i in range(0, np.shape(real_data)[0]):
        x_vals[i] = mesh_dictionary[i][0]
        y_vals[i] = mesh_dictionary[i][1]

    interval = np.hstack([np.linspace(0, 0.35), np.linspace(0.65, 1)])
    colors = plt.cm.PRGn_r(interval)
    new_cmap = LinearSegmentedColormap.from_list('name', colors)
    new_cmap.set_under('white')

    # ax1.set_title('Concentration values \nof dye at t = ' + str(t_ss0), size=11)
    ax1.set_ylabel('y', labelpad=label_pad)
    ax1.set_xlabel('x', labelpad=label_pad-2)
    ax1.text(0.5, -0.43, "(a)", size=label_size, ha="center",
             transform=ax1.transAxes)

    cnc = ax1.scatter(np.array(y_vals), 599-np.array(x_vals), c=real_data[:,t_ss0], cmap=new_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)

    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    ax1.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax1.set_xlim((0, 600))
    ax1.set_ylim((0, 600))
    ax1.set_xticks([0, 300, 600])
    ax1.set_yticks([0, 300, 600])

    #
    tick_labels = [item.get_text() for item in ax1.get_yticklabels()]
    new_labels = np.linspace(0.0, 1.0, 3, endpoint=True)
    new_labels = [str(round(l, 2)) for l in new_labels]
    ax1.set_xticklabels(new_labels)
    ax1.set_yticklabels(new_labels)

    interval = np.linspace(0.25, 1.0)
    colors = plt.cm.Blues(interval)
    e_cmap = LinearSegmentedColormap.from_list('name', colors)
    e_cmap.set_under('white')

    # ax3.set_title('Concentration values \nof dye at t = ' + str(t_ss1), size=11)
    ax3.set_xlabel('x', labelpad=label_pad-2)
    ax3.text(0.5, -0.43, "(c)", size=label_size, ha="center",
             transform=ax3.transAxes)

    ax3.scatter(np.array(y_vals), 599-np.array(x_vals), c=real_data[:,t_ss1], cmap=new_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)

    # tick_labels = [item.get_text() for item in ax3.get_yticklabels()]
    # new_labels = np.linspace(0.0, 1.0, 4, endpoint=True)
    # new_labels = [str(round(l, 2)) for l in new_labels]
    # ax1.set_xticklabels(new_labels)
    # ax1.set_yticklabels(new_labels)

    ax3.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax3.set_xlim((0, 600))
    ax3.set_ylim((0, 600))
    ax3.set_xticks([0, 300, 600])
    ax3.set_yticks([0, 300, 600])
    ax3.set_xticklabels(new_labels)
    ax3.set_yticklabels(new_labels)

    e = abs(real_data[:, t_ss0] - distr_data[:, t_ss0])
    dist_data_error = e

    ax2.scatter(np.array(y_vals), 599-np.array(x_vals), c=dist_data_error, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # ax2.set_title('Distributed optimal \n placement at t = ' + str(t_ss0), size=11)
    ax2.set_ylabel('y', labelpad=label_pad)
    ax2.set_xlabel('x', labelpad=label_pad-2)
    ax2.text(0.5, -0.43, "(b)", size=label_size, ha="center",
             transform=ax2.transAxes)

    ax2.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax2.set_xlim((0, 600))
    ax2.set_ylim((0, 600))
    ax2.set_xticks([0, 300, 600])
    ax2.set_xticklabels(new_labels)
    ax2.set_yticklabels([])

    e = abs(real_data[:, t_ss1] - distr_data[:, t_ss1])
    dist_data_error = e

    sct = ax4.scatter(np.array(y_vals), 599-np.array(x_vals), c=dist_data_error, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # ax4.set_title('Distributed optimal \n placement at t = ' + str(t_ss1), size=11)
    ax4.set_xlabel('x', labelpad=label_pad-2)
    ax4.text(0.5, -0.43, "(d)", size=label_size, ha="center",
             transform=ax4.transAxes)
    #
    ax4.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax4.set_xlim((0, 600))
    ax4.set_ylim((0, 600))
    ax4.set_xticks([0, 300, 600])
    ax4.set_xticklabels(new_labels)
    ax4.set_yticklabels([])

    # title = f.text(0.5, 0.96, 'Comparison of Point-wise Error between Field Simulation \n and Various Field Estimation Algorithm',
    #              size=14, horizontalalignment='center', verticalalignment='top')

    f.subplots_adjust(hspace=0.3,wspace=-0.25)
    txt = 't = ' + str.strip(str(round(time_values[t_ss0][0], 2))) + ' secs'
    time_subtitle = f.text(0.30, 0.25, txt, ha='center', size=11)

    txt1 = 't = ' + str.strip(str(round(time_values[t_ss1][0], 2))) + ' secs'
    time1_subtitle = f.text(0.70, 0.25, txt1, ha='center', size=11)

    cbar_ax = f.add_axes([0.55, 0.68,  0.3, 0.02])
    cb = f.colorbar(sct, extend='neither', orientation='horizontal', cax=cbar_ax)
    cbar_ax.set_title('Absolute error at spatial points', size=label_size)

    cbar_ax2 = f.add_axes([0.15, 0.68,  0.3, 0.02])
    cb2 = f.colorbar(cnc, extend='neither', orientation='horizontal', cax=cbar_ax2)
    cbar_ax2.set_title('Concentration value at field', size=label_size)

    fig_title = 'paper_abs_snapshot_error_of_regions_field_distr_time_' + str(t_ss1) + '_train_' + str(t0_val) + '_resample_' +\
                    str(t_resample_val) + 'secs.png'

    plt.savefig(fig_title, bbox_extra_artists=(cbar_ax,time_subtitle, time1_subtitle))
    plt.gcf().clear()


def interpolate_from_scattered(distributed_data, real_data, time_values, mesh_dictionary, region_dictionary, t0_val, t_resample_val):
    total_time = np.shape(real_data)[1]
    relative_error = np.zeros_like(distributed_data)

    switching_coords = []
    for t in range(0, total_time):
        e = abs(real_data[:, t] - distributed_data[:, t])
        relative_error[:, t] = e


    relative_error_time_avg = np.average(relative_error, axis=1)

    x_vals = [0] * np.shape(relative_error_time_avg)[0]
    y_vals = [0] * np.shape(relative_error_time_avg)[0]
    for i in range(0, np.shape(relative_error_time_avg)[0]):
        x_vals[i] = mesh_dictionary[i][0]
        y_vals[i] = mesh_dictionary[i][1]

    ti = np.linspace(0, 600, 600)
    XI, YI = np.meshgrid(ti, ti)

    distributed_data = np.maximum(0, distributed_data)
    distributed_data = np.minimum(255, distributed_data)
    distributed_data = 255 - distributed_data

    import cv2
    vid_name = 'estimate_video_' + str(t0_val) + '_' + str(t_resample_val)
    cap_estimate = cv2.VideoCapture(vid_name+'.avi')
    vid_length_est = int(cap_estimate.get(cv2.CAP_PROP_FRAME_COUNT))

    print('est', vid_length_est)
    # fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    # vid_name = 'estimate_video_' + str(t0_val) + '_' + str(t_resample_val)
    # out = cv2.VideoWriter(vid_name+'.avi', fourcc, 20.0, (600, 600))
    #
    # for t in range(0, total_time):
    #     # e = abs(real_data[:, t] - distributed_data[:, t])
    #     # relative_error[:, t] = e
    #
    #     # place_where_white = np.where(distributed_data == 255)
    #     # distributed_data[place_where_white] = 0
    #
    #     rbf = Rbf(np.array(y_vals), 599-np.array(x_vals), distributed_data[:, t], function='cubic')
    #     ZI = rbf(XI, YI)
    #
    #
    #     # sct = plt.scatter(np.array(y_vals), 599-np.array(x_vals), c=distributed_data[:, t], cmap='jet', antialiased='False', vmin=0.001, vmax=255)
    #
    #     # plt.savefig('test_interp0.png',)
    #     # plt.gcf().clear()
    #
    #     # sct = plt.pcolor(XI, YI, ZI, cmap='jet', antialiased='False', vmin=0.001, vmax=255)
    #
    #     estimated_intensities_matrix = np.uint8(ZI)
    #     place_where_black = np.where(estimated_intensities_matrix <= 10)
    #     estimated_intensities_matrix[place_where_black[0], place_where_black[1]] = 255
    #
    #     cimg_to_rgb = cv2.cvtColor(estimated_intensities_matrix, cv2.COLOR_GRAY2BGR)
    #     out.write(cimg_to_rgb)
    #
    #     # cv2.imshow('frame', cimg_to_rgb)
    #     # k = cv2.waitKey(0)
    # out.release()

        # plt.savefig('test_interp1.png',)
        # plt.gcf().clear()



def paper_plot_snapshot_error_comparison_train_sample_by_region3(real_data, distr_data_small_train_disc_0, distr_data_large_train_disc_0, distr_data_small_train_disc_1, distr_data_large_train_disc_1, mesh_dictionary, time_values, t0_val_0, t_resample_val_0, t0_val_1, t_resample_val_1):
    markersize = 1
    label_pad = -0.5
    label_size = 10

    # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)


    total_time = np.shape(real_data)[1]
    relative_error = np.zeros_like(real_data)

    relative_error_with_time = [0] * total_time

    x_vals = [0] * np.shape(real_data)[0]
    y_vals = [0] * np.shape(real_data)[0]
    for i in range(0, np.shape(real_data)[0]):
        x_vals[i] = mesh_dictionary[i][0]
        y_vals[i] = mesh_dictionary[i][1]

    interval = np.linspace(0.25, 1.0)
    colors = plt.cm.Oranges(interval)
    e_cmap = LinearSegmentedColormap.from_list('name', colors)
    e_cmap.set_under('white')

    # e_cmap = 'Purples'

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_small_train_disc_0[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    cnc = ax1.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    tick_labels = [item.get_text() for item in ax1.get_yticklabels()]
    new_labels = np.linspace(0.0, 1.0, 3, endpoint=True)
    new_labels = [str(round(l, 2)) for l in new_labels]
    ax1.set_xticklabels(new_labels)
    ax1.set_yticklabels(new_labels)

    ax1.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax1.set_xlim((0, 600))
    ax1.set_ylim((0, 600))
    ax1.set_xticks([0, 300, 600])
    ax1.set_yticks([0, 300, 600])
    ax1.set_ylabel('y', labelpad=label_pad+1)

    ax1.set_xlabel('x', labelpad=label_pad-2)
    ax1.text(0.5, -0.43, "(a)", size=label_size, ha="center",
             transform=ax1.transAxes)

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_small_train_disc_0[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_large_train_disc_0[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax2.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # ax2.set_title('Distributed optimal \n placement at t = ' + str(t_ss0), size=11)
    # ax2.set_xlabel('(b)', size=label_size, labelpad=label_pad)
    ax2.set_xlabel('x', labelpad=label_pad-2)
    ax2.text(0.5, -0.43, "(b)", size=label_size, ha="center",
             transform=ax2.transAxes)
    #
    # x0, x1 = ax2.get_xlim()
    # y0, y1 = ax2.get_ylim()
    # tick_labels = [item.get_text() for item in ax2.get_yticklabels()]
    # new_labels = np.linspace(0.0, 1.0, 3, endpoint=True)
    # new_labels = [str(round(l, 2)) for l in new_labels]
    # ax2.set_xticklabels(new_labels)
    # ax2.set_yticklabels(new_labels)

    ax2.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax2.set_xlim((0, 600))
    ax2.set_ylim((0, 600))
    ax2.set_xticks([0, 300, 600])
    ax2.set_xticklabels(new_labels)
    ax2.set_yticklabels([])

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_large_train_disc_0[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_small_train_disc_1[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax3.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    ax3.set_xlabel('x', labelpad=label_pad-2)
    ax3.text(0.5, -0.43, "(c)", size=label_size, ha="center",
             transform=ax3.transAxes)

    ax3.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax3.set_xlim((0, 600))
    ax3.set_ylim((0, 600))
    ax3.set_xticks([0, 300, 600])
    ax3.set_xticklabels(new_labels)
    ax3.set_yticklabels([])

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_small_train_disc_1[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_large_train_disc_1[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    sct = ax4.scatter(np.array(y_vals), 599-np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # ax4.set_title('Distributed optimal \n placement at t = ' + str(t_ss1), size=11)
    ax4.set_xlabel('x', labelpad=label_pad-2)
    ax4.text(0.5, -0.43, "(d)", size=label_size, ha="center",
             transform=ax4.transAxes)
    #
    ax4.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax4.set_xlim((0, 600))
    ax4.set_ylim((0, 600))
    ax4.set_xticks([0, 300, 600])
    ax4.set_xticklabels(new_labels)
    ax4.set_yticklabels([])

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_large_train_disc_1[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    f.subplots_adjust(hspace=0.01)
    # gs.update(top=0.965)

    # title = f.text(0.5, 0.96, 'Comparison of Point-wise Error between Field Simulation \n and Various Field Estimation Algorithm',
    #              size=14, horizontalalignment='center', verticalalignment='top')

    # f.subplots_adjust(hspace=0.01)
    # txt = 't = ' + str.strip(str(round(time_values[t_ss0][0], 2))) + ' secs'
    # time_subtitle = f.text(0.30, 0.25, txt, ha='center', size=11)
    #
    # txt1 = 't = ' + str.strip(str(round(time_values[t_ss1][0], 2))) + ' secs'
    # time1_subtitle = f.text(0.70, 0.25, txt1, ha='center', size=11)

    cbar_ax = f.add_axes([0.35, 0.68, 0.3, 0.02])
    cb = f.colorbar(sct, extend='neither', orientation='horizontal', cax=cbar_ax)
    cbar_ax.set_title('Absolute error at spatial points', size=label_size)

    # cbar_ax2 = f.add_axes([0.15, 0.68,  0.3, 0.02])
    # cb2 = f.colorbar(cnc, extend='neither', orientation='horizontal', cax=cbar_ax2)
    # cbar_ax2.set_title('Concentration value at field', size=label_size)

    fig_title = 'paper_compare_train_disc_' + '_train_' + str(t0_val_0) + '_resample_' +\
                    str(t_resample_val_0) + '_train_' + str(t0_val_1) + '_resample_' +\
                    str(t_resample_val_1) + 'secs.png'

    plt.savefig(fig_title)

    # plt.savefig(fig_title, bbox_extra_artists=(cbar_ax,))
    plt.gcf().clear()


def paper_plot_snapshot_error_comparison_train_sample_by_region4(real_data, distr_data_small_train_disc_0, distr_data_large_train_disc_0, distr_data_small_train_disc_1, distr_data_large_train_disc_1, mesh_dictionary, time_values, t0_val_0, t_resample_val_0, t0_val_1, t_resample_val_1):
    markersize = 2
    f, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, sharex='col', sharey='row')
    label_pad = -0.5
    label_size = 10


    total_time = np.shape(real_data)[1]
    relative_error = np.zeros_like(real_data)

    relative_error_with_time = [0] * total_time

    x_vals = [0] * np.shape(real_data)[0]
    y_vals = [0] * np.shape(real_data)[0]
    for i in range(0, np.shape(real_data)[0]):
        x_vals[i] = mesh_dictionary[i][0]
        y_vals[i] = mesh_dictionary[i][1]

    interval = np.linspace(0.25, 1.0)
    colors = plt.cm.Oranges(interval)
    e_cmap = LinearSegmentedColormap.from_list('name', colors)
    e_cmap.set_under('white')

    # e_cmap = 'Purples'

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_small_train_disc_0[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)


    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_small_train_disc_0[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    ax5.scatter(time_values, relative_error_with_time, c='blue', marker='.', s=markersize)
    ax5.xaxis.set_tick_params(labelbottom=True, labelsize=label_size)
    ax5.yaxis.set_tick_params(labelleft=True, labelsize=label_size)
    ax5.set_xlabel('(a)', size=label_size, labelpad=label_pad)

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_large_train_disc_0[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_large_train_disc_0[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    ax6.scatter(time_values, relative_error_with_time, c='blue', marker='.', s=markersize)
    ax6.xaxis.set_tick_params(labelbottom=True, labelsize=label_size)
    # ax6.yaxis.set_tick_params(labelleft=True, labelsize=label_size)
    ax6.set_xlabel('(b)', size=label_size, labelpad=label_pad)

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_small_train_disc_1[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_small_train_disc_1[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    ax7.scatter(time_values, relative_error_with_time, c='blue', marker='.', s=markersize)
    ax7.set_xlabel('(c)', size=label_size, labelpad=label_pad)

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_large_train_disc_1[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    ax8.scatter(time_values, relative_error_with_time, c='blue', marker='.', s=markersize)
    ax8.set_xlabel('(d)', size=label_size, labelpad=label_pad)

    y_min = 0
    y_max = 0

    error_over_time_axes = [ax5, ax6, ax7, ax8]
    for ax in error_over_time_axes:
        _, y_upper = ax.get_ylim()
        if y_upper > y_max:
            y_max = y_upper
    for ax in error_over_time_axes:
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.linspace(0, 1, 6, endpoint=True))

        ax.set_xlim(0, 60)


    # title = f.text(0.5, 0.96, 'Comparison of Point-wise Error between Field Simulation \n and Various Field Estimation Algorithm',
    #              size=14, horizontalalignment='center', verticalalignment='top')

    # f.subplots_adjust(hspace=0.01)
    # txt = 't = ' + str.strip(str(round(time_values[t_ss0][0], 2))) + ' secs'
    # time_subtitle = f.text(0.30, 0.25, txt, ha='center', size=11)
    #
    # txt1 = 't = ' + str.strip(str(round(time_values[t_ss1][0], 2))) + ' secs'
    # time1_subtitle = f.text(0.70, 0.25, txt1, ha='center', size=11)

    # cbar_ax2 = f.add_axes([0.15, 0.68,  0.3, 0.02])
    # cb2 = f.colorbar(cnc, extend='neither', orientation='horizontal', cax=cbar_ax2)
    # cbar_ax2.set_title('Concentration value at field', size=label_size)

    # f.text(0.5, 0.96, 'Normwise Relative Error between Field Simulation \n and Various Field Estimation Algorithms', size=14, horizontalalignment='center', verticalalignment='top')
    f.subplots_adjust(bottom=0.15, hspace=0.35)
    f.text(0.5, 0.04, 'Time (secs)', ha='center', size=11)
    f.text(0.04, 0.5, 'Relative error using Frobenius norm', va='center', rotation='vertical', size=11)

    fig_title = 'paper_compare_train_disc_' + '_train_' + str(t0_val_0) + '_resample_' +\
                    str(t_resample_val_0) + '_train_' + str(t0_val_1) + '_resample_' +\
                    str(t_resample_val_1) + 'secs2.png'

    plt.savefig(fig_title)

    # plt.savefig(fig_title, bbox_extra_artists=(cbar_ax,))
    plt.gcf().clear()


def plot_snapshot_error_comparison_train_sample_by_region_localization(real_data, distr_data_0, distr_data_1, distr_data_2, distr_data_3, mesh_dictionary, time_values):
    markersize = 2
    f, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, sharex='col', sharey='row')
    label_pad = -0.5
    label_size = 10


    total_time = np.shape(real_data)[1]
    relative_error = np.zeros_like(real_data)

    relative_error_with_time = [0] * total_time

    x_vals = [0] * np.shape(real_data)[0]
    y_vals = [0] * np.shape(real_data)[0]
    for i in range(0, np.shape(real_data)[0]):
        x_vals[i] = mesh_dictionary[i][0]
        y_vals[i] = mesh_dictionary[i][1]

    interval = np.linspace(0.25, 1.0)
    colors = plt.cm.Oranges(interval)
    e_cmap = LinearSegmentedColormap.from_list('name', colors)
    e_cmap.set_under('white')

    # e_cmap = 'Purples'

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_0[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)


    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_0[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    ax5.scatter(time_values, relative_error_with_time, c='blue', marker='.', s=markersize)
    ax5.xaxis.set_tick_params(labelbottom=True, labelsize=label_size)
    ax5.yaxis.set_tick_params(labelleft=True, labelsize=label_size)
    ax5.set_xlabel('(a)', size=label_size, labelpad=label_pad)

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_1[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_1[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    ax6.scatter(time_values, relative_error_with_time, c='blue', marker='.', s=markersize)
    ax6.xaxis.set_tick_params(labelbottom=True, labelsize=label_size)
    # ax6.yaxis.set_tick_params(labelleft=True, labelsize=label_size)
    ax6.set_xlabel('(b)', size=label_size, labelpad=label_pad)

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_2[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_2[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    ax7.scatter(time_values, relative_error_with_time, c='blue', marker='.', s=markersize)
    ax7.set_xlabel('(c)', size=label_size, labelpad=label_pad)

    for t in range(0, total_time):
        e = np.linalg.norm(real_data[:, t] - distr_data_3[:, t]) / np.linalg.norm(real_data[:, t])
        relative_error_with_time[t] = e

    ax8.scatter(time_values, relative_error_with_time, c='blue', marker='.', s=markersize)
    ax8.set_xlabel('(d)', size=label_size, labelpad=label_pad)

    y_min = 0
    y_max = 0

    error_over_time_axes = [ax5, ax6, ax7, ax8]
    for ax in error_over_time_axes:
        _, y_upper = ax.get_ylim()
        if y_upper > y_max:
            y_max = y_upper
    for ax in error_over_time_axes:
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.linspace(0, 1, 6, endpoint=True))

        ax.set_xlim(0, 60)


    title = f.text(0.5, 0.96, 'Normwise Relative Error between Field Simulation \n and Proposed Algorithm with Localization Error \n in Initial ROM and Sensor Measurements',
                 size=14, horizontalalignment='center', verticalalignment='top')

    # f.subplots_adjust(hspace=0.01)
    # txt = 't = ' + str.strip(str(round(time_values[t_ss0][0], 2))) + ' secs'
    # time_subtitle = f.text(0.30, 0.25, txt, ha='center', size=11)
    #
    # txt1 = 't = ' + str.strip(str(round(time_values[t_ss1][0], 2))) + ' secs'
    # time1_subtitle = f.text(0.70, 0.25, txt1, ha='center', size=11)

    # cbar_ax2 = f.add_axes([0.15, 0.68,  0.3, 0.02])
    # cb2 = f.colorbar(cnc, extend='neither', orientation='horizontal', cax=cbar_ax2)
    # cbar_ax2.set_title('Concentration value at field', size=label_size)

    # f.text(0.5, 0.96, 'Normwise Relative Error between Field Simulation \n and Various Field Estimation Algorithms', size=14, horizontalalignment='center', verticalalignment='top')
    f.subplots_adjust(bottom=0.15, hspace=0.35, top=0.8)
    f.text(0.5, 0.04, 'Time (secs)', ha='center', size=11)
    f.text(0.04, 0.5, 'Relative error using Frobenius norm', va='center', rotation='vertical', size=11)

    fig_title = 'localization_error_compare.png'

    plt.savefig(fig_title)

    # plt.savefig(fig_title, bbox_extra_artists=(cbar_ax,))
    plt.gcf().clear()


def localization_error_plot_snapshot_error_comparison_train_sample_by_region(real_data, distr_data_0, distr_data_1, distr_data_2, distr_data_3, mesh_dictionary, time_values):
    markersize = 1
    label_pad = -0.5
    label_size = 10

    # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    total_time = np.shape(real_data)[1]
    relative_error = np.zeros_like(real_data)

    relative_error_with_time = [0] * total_time

    x_vals = [0] * np.shape(real_data)[0]
    y_vals = [0] * np.shape(real_data)[0]
    for i in range(0, np.shape(real_data)[0]):
        x_vals[i] = mesh_dictionary[i][0]
        y_vals[i] = mesh_dictionary[i][1]

    interval = np.linspace(0.25, 1.0)
    colors = plt.cm.Oranges(interval)
    e_cmap = LinearSegmentedColormap.from_list('name', colors)
    e_cmap.set_under('white')

    # e_cmap = 'Purples'

    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_0[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    cnc = ax1.scatter(np.array(y_vals), 599 - np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap,
                      antialiased='False', vmin=0.001, vmax=255, s=markersize)
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    tick_labels = [item.get_text() for item in ax1.get_yticklabels()]
    new_labels = np.linspace(0.0, 1.0, 3, endpoint=True)
    new_labels = [str(round(l, 2)) for l in new_labels]
    ax1.set_xticklabels(new_labels)
    ax1.set_yticklabels(new_labels)

    ax1.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax1.set_xlim((0, 600))
    ax1.set_ylim((0, 600))
    ax1.set_xticks([0, 300, 600])
    ax1.set_yticks([0, 300, 600])
    ax1.set_ylabel('y', labelpad=label_pad + 1)

    ax1.set_xlabel('x', labelpad=label_pad - 2)
    ax1.text(0.5, -0.43, "(a)", size=label_size, ha="center",
             transform=ax1.transAxes)


    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_1[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax2.scatter(np.array(y_vals), 599 - np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False',
                vmin=0.001, vmax=255, s=markersize)
    # ax2.set_title('Distributed optimal \n placement at t = ' + str(t_ss0), size=11)
    # ax2.set_xlabel('(b)', size=label_size, labelpad=label_pad)
    ax2.set_xlabel('x', labelpad=label_pad - 2)
    ax2.text(0.5, -0.43, "(b)", size=label_size, ha="center",
             transform=ax2.transAxes)
    #
    # x0, x1 = ax2.get_xlim()
    # y0, y1 = ax2.get_ylim()
    # tick_labels = [item.get_text() for item in ax2.get_yticklabels()]
    # new_labels = np.linspace(0.0, 1.0, 3, endpoint=True)
    # new_labels = [str(round(l, 2)) for l in new_labels]
    # ax2.set_xticklabels(new_labels)
    # ax2.set_yticklabels(new_labels)

    ax2.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax2.set_xlim((0, 600))
    ax2.set_ylim((0, 600))
    ax2.set_xticks([0, 300, 600])
    ax2.set_xticklabels(new_labels)
    ax2.set_yticklabels([])


    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_2[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    ax3.scatter(np.array(y_vals), 599 - np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap, antialiased='False',
                vmin=0.001, vmax=255, s=markersize)
    ax3.set_xlabel('x', labelpad=label_pad - 2)
    ax3.text(0.5, -0.43, "(c)", size=label_size, ha="center",
             transform=ax3.transAxes)

    ax3.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax3.set_xlim((0, 600))
    ax3.set_ylim((0, 600))
    ax3.set_xticks([0, 300, 600])
    ax3.set_xticklabels(new_labels)
    ax3.set_yticklabels([])


    for t in range(0, total_time):
        e = abs(real_data[:, t] - distr_data_3[:, t])
        relative_error[:, t] = e
    relative_error_time_avg = np.average(relative_error, axis=1)

    sct = ax4.scatter(np.array(y_vals), 599 - np.array(x_vals), c=relative_error_time_avg, cmap=e_cmap,
                      antialiased='False', vmin=0.001, vmax=255, s=markersize)
    # ax4.set_title('Distributed optimal \n placement at t = ' + str(t_ss1), size=11)
    ax4.set_xlabel('x', labelpad=label_pad - 2)
    ax4.text(0.5, -0.43, "(d)", size=label_size, ha="center",
             transform=ax4.transAxes)
    #
    ax4.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax4.set_xlim((0, 600))
    ax4.set_ylim((0, 600))
    ax4.set_xticks([0, 300, 600])
    ax4.set_xticklabels(new_labels)
    ax4.set_yticklabels([])


    f.subplots_adjust(hspace=0.01)
    # gs.update(top=0.965)

    title = f.text(0.5, 0.96, 'Comparison of Point-wise Error between Field Simulation \n and Proposed Algorithm with Localization Error \n in Initial ROM and Sensor Measurements',
                 size=14, horizontalalignment='center', verticalalignment='top')

    # f.subplots_adjust(hspace=0.01)
    # txt = 't = ' + str.strip(str(round(time_values[t_ss0][0], 2))) + ' secs'
    # time_subtitle = f.text(0.30, 0.25, txt, ha='center', size=11)
    #
    # txt1 = 't = ' + str.strip(str(round(time_values[t_ss1][0], 2))) + ' secs'
    # time1_subtitle = f.text(0.70, 0.25, txt1, ha='center', size=11)

    cbar_ax = f.add_axes([0.35, 0.68, 0.3, 0.02])
    cb = f.colorbar(sct, extend='neither', orientation='horizontal', cax=cbar_ax)
    cbar_ax.set_title('Absolute error at spatial points', size=label_size)

    fig_title = 'localization_error_compare_orange.png'

    plt.savefig(fig_title)

    # plt.savefig(fig_title, bbox_extra_artists=(cbar_ax,))
    plt.gcf().clear()