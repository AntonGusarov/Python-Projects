""" 
CA 1: STEP 2
Group 26: Anton Gusarov, Martin Gustavsson
"""
import os
import itertools
import h5py
from scipy import stats
from scipy.spatial import distance
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

def plot_statistical_analysis(R, V, time_step, filename):
    """
    Analyze R and V and plots statistical analysis results 
    """
    time = np.arange(0, R.shape[0]*time_step, time_step) # [sec]

    # CORRECTED according to the following feedback:
    # General comment on reshaping of arrays:

    # Please do not hard code the dimensions in the reshaping of arrays
    # This makes your function only work for 1000 timesteps with 400 particles

    # Instead of doing reshaping with given shapes please use the ndarray method .flatten()

    # It is also convenient to refrain from defining flattened variables and instead
    # flatten the array when calling .fit and .hist, i.e.
    # plt.hist(R[:, 0, :].flatten(), ...)


    # Prepare data for x,y velocities and coordinates plots:
    #---------------------------------------------------------------
    # fit model to the data i.e. estimate parameters of Gaussian distribution:
    v_x_loc, v_x_scale = stats.norm.fit(V[:, 0, :].flatten())
    # create modelled distribution from the estimayed above parameters:
    v_x_pdf = stats.norm.pdf(np.arange(-0.5, 0.5, 0.01), loc=v_x_loc, scale=v_x_scale) 

    v_y_loc, v_y_scale = stats.norm.fit(V[:, 1, :].flatten())
    v_y_pdf = stats.norm.pdf(np.arange(-0.5, 0.5, 0.01), loc=v_y_loc, scale=v_y_scale)

    x_loc, x_scale = stats.uniform.fit(R[:, 0, :].flatten())
    x_pdf = stats.uniform.pdf(np.arange(-1.5, 1.5, 0.01), loc=x_loc, scale=x_scale)

    y_loc, y_scale = stats.uniform.fit(R[:, 1, :].flatten())
    y_pdf = stats.uniform.pdf(np.arange(-1.5, 1.5, 0.01), loc=y_loc, scale=y_scale)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=0.5, hspace=0.7)  # tune the subplot layout

    # Prepare data for pair distance distribution function:
    # (Implemented using Scipy pdist)
    #---------------------------------------------------------------
    # R_split_time = np.split(R, R.shape[0], axis=0) # list of 1000 1x2x400 nparrays
    # def calculate_distances(R_split_time):
    #     dist = distance.pdist(np.swapaxes(np.squeeze(R_split_time), 0, 1), 
    #                           metric='euclidean')  # returns condensed vector of distances
    #     return dist
    # Dist = list(map(calculate_distances, R_split_time))  # produces the list of condensed distances
    # Dist = np.asarray(Dist)  # (1000, 79800) ndarray

    # CORRECTED:
    # (Alternative implementation of pairwise dist using the broadcasting)
    #----------------------------------------------------------------------
    # R_x = R[:, 0, :] # x positions (1000, 400)
    # R_y = R[:, 1, :] # y positions (1000, 400)
    # x_dist = R_x[:, :, None] - R_x[:, None, :]
    # y_dist = R_y[:, :, None] - R_y[:, None, :]
    # dist_matrix = np.sqrt(np.square(x_dist) + np.square(y_dist))

    # CORRECTED:
    #(Alternative implementation of pairwise dist using the memory-efficient broadcasting)
    #-------------------------------------------------------------------------------------
    xy_sqrd_sum = np.sum(R**2, axis=1)
    xy_sum_sqrd_sum = xy_sqrd_sum[:,:,None] + xy_sqrd_sum[:,None,:]
    xy_2_prod = -2*np.matmul(R.swapaxes(1,2), R) 
    dist_matrix = np.sqrt(xy_sum_sqrd_sum + xy_2_prod)

    # remove diagonal elements and repeated symmetrical values from a distance matrix:
    def extract_upper_triang(time_slice):
        '''Take only values of the upper triangular part of a distance matrix and flatten'''
        out = np.tril(time_slice, k=1).squeeze()
        out = out[out!=0.0] # remove distances betwee
        return out

    dist_matrix_time_slices = np.split(dist_matrix, dist_matrix.shape[0], axis=0)
    Dist = list(map(extract_upper_triang, dist_matrix_time_slices))
    Dist = np.asarray(Dist)

    # Prepare data for velocity norm pdf:
    #---------------------------------------------------------------
    V_norm = np.linalg.norm(V, ord=2, axis=1)
    v_loc, v_scale = stats.rayleigh.fit(V_norm.flatten())
    v_pdf = stats.rayleigh.pdf(np.arange(0, 0.5, 0.01), loc=v_loc, scale=v_scale)

    # Prepare data for E_k and V_tot:
    #---------------------------------------------------------------
    V_sum = np.sum(V, 2)
    V_tot = np.linalg.norm(V_sum, ord=2, axis=1)
    E_k = np.sum(np.square(V_norm)/2, 1)

    # PLOTS:
    #===============================================================
    plt.figure(1, figsize=(20, 30))  # ’figsize ’ to make a figure larger
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=0.5, hspace=1.5)  # tune the subplot layout
    # V_tot plot:                 
    #---------------------------------------------------------------
    plt.subplot(4,2,1)
    plt.plot(time, V_tot, color='blue', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('Total velocity')
    plt.grid(True)

    # E_k plot:                 
    #---------------------------------------------------------------
    plt.subplot(4,2,2)
    plt.plot(time, E_k, color='blue', linewidth=1.5)
    plt.ylim(0, 5) # can be removed to look at the noisy component
    plt.xlabel('Time')
    plt.ylabel('Kinetic energy')
    plt.grid(True)

    # pair distance distribution function plot:                 
    #---------------------------------------------------------------
    plt.subplot(4,2,3)
    plt.hist(Dist[500], 70, density=True, facecolor='blue', alpha=0.75)  # plot hist only at a single time-point
    plt.xlabel('Pair distance, $d_{ij}$')
    plt.ylabel('Probability')
    plt.xlim(0, 3.0)  # can be removed to look at the noisy component
    plt.grid(True)

    # Plot velocity norm pdf and hist:
    #---------------------------------------------------------------
    plt.subplot(4,2,4)
    plt.hist(V_norm.flatten(), 70, density=True, facecolor='blue', alpha=0.75)
    plt.plot(np.arange(0, 0.5, 0.01), v_pdf, linewidth=2, color='red')
    plt.xlabel('Velocity norm')
    plt.ylabel('Probability')
    plt.xlim(0, 0.5) # can be removed to look at the noisy component
    plt.grid(True)

    # Plots for x,y velocities and coordinates:                 
    #---------------------------------------------------------------
    plt.subplot(4,2,5)  # Position number option
    plt.hist(V[:, 0, :].flatten(), 40, density=True, facecolor='blue', alpha=0.75)
    plt.plot(np.arange(-0.5, 0.5, 0.01), v_x_pdf, linewidth=2, color='red')
    plt.xlabel('x velocities')
    plt.ylabel('Probability')
    plt.xlim(-0.4, 0.4) # can be removed to look at the noisy component
    plt.grid(True)

    plt.subplot(4,2,6)  # Position number option
    plt.hist(V[:, 1, :].flatten(), 40, density=True, facecolor='blue', alpha=0.75)
    plt.plot(np.arange(-0.5, 0.5, 0.01), v_y_pdf, linewidth=2, color='red')
    plt.xlabel('y velocities')
    plt.ylabel('Probability')
    plt.xlim(-0.4, 0.4) # can be removed to look at the noisy component
    plt.grid(True)

    plt.subplot(4,2,7)
    plt.hist(R[:, 0, :].flatten(), 70, density=True, facecolor='blue', alpha=0.75)
    plt.plot(np.arange(-1.5, 1.5, 0.01), x_pdf, linewidth=2, color='red')
    plt.xlabel('x coordinates')
    plt.ylabel('Probability')
    plt.xlim(-1.5, 1.5) # can be removed to look at the noisy component
    plt.grid(True)

    plt.subplot(4,2,8)
    plt.hist(R[:, 1, :].flatten(), 70, density=True, facecolor='blue', alpha=0.75)
    plt.plot(np.arange(-1.5, 1.5, 0.01), y_pdf, linewidth=2, color='red')
    plt.xlabel('y coordinates')
    plt.ylabel('Probability')
    plt.xlim(-1.5, 1.5) # can be removed to look at the noisy component
    plt.grid(True)

    plt.savefig(filename)
    plt.show()


def animate_particles(R, time_step, filename='movie.mp4'):
    """Generates an mp4 movie with the particle trajectories
    using MatPlotLib. """

    if os.path.isfile(filename):
        print('WARNING (animate_particles): The output file', filename, 'exists. Skipping animation.')
        return

    frames = R.shape[0]
    frames_per_second = 1. / time_step

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('image')
    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.0])

    markers = ax.scatter([], [], facecolor='red', s=180, alpha=0.5)
    text = plt.text(-0.9, 0.9, 'frame =', ha='left')

    def update(frame, markers, text):
        print('frame =', frame)
        r = R[frame]
        markers.set_offsets(r.T)
        text.set_text('frame = {}'.format(frame))

    anim = animation.FuncAnimation(
        fig, update,
        frames=frames, interval=50,
        fargs=(markers, text),
        blit=False, repeat=False,
    )

    writer = animation.writers['ffmpeg'](fps=frames_per_second)
    anim.save(filename, writer=writer, dpi=100)
    plt.close()


if __name__ == '__main__':
    # Read h5:
    with h5py.File('ca1_step1_output_data.h5', 'r') as hfile:
        print(hfile.keys())

        dataset_R = hfile['R']
        R = dataset_R[:]
        dataset_V = hfile['V']
        V = dataset_V[:]

        time_step_s = hfile.attrs['time_step_s']
        radius = hfile.attrs['radius']
        v_variance = hfile.attrs['v_variance']

    # Plot the y-position of particle number 123 as a function of time t:
    y_position_n123 = R[:, 1, 123]
    time = np.arange(0, R.shape[0]*time_step_s, time_step_s) # [sec]

    plt.plot(time, y_position_n123, 
            color='red', linewidth=1.5)
    plt.title('Particle 123 y-position') 
    plt.xlabel('Time')
    plt.ylabel('y-position')
    plt.grid(True)
    plt.show()

    # Call the statistical plots function:
    plot_statistical_analysis(R, V, time_step_s, filename='ca1_step2_figure_summary.svg')

    # Call the animation function:
    animate_particles(R, time_step_s, filename='ca1_step1_movie.mp4')
