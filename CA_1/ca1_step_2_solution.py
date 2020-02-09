# %%
# =========================
#       CA 1: STEP 2
# =========================
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
    Analyzes R and V and plots statistical analysis results 
    """
    time = np.arange(0, R.shape[0]*time_step, time_step) # [sec]

    # Prepare data for x,y velocities and coordinates plots:
    #---------------------------------------------------------------
    # 1. x and y positions:
    R_reshaped_x = np.reshape(R[:,0,:], (400*1000,))
    R_reshaped_y = np.reshape(R[:,1,:], (400*1000,))
    V_reshaped_x = np.reshape(V[:,0,:], (400*1000,))
    V_reshaped_y = np.reshape(V[:,1,:], (400*1000,))

    v_x_loc, v_x_scale = stats.norm.fit(V_reshaped_x)  # fit model to the data
    v_x_pdf = stats.norm.pdf(np.arange(-0.5, 0.5, 0.01), loc=v_x_loc, scale=v_x_scale)  # modelled distribution

    v_y_loc, v_y_scale = stats.norm.fit(V_reshaped_y)
    v_y_pdf = stats.norm.pdf(np.arange(-0.5, 0.5, 0.01), loc=v_y_loc, scale=v_y_scale)

    x_loc, x_scale = stats.uniform.fit(R_reshaped_x)
    x_pdf = stats.uniform.pdf(np.arange(-1.5, 1.5, 0.01), loc=x_loc, scale=x_scale)

    y_loc, y_scale = stats.uniform.fit(R_reshaped_y)
    y_pdf = stats.uniform.pdf(np.arange(-1.5, 1.5, 0.01), loc=y_loc, scale=y_scale)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=0.5, hspace=0.7) # Tune the subplot layout

    # Prepare data for pair distance distribution function:
    #---------------------------------------------------------------
    R_split_time = np.split(R, R.shape[0], axis=0) 
    def calculate_distances(R_split_time):
        dist = distance.pdist(np.swapaxes(np.squeeze(R_split_time), 0, 1), 
                           metric='euclidean')  # returns condensed vector of distances
        return dist
    Dist = list(map(calculate_distances, R_split_time))  # produces the list of condensed distances
    Dist = np.asarray(Dist)  # (1000, 79800) ndarray

    # Prepare data for velocity norm pdf:
    #---------------------------------------------------------------
    V_norm = np.linalg.norm(V, ord=2, axis=1)
    V_reshaped = np.reshape(V_norm, (400*1000,))
    v_loc, v_scale = stats.rayleigh.fit(V_reshaped)
    v_pdf = stats.rayleigh.pdf(np.arange(0, 0.5, 0.01), loc=v_loc, scale=v_scale)

    # Prepare data for E_k and V_tot:
    #---------------------------------------------------------------
    V_sum = np.sum(V, 2)
    V_tot = np.linalg.norm(V_sum, ord=2, axis=1)

    V_norm = np.linalg.norm(V, ord=2, axis=1)
    V_norm.shape 
    E_k = np.sum(np.square(V_norm)/2, 1)

    # PLOTS:
    #=======
    plt.figure(1, figsize=(20, 30))  # ’figsize ’ to make the figure larger
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=0.5, hspace=1.5) # Tune the subplot layout
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
    plt.hist(V_reshaped, 70, density=True, facecolor='blue', alpha=0.75)
    plt.plot(np.arange(0, 0.5, 0.01), v_pdf, linewidth=2, color='red')
    plt.xlabel('Velocity norm')
    plt.ylabel('Probability')
    plt.xlim(0, 0.5) # can be removed to look at the noisy component
    plt.grid(True)

    # Plots for x,y velocities and coordinates:                 
    #---------------------------------------------------------------
    plt.subplot(4,2,5)  # Position number option
    plt.hist(V_reshaped_x, 40, density=True, facecolor='blue', alpha=0.75)
    plt.plot(np.arange(-0.5, 0.5, 0.01), v_x_pdf, linewidth=2, color='red')
    plt.xlabel('x velocities')
    plt.ylabel('Probability')
    plt.xlim(-0.4, 0.4) # can be removed to look at the noisy component
    plt.grid(True)

    plt.subplot(4,2,6)  # Position number option
    plt.hist(V_reshaped_x, 40, density=True, facecolor='blue', alpha=0.75)
    plt.plot(np.arange(-0.5, 0.5, 0.01), v_x_pdf, linewidth=2, color='red')
    plt.xlabel('y velocities')
    plt.ylabel('Probability')
    plt.xlim(-0.4, 0.4) # can be removed to look at the noisy component
    plt.grid(True)

    plt.subplot(4,2,7)
    plt.hist(R_reshaped_x, 70, density=True, facecolor='blue', alpha=0.75)
    plt.plot(np.arange(-1.5, 1.5, 0.01), x_pdf, linewidth=2, color='red')
    plt.xlabel('x coordinates')
    plt.ylabel('Probability')
    plt.xlim(-1.5, 1.5) # can be removed to look at the noisy component
    plt.grid(True)

    plt.subplot(4,2,8)
    plt.hist(R_reshaped_y, 70, density=True, facecolor='blue', alpha=0.75)
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

        time_step_s = dataset_V.attrs['time_step_s']
        radius = dataset_V.attrs['radius']
        v_variance = dataset_V.attrs['v_variance']
    del dataset_R, dataset_V, hfile

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
