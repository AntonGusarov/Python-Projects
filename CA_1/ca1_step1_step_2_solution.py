#%%
# =========================
#        CA 1: STEP 1
# =========================

import re
import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from itertools import islice

#%% Simply reading the first 2 lines:
with open('ca1_step1_input_data.txt', 'r', errors='strict') as file: # opener is a custom callable   
    lines = list()
    for i in range(2): 
        lines = file.readline()

# %% Parse the main header of the file (first 2 lines):
with open('ca1_step1_input_data.txt', 'r', errors='strict') as file:
    # read the header only i.e. the first 2 lines
    # reg. expr. used where assumed that ',' or ';' are the separators:
    lines = [re.split(r'[,;]', line) for line in islice(file.readlines(), 2)] 

header_keys = tuple(i.strip() for i in lines[0]) # remove trailing spaces by .strip()
header_keys = tuple(re.sub(r'[\s#)]','', i) for i in header_keys)
header_keys = tuple(re.sub(r'[(]','_', i) for i in header_keys)

# %% remove all space characters from header line 2 and tx to float:
header_values = [float(i.strip()) for i in lines[1]]
header_values[0] = int(header_values[0])  # time_steps to be int
header_values[4] = int(header_values[4])  # N_particles to be int

# %% Create a dict {header_key -> header_value}:
header = {}
for i in range(len(header_keys)):
    header[header_keys[i]] = header_values[i] # build dict incrementally

locals().update(header)  # unzip the dict to a set of variables 

# %% You need to identify the start of each time step by looking for the 
# time-step header # time_step. And then parse the N_particles rows 
# of position and velocity data in each time-step block

# First task: print headers for time_step:
with open('ca1_step1_input_data.txt', 'r', errors='strict') as file:   
    for line in islice(file.readlines(), 4):
        if '# time_step ' in line: print(line)

# %%
# Parse the data in each time-step block.
# Store the data in a (three level) nested list called data.
# 1. The first level should be the time step index,
# 2. the second level the particle index,
# 3. third level a list of the four values of the x, y position- 
#    and v_x, v_y velocitiy-components stored as floating point numbers (float).

inner_list_particle = []
data = []
particle_count = 0

# The parser:
with open('ca1_step1_input_data.txt', 'r', errors='strict') as file:   
    for line in islice(file.readlines(), 3, None, None):  # start from trird line - skip the header
        if (not line) or line.isspace() : # skip empty line: no symbols or only spaces 
            #print('empty string')
            continue

        if '# time_step ' in line: 
            time_step_index = int(''.join(re.findall(r'\d',line))) # parsed time-step index
            #print('time step:', time_step_index, '\n')

        if re.search(r'# ', line) is None: # find a line without '#' symbol i.e. with position and speeds numbers
            inner_list_particle.append([float(i.strip()) for i in re.split(r'[,;]', line)])
            particle_count += 1
            if particle_count == 400:
                data.append(inner_list_particle.copy())
                inner_list_particle.clear()
                particle_count = 0

print('data list size:', len(data), len(data[0]), len(data[0][0])) # check data list dimensions

# %% Convert a nested list to a multidimensional (NumPy) array
data = np.array(data)
print(type(data))
print(data.dtype)
print(data.shape)

# %% Split position vectors and velocity vectors
# To make the numerical analysis easier you want to store the final result in two separate ndarrays, 
# one for the positions called R and one for the velocities called V
R = data[:,:,:2]
V = data[:,:,2:]

print(R.shape)
print(V.shape)

# %% To swap the order of the axes use the NumPy "transpose-like" manipulation routines:
R = np.swapaxes(R,1,2)
V = np.swapaxes(V,1,2)

print(R.shape)
print(V.shape)

# %% Plot the first time step with Matplotlib
fig, axs = plt.subplots()
axs.set_aspect(aspect=1)
axs.scatter(R[0, 0, :], R[0, 1, :], 
            c='red', s=50.0, alpha=0.4, edgecolors='none')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show(fig)

# %% Storing data using the hdf5 file format
# use its h5py.File class together with the 'with'-statement to open the file ca1_step1_output_data.h5 file. 
# Use '.create_dataset' method to store the ndarrays R and V, 
# and store the other parameters as attributes using the .attrs.create method.

with h5py.File('ca1_step1_output_data.h5', 'w') as hfile:
    dset1 = hfile.create_dataset('R', data=R)
    dset1 = hfile.create_dataset('V', data=V)

    dset1.attrs['N_particles'] = N_particles
    dset1.attrs['time_steps'] = time_steps
    dset1.attrs['time_step_s'] = time_step_s
    dset1.attrs['radius'] = radius
    dset1.attrs['v_variance'] = v_variance

#%%
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

#%% Read h5:
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

#%%
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

# %%
# Calculate and plot total velocity V_tot and total kinetic energy E_k:
V_sum = np.sum(V, 2)
V_tot = np.linalg.norm(V_sum, ord=2, axis=1)

plt.plot(time, V_tot, 
        color='blue', linewidth=1.5)
plt.xlabel('Time')
plt.ylabel('Total velocity')
plt.grid(True)
plt.show()

V_norm = np.linalg.norm(V, ord=2, axis=1)
V_norm.shape 
E_k = np.sum(np.square(V_norm)/2, 1)

plt.plot(time, E_k, color='blue', linewidth=1.5)
plt.ylim(0, 5) # can be removed to look at the noisy component
plt.xlabel('Time')
plt.ylabel('Total kinetic energy')
plt.grid(True)
plt.show()

# %% Plotting the distribution diagrams of coordinates and velocities:

# 1. x and y positions:
R_reshaped_x = np.reshape(R[:,0,:], (400*1000,))
R_reshaped_y = np.reshape(R[:,1,:], (400*1000,))

plt.hist(R_reshaped_x, 70, density=True, facecolor='green', alpha=0.75)
plt.xlabel('x coordinates')
plt.ylabel('Probability')
plt.xlim(-1.5, 1.5)
plt.grid(True)
plt.show()

plt.hist(R_reshaped_y, 70, density=True, facecolor='green', alpha=0.75)
plt.xlabel('y coordinates')
plt.ylabel('Probability')
plt.xlim(-1.5, 1.5) # can be removed to look at the noisy component
plt.grid(True)
plt.show()

#%% 2. x and y velocities:
V_reshaped_x = np.reshape(V[:,0,:], (400*1000,))
V_reshaped_y = np.reshape(V[:,1,:], (400*1000,))

plt.hist(V_reshaped_x, 40, density=True, facecolor='blue', alpha=0.75)
plt.xlabel('x velocities')
plt.ylabel('Probability')
plt.xlim(-0.4, 0.4) 
plt.grid(True)
plt.show()

plt.hist(V_reshaped_y, 40, density=True, facecolor='blue', alpha=0.75)
plt.xlabel('y velocities')
plt.ylabel('Probability')
plt.xlim(-0.4, 0.4) 
plt.grid(True)
plt.show()

#%% Make the same as above but in multiple subplots:

plt.figure(1, figsize=(9, 9))  # ’figsize ’ to make the figure larger

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                    wspace=0.5, hspace=0.5) # Tune the subplot layout

plt.subplot(2,2,1)  # Position number option
plt.hist(V_reshaped_x, 40, density=True, facecolor='blue', alpha=0.75)
plt.xlabel('x velocities')
plt.ylabel('Probability')
plt.xlim(-0.4, 0.4) # can be removed to look at the noisy component
plt.grid(True)

plt.subplot(2,2,2)  # Position number option
plt.hist(V_reshaped_y, 40, density=True, facecolor='blue', alpha=0.75)
plt.xlabel('y velocities')
plt.ylabel('Probability')
plt.xlim(-0.4, 0.4) # can be removed to look at the noisy component
plt.grid(True)

plt.subplot(2,2,3)  # Position number option
plt.hist(R_reshaped_x, 70, density=True, facecolor='green', alpha=0.75)
plt.xlabel('x coordinates')
plt.ylabel('Probability')
plt.xlim(-1.5, 1.5) # can be removed to look at the noisy component
plt.grid(True)

plt.subplot(2,2,4)  # Position number option
plt.hist(R_reshaped_y, 70, density=True, facecolor='green', alpha=0.75)
plt.xlabel('y coordinates')
plt.ylabel('Probability')
plt.xlim(-1.5, 1.5) # can be removed to look at the noisy component
plt.grid(True)

plt.show()

# %% Fitting probability distributions
# We now want to fit the model parameters to the experimental data 
# and ovelay the fitted model probability distribution function 
# for both the positions and the velocities on the experimental histogram plots.
# All model distribution implements a .fit(data) method that gives 
# a maximum likelyhood fit of the model parameters.

# print(stats.norm.__doc__)  # thats how help works (!)
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

plt.subplot(2,2,1)  # Position number option
plt.hist(V_reshaped_x, 40, density=True, facecolor='blue', alpha=0.75)
plt.plot(np.arange(-0.5, 0.5, 0.01), v_x_pdf, linewidth=2, color='red')
plt.xlabel('x velocities')
plt.ylabel('Probability')
plt.xlim(-0.4, 0.4) # can be removed to look at the noisy component
plt.grid(True)

plt.subplot(2,2,2)  # Position number option
plt.hist(V_reshaped_x, 40, density=True, facecolor='blue', alpha=0.75)
plt.plot(np.arange(-0.5, 0.5, 0.01), v_x_pdf, linewidth=2, color='red')
plt.xlabel('y velocities')
plt.ylabel('Probability')
plt.xlim(-0.4, 0.4) # can be removed to look at the noisy component
plt.grid(True)

plt.subplot(2,2,3)
plt.hist(R_reshaped_x, 70, density=True, facecolor='green', alpha=0.75)
plt.plot(np.arange(-1.5, 1.5, 0.01), x_pdf, linewidth=2, color='red')
plt.xlabel('x coordinates')
plt.ylabel('Probability')
plt.xlim(-1.5, 1.5) # can be removed to look at the noisy component
plt.grid(True)

plt.subplot(2,2,4)
plt.hist(R_reshaped_y, 70, density=True, facecolor='green', alpha=0.75)
plt.plot(np.arange(-1.5, 1.5, 0.01), y_pdf, linewidth=2, color='red')
plt.xlabel('y coordinates')
plt.ylabel('Probability')
plt.xlim(-1.5, 1.5) # can be removed to look at the noisy component
plt.grid(True)

plt.show()

# %% Speed distribution
# Compute the speed of each particle as a function of time. 
# The speed of particle i is defined as the norm of the velocity vector:

V_norm = np.linalg.norm(V, ord=2, axis=1)
V_reshaped = np.reshape(V_norm, (400*1000,))

v_loc, v_scale = stats.rayleigh.fit(V_reshaped)
v_pdf = stats.rayleigh.pdf(np.arange(0, 0.5, 0.01), loc=v_loc, scale=v_scale)

plt.hist(V_reshaped, 70, density=True, facecolor='green', alpha=0.75)
plt.plot(np.arange(0, 0.5, 0.01), v_pdf, linewidth=2, color='red')
plt.xlabel('Velocity norm')
plt.ylabel('Probability')
plt.xlim(0, 0.5) # can be removed to look at the noisy component
plt.grid(True)

# %% Pair distance distribution function

# split into the list of to 2D arrays to make map() possible:
R_split_time = np.split(R, R.shape[0], axis=0) 

def calculate_distances(R_split_time):
    dist = distance.pdist(np.swapaxes(np.squeeze(R_split_time), 0, 1), 
                           metric='euclidean')  # returns condensed vector of distances
    return dist

Dist = list(map(calculate_distances, R_split_time))  # produces the list of condensed distances
Dist = np.asarray(Dist)  # (1000, 79800) ndarray

plt.hist(Dist[500], 70, density=True, alpha=0.75)  # plot hist only at a single time-point
plt.xlabel('Pair distance, $d_{ij}$')
plt.ylabel('Probability')
plt.xlim(0, 3.0)  # can be removed to look at the noisy component
plt.grid(True)

# %% Combined statistical analysis function
def plot_statistical_analysis(R, V, time_step, filename):
    """
    Analyzes R and V and plots subsequent results 
    """

    time = np.arange(0, R.shape[0]*time_step, time_step) # [sec]

    # Prepare data for x,y velocities and coordinates plots:
    #---------------------------------------------------------------
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

# Call the function:
plot_statistical_analysis(R, V, time_step_s, filename='ca1_step2_figure_summary.svg')

#%% Animation
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

    markers = ax.scatter([], [], s=200, alpha=0.5)
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

# call the function above:
animate_particles(R, time_step_s, filename='ca1_step1_movie.mp4')


# %%
