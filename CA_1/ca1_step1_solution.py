#%%
""" 
CA 1: STEP 1
Group 26: Anton Gusarov, Martin Gustavsson
"""
import re
import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from itertools import islice

#%% Read the first 2 lines:
with open('ca1_step1_input_data.txt', 'r', errors='strict') as file:  
    header_lines = list() # will have a list of 2 strings
    for i in range(2): 
        header_lines.append(file.readline())

#%% Parse a header of the file (first 2 lines):
with open('ca1_step1_input_data.txt', 'r', errors='strict') as file:
    # regular expressions used where assumed that ',' or ';' are the separators;
    # itertools.islice to iterate over lines 1 and 2:
    header_lines = [re.split(r'[,;]', line) for line in islice(file.readlines(), 2)]
header_keys = tuple(key.strip() for key in header_lines[0])  # remove trailing spaces by .strip()
header_keys = tuple(re.sub(r'[\s#)]','', key) for key in header_keys)  # remove aux symbols
header_keys = tuple(re.sub(r'[(]','_', key) for key in header_keys)  # substitute aux symbols by "_"

#%% Remove all space characters from header line 2 and transform to float:
header_values = [float(value.strip()) for value in header_lines[1]]
header_values[0] = int(header_values[0])  # time_steps to be int
header_values[4] = int(header_values[4])  # N_particles to be int

#%% Create a dict {header_key: header_value}:
# CORRECTED: Pythonic way using zip()
header = {}
for key, value in zip(header_keys, header_values):
    header[key] = value
locals().update(header)  # unzip the dict to a set of variables 

#%% Identify the start of each time step by looking for the 
# time-step header # time_step. And then parse the N_particles rows 
# of position and velocity data in each time-step block

# First task: print headers for time_step:
with open('ca1_step1_input_data.txt', 'r', errors='strict') as file:   
    for line in islice(file.readlines(), 4):
        if '# time_step ' in line: print(line)

#%% Parse the data in each time-step block.
# Store the data in a (three-level) nested list called data.
# 1. The first level should be the time-step index,
# 2. the second level the particle index,
# 3. third level a list of the four values of the x, y position-
#    and v_x, v_y velocitiy- components stored as floating point numbers (float).

inner_list_particle = []
data = []
particle_count = 0

# The parser:
with open('ca1_step1_input_data.txt', 'r', errors='strict') as file:   
    for line in islice(file.readlines(), 3, None, None):  # start iteratieng from trird line - skip the header
        if (not line) or line.isspace() :  # skip empty line: no symbols or only spaces 
            #print('empty string')
            continue

        if '# time_step ' in line: 
            time_step_index = int(''.join(re.findall(r'\d',line))) # parse time-step index

        if re.search(r'# ', line) is None:  # find a line without '#' symbol i.e. with position and speeds numbers
            inner_list_particle.append([float(i.strip()) for i in re.split(r'[,;]', line)])
            particle_count += 1
            if particle_count == N_particles:  # CORRECTED: number of particles is no longer hardcoded
                # This works, but is extra work for the computer
                # Please do not copy the data and then remove the original
                # Just pass the data along by appending it
                # and set inner_list_particle to a new empty list []
                # CORRECTED according to the commenst above:
                data.append(inner_list_particle)
                inner_list_particle = []
                particle_count = 0

print('data list size:', len(data), len(data[0]), len(data[0][0]))  # check data list dimensions

#%% Convert a nested list to a multidimensional (NumPy) array
data = np.array(data)
print(type(data))
print(data.dtype)
print(data.shape)

#%% Split position vectors and velocity vectors
# To make the numerical analysis easier you want to store the final result in two separate ndarrays, 
# one for the positions called R and one for the velocities called V
R = data[:,:,:2]
V = data[:,:,2:]

print(R.shape)
print(V.shape)

#%% To swap the order of the axes use the NumPy "transpose-like" manipulation routines:
R = np.swapaxes(R,1,2)
V = np.swapaxes(V,1,2)

print(R.shape)
print(V.shape)

#%% Plot the first time step with Matplotlib
fig, axs = plt.subplots()
axs.set_aspect(aspect=1)
axs.scatter(R[0, 0, :], 
            R[0, 1, :], 
            c='red', s=50.0, alpha=0.4, edgecolors='none')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show(fig)

#%% Storing data using the hdf5 file format
# use its h5py.File class together with the 'with'-statement to open the file ca1_step1_output_data.h5 file. 
# Use '.create_dataset' method to store the ndarrays R and V, 
# and store the other parameters as attributes using the .attrs.create method.

# CORRECTED:
# Please store the attributes in the root of the hdf5 file not in dset1
# using the function pointed out in the description

with h5py.File('ca1_step1_output_data.h5', 'w') as hfile:
    dset1 = hfile.create_dataset('R', data=R)
    dset1 = hfile.create_dataset('V', data=V)

    # dset1.attrs['N_particles'] = N_particles
    # dset1.attrs['time_steps'] = time_steps
    # dset1.attrs['time_step_s'] = time_step_s
    # dset1.attrs['radius'] = radius
    # dset1.attrs['v_variance'] = v_variance

    hfile.attrs.create('N_particles', N_particles)
    hfile.attrs.create('time_steps', time_steps)
    hfile.attrs.create('time_step_s', time_step_s)
    hfile.attrs.create('radius', radius)
    hfile.attrs.create('v_variance', v_variance)


# %%
