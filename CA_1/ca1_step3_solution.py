#%%
# =========================
#        CA 1: STEP 3
# =========================
import os
import itertools
import h5py
from scipy import stats
from scipy.spatial import distance
import numpy as np
from numpy import random
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from scipy.spatial import distance
from scipy.spatial import cKDTree

from ca1_step_2_solution import animate_particles  
from ca1_step_2_solution import plot_statistical_analysis 

#%% 
# 1. Particle system initialization
#------------------------------------
N_particles = 400 
sigma_v = 0.1
r0 = 2 * random.rand(2, N_particles) - 1
v0 = sigma_v * random.randn(2, N_particles)

print(r0.shape, v0.shape) # shape check

# sanity check plot of initial positions:
x0, y0 = r0[0, :], r0[1, :]
fig, axs = plt.subplots()
axs.set_aspect(aspect=1)
axs.scatter(x0, y0, color='green', s=50.0, alpha=0.4, edgecolors='none')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.show(fig)

# %% 
# 2. Time propagation
#------------------------------------
time_step = 0.02  # 50 steps/sec

def time_propagation(r, v, time_step):
    r = r + v * time_step  # broadcasting was used to implement this in a single line
    return r, v


r, v = time_propagation(r0, v0, time_step) # call the function

# sanity check plot of time propagation:
x, y = r[0, :], r[1, :]
fig, axs = plt.subplots()
axs.set_aspect(aspect=1)
axs.scatter(x, y, color='blue', s=50.0, alpha=0.4, edgecolors='none')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.show(fig)

# %%
# 3. Simulation function
#------------------------------------
# The function uses the time_propagation function to compute 
# the time evolution of the particles for a given number of 'time_steps'
# and returns the positions R and velocities V of all particles for all time steps.

def simulate(r0, v0, time_step, time_steps, update_function, **kwargs):
    # Allocate R and V:
    R = np.empty((time_steps, r0.shape[0], r0.shape[1]))
    V = np.empty((time_steps, v0.shape[0], v0.shape[1]))

    # Initialize R[0] and V[0] using r0 and v0:
    R[0,:,:] = r0
    V[0,:,:] = v0

    # Loop over all time steps and call:
    for t in range(1, time_steps):
        R[t,:,:], V[t,:,:] = update_function(R[t-1,:,:], V[t-1,:,:], time_step, **kwargs)
    return R, V


simulation1 = simulate(
    r0, v0, time_step, time_steps=400, 
    update_function=time_propagation)

#%%
# call the animation:
animate_particles(simulation1[0], time_step, filename='ca1_step3_movie1.mp4')

# %%
# 4. Hard wall boundary collisions
#------------------------------------
# -- check if the position coordinates (x and  y) are out of bounds and
# -- the particle at the same time has a velocity component going in the wrong direction (relative to the "wall").
# -- In this case change the sign of the corresponding velocity component (v_x or v_y)

# Use numpy boolean arrays for indexing so that each line of code treats all particles at the same time. 
# Modifiy the velocity vector v in-place so that the function does not have to return anything.

def boundary_collisions(r, v):
    ''' Modifies the velocity vector according to the hard wall boundary conditions.
    '''
    boundary = np.full(r.shape, 1)  # defines the abs value of coordinates |x| or |y|
    boundary_check = np.greater(np.absolute(r), boundary) # returns True if any module of coordinate > 1
    boundary_check = -2 * boundary_check.astype(int) + 1 # True/False to -1/1
    v = np.multiply(boundary_check, v)  # changes the sign of v components for where boundary_check is True
    return r, v


def update_with_boundaries(r, v, time_step):
    ''' Time propagation with boundary conditions.
    '''
    r, v = time_propagation(r, v, time_step)
    r, v = boundary_collisions(r, v)
    return r, v


simulation2 = simulate(
    r0, v0, time_step, time_steps=400, 
    update_function=update_with_boundaries)

animate_particles(simulation2[0], time_step, 'ca1_step3_movie2.mp4')

# %%
# 5. Particle-prticle collision
#------------------------------------
# -- loops over all pairs of particles,
# -- check if they are closer to each other than 2*radius and are travelling towards each other.
# -- If yes then modifiy their velocity vectors according to a momentum conserving collision.

radius = 0.05
def particle_collisions(r, v, radius):
    dist = distance.pdist(np.swapaxes(r, 0,1), metric='euclidean')  # returns condensed vector of distances
    dist = distance.squareform(dist) # find pairs closer to each other than 2*radius

    check_dist = np.less(dist, 2*radius)
    np.fill_diagonal(check_dist, False) # fill diagonal
    check_dist = np.triu(check_dist) # remove duplicates due to symmetry
    indx = np.asarray(np.where(check_dist==True))  # return indices of pairs where distance is smaller than 2*r
    
    indx = np.split(indx, indx.shape[1], axis=1) 
    for i in indx: # i - pair of indexes of the colliding particles
        # v_i = v[:, i[0]] # i-particle pair of v components (x, y)
        # v_j = v[:, i[1]] # j-particle pair of v components (x, y)

        v_i_x = v[:, i[0]][0] - (v[:, i[0]][0] - v[:, i[1]][0])
        v_i_y = v[:, i[0]][1] - (v[:, i[0]][1] - v[:, i[1]][1])

        v_j_x = v[:, i[1]][0] + (v[:, i[0]][0] - v[:, i[1]][0])
        v_j_y = v[:, i[1]][1] + (v[:, i[0]][1] - v[:, i[1]][1])


        v[:, i[0]] = [v_i_x, v_i_y]
        v[:, i[1]] = [v_j_x, v_j_y]
    return r, v


def update_with_interactions_slow(r, v, time_step, radius):
    r, v = time_propagation(r, v, time_step)
    r, v = boundary_collisions(r, v)
    r, v = particle_collisions(r, v, radius)
    return r, v


simulation3 = simulate(
    r0, v0, time_step, time_steps=100, 
    update_function=update_with_interactions_slow, radius=radius)

animate_particles(simulation3[0], time_step, 'ca1_step3_movie3.mp4')


# %%
# 6. Fast hard spheres
#------------------------------------
# Use the KDTree class in scipy.spatial to rapidly find all pairs of particles 
# closer than two times the radius to each other.

def particle_collisions_fast(r, v, radius):

    tree = cKDTree(np.swapaxes(r, 0,1))
    dist_set = tree.query_pairs(2*radius) # method; returns Python SET of pairs of points whose distance is at most r

    for i in list(dist_set): # i - pair of indexes of the colliding particles
        # v_i = v[:, i[0]] # i-particle pair of v components (x, y)
        # v_j = v[:, i[1]] # j-particle pair of v components (x, y)

        v_i_x = v[:, i[0]][0] - (v[:, i[0]][0] - v[:, i[1]][0])
        v_i_y = v[:, i[0]][1] - (v[:, i[0]][1] - v[:, i[1]][1])

        v_j_x = v[:, i[1]][0] + (v[:, i[0]][0] - v[:, i[1]][0])
        v_j_y = v[:, i[1]][1] + (v[:, i[0]][1] - v[:, i[1]][1])

        v[:, i[0]] = [v_i_x, v_i_y]
        v[:, i[1]] = [v_j_x, v_j_y]
    return r, v


def update_with_interactions_fast(r, v, time_step, radius):
    r, v = time_propagation(r, v, time_step)
    r, v = boundary_collisions(r, v)
    r, v = particle_collisions_fast(r, v, radius)
    return r, v


simulation4 = simulate(
    r0, v0, time_step, time_steps=1000, 
    update_function=update_with_interactions_fast, radius=radius)

animate_particles(simulation4[0], time_step, 'ca1_step3_movie4.mp4')

# %%
# 7. Statistical analysis of interacting particles
#-------------------------------------------------
plot_statistical_analysis(simulation4[0], simulation4[1], time_step, 'ca1_step3_figure_summary.svg')

