# -*- coding: utf-8 -*-
"""
"""

# Import python modules
import os 
import numpy as np
import joblib
from joblib import Parallel, delayed
import itertools
import time
import pickle
import scipy
import sklearn.preprocessing as sk

## Set working directory
wd = 'YOUR_PATH'
os.chdir(wd)
# Import function required to run the model :
import model_function as model    

# =============================================================================
# I. Define the fix parameters for all the simulations
# =============================================================================

# np.random.seed(25)

## Species parameters
S = 100          # Number of species
mu = 0.2          # Mortality rate
Fecundity = 0.1   # Number of seed produced per individual
omega = 0.1       # Environmental niche breadth of the species
sigma_s = np.ones(S) * 0.1  # Symmetric competition breadth
sigma_h = np.ones(S) * 0.2  # Hierarchical competition breadth
Ext_seed_rain = 1 #  External migration rate

## Landscape parameters
n = 50                # Size of the landscape grid
auto_corr = 5        # Autocorrelation of the environment across the landscape
K = 10               # Carrying capacity of the cells
structure = 'mosaic' # Structure of the environmental grid, can be: homogeneous, autocorelated ('mosaic'), random or gradient
env_range = [0, 1]   # Range of the environmental values

# Simulation processes parameters
We = 1            # Strength of the environmental filtering
Wc = 10            # Strength of the competition

#%% ===========================================================================
# II. Set of tested parameters 
# =============================================================================

## Note that any parameter can be set as a fix or tested with this script, but need to modify the entry parameter in par_simul function
# The parameters and range tested for the results
correl = [0, 0.5, 0.8, 1]    # Traits correlations
phi = [0, 0.5, 1]            # Relative importance of symmetric competition vs hierarchical competition
pool_sp = list(np.linspace(1,16,16).astype('int8')) # Number of iteration for each parameter combination

Set_of_param = np.array(list(itertools.product(correl, phi, pool_sp)))

name = 'Simulation_experiment'

#%% ===========================================================================
# III. Definition of the paralell simulation function 
# =============================================================================

def par_simul(i, S, Fecundity, omega, sigma_s, sigma_h, Ext_seed_rain,
              mu, Set_of_param, K, n, wd):                   
    
    import os
    os.chdir(wd)
    import numpy as np
    import copy
    import model_function as model

    ### I.1. Extract the current simulation parameters
    correl = Set_of_param[i, 0]
    phi = Set_of_param[i, 1]

    ### I.2. Generate the species pool

    ## Draw the three functional traits from uniform distribution, with a correl as correlation coefficient 
    correlated_traits = model.gen_corr_traits(correl, S).astype('float32')
    Symmetric_trait = correlated_traits[:, 0]
    Environmental_trait = correlated_traits[:, 1] 
    Hierarchic_trait = correlated_traits[:, 2]

    ## Build the interaction matrix given the species traits, and competitive regime specified by phi
    Aij =  model.Aij_matrix(S, Symmetric_trait, Hierarchic_trait, sigma_s, sigma_h, phi).astype('float32')

    ### I.3. Generate environmental grid
    Environment_matrix = model.Env_generation(n, structure, env_range, 10, auto_corr).astype('float32')

    ### I.4 Initialization
    Comm_matrix = model.init_commu(K, S, n)

    ### II. Run the simulation
    Results = model.Simulation_model(max_tick = 300,
                                     Community_matrix = Comm_matrix,
                                     S = S,
                                     omega = omega,
                                     Environmental_trait = Environmental_trait,
                                     Symmetric_trait = Symmetric_trait,
                                     Hierarchic_trait = Hierarchic_trait,
                                     Fecundity = Fecundity,
                                     Ext_seed_rain = Ext_seed_rain,
                                     mu = mu,
                                     We = We,
                                     Wc = Wc,
                                     K = K,
                                     Aij = Aij,
                                     Environment_matrix = Environment_matrix,
                                     n = n)
                                      
    Final_community = Results[0]
    Abundances = Results[3]
    Traits = np.vstack((Symmetric_trait, Environmental_trait, Hierarchic_trait)).T
    
    return list((Final_community, Environment_matrix, Abundances, Traits))


#%% =============================================================================
# IV. Run the simulation in parallel
# ===============================================================================    

#### /!\ If you are working on windows, for some (obscure) reason python may return:
    # ModuleNotFoundError: No module named  'model_function'.
    # It'sa bug from the parallelization process. If it's appen, run the script with parameters set for very small 
    # simulations (eg. n = 5, S = 10, max_tick = 3) in debug mode, with a break point somewhere in par_simul().
    # Then the script should run correctly in 'normal' mode, for this session/console.

## Simulation experiment can take from several minutes to several hours to run depending on :
    # Number of simulations 
    # n_jobs and your hardware
    
n_jobs = 16
start_time = time.time()

out_simu = Parallel(n_jobs = n_jobs) (delayed(par_simul)(i,
                                                         S = S,
                                                         Fecundity = Fecundity,
                                                         omega = omega, 
                                                         sigma_s = sigma_s,
                                                         sigma_h = sigma_h,
                                                         Ext_seed_rain = Ext_seed_rain,
                                                         mu = mu,
                                                         Set_of_param = Set_of_param,
                                                         K = K,
                                                         n = n,
                                                         wd = wd) for i in (range(len(Set_of_param[:,1]))))

print("Simulation experiment %s minutes ---" % ((time.time() - start_time)/60))

#%% =============================================================================
# V. Compile and export results
# =============================================================================== 

# Change wd if you want to save results in other file than the codes
wd = 'YOUR_PATH'
os.chdir(wd)

Final_community = dict()
n_time = dict()
Abundances = dict()
Traits = dict()
Environment_matrix = dict()
niche_width = dict()

for i in range(len(Set_of_param[:, 0])):
    Final_community['corr=' + str(Set_of_param[i, 0]) + '_phi=' + str(Set_of_param[i, 1]) + '_pool=' + str(Set_of_param[i, 2])] = out_simu[i][0]
    Environment_matrix['corr=' + str(Set_of_param[i, 0]) + '_phi=' + str(Set_of_param[i, 1]) + '_pool=' + str(Set_of_param[i, 2])] = out_simu[i][1]
    Abundances['corr=' + str(Set_of_param[i, 0]) + '_phi=' + str(Set_of_param[i, 1]) + '_pool=' + str(Set_of_param[i, 2])] = out_simu[i][2]
    Traits['corr=' + str(Set_of_param[i, 0]) + '_phi=' + str(Set_of_param[i, 1]) + '_pool=' + str(Set_of_param[i, 2])] = out_simu[i][3]
    
    n_time['corr=' + str(Set_of_param[i, 0]) + '_phi=' + str(Set_of_param[i, 1]) + '_pool=' + str(Set_of_param[i, 2])] = len(out_simu[i][2])


file_name = "Results_" + name + '.pkl'
open_file = open(file_name, "wb")
pickle.dump(Final_community, open_file)
open_file.close()

file_name = "Abundance_" + name + '.pkl'
open_file = open(file_name, "wb")
pickle.dump(Abundances, open_file)
open_file.close()

file_name = "Environment_" + name + '.pkl'
open_file = open(file_name, "wb")
pickle.dump(Environment_matrix, open_file)
open_file.close()

file_name = "Trait_" + name + '.pkl'
open_file = open(file_name, "wb")
pickle.dump(Traits, open_file)
open_file.close()

print(n_time)





