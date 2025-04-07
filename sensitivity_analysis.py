# -*- coding: utf-8 -*-
"""
Created on January 2025

@author: Matthias Rohr
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

## Set working directory
wd = 'YOUR_PATH'
os.chdir(wd)
## Import function required to run the model :
import model_function as model    
#%% =============================================================================
# I. Define the fix parameters, generates environment and species pool
# ===============================================================================

### I.1. Define the fix parameters for the simulations

## Species parameters
S = 100           # Number of species
correl = 0      # Traits correlations
Ext_seed_rain = 1  #  External migration rate

## Landscape parameters
n = 50               # Size of the landscape grid
auto_corr = 5        # Autocorrelation of the environment across the landscape
structure = 'mosaic' # Structure of the environmental grid, can be: homogeneous, autocorelated ('mosaic'), random or gradient
env_range = [0, 1]   # Range of the environmental values

### I.2. Build the latin hypercube for the sensitivity analisys
## Params to test:
n_test = 200
params_values = np.array([10,   # K: Carrying capacity of the cells
                          0.2,  # mu: Mortality rate
                          0.1,  # Fecundity: Number of seed produced per individual
                          0.1,  # omega:  Environmental niche breadth of the species
                          0.1,  # sigma_s: Symmetric competition breadth
                          0.2]) # sigma_h Hierarchical competition breadth

sampler = scipy.stats.qmc.LatinHypercube(d=len(params_values))
sample = sampler.random(n = n_test)
l_bounds = params_values * 0.2 # Lower bounds of the parameters values (set at 20% of the original values)
u_bounds = params_values * 1.8 # Upper bounds of teh parameters values (set at +80% of the original values)
LHC = scipy.stats.qmc.scale(sample, l_bounds, u_bounds)
LHC[:, 0] = np.round(LHC[:, 0]) # K parameters only take integer values

### I.3. Define the type of simulation (type of competition, trait correlation)
## Simulation processes parameters
phi = .5     # Relative importance of symmetric competition vs hierarchical competition
We = 1       # Strength of the environmental filtering
Wc = 10      # Strength of the competition
correl = 0   # Traits correlations
pool_sp = 16 # Number of species pool to generate for each parameter combination 

## Output name
name = 'Sensi_analysis_phi05_corr0'


#%% ===========================================================================
# II. Definition of the paralell simulation function 
# =============================================================================

def par_simul(i, S, n, auto_corr, structure, env_range, phi, We, Wc, correl, pool_sp, LHC):                
    
    import os
    os.chdir(wd)
    import numpy as np
    import copy
    import model_function as model
    
    ### II.1. Get the parameters from the LHC
    K = LHC[i, 0]
    mu = LHC[i, 1]
    Fecundity = LHC[i, 2]
    omega = LHC[i, 3]
    sigma_s =  np.ones(S) * LHC[i, 4]
    sigma_h =  np.ones(S) * LHC[i, 5]
    
    param_res = dict()
    
    for sp in range(pool_sp):
        ### II.2. Generate the species pool
        ## Draw the three functional traits from uniform distribution, with a correl as correlation coefficient 
        correlated_traits = model.gen_corr_traits(correl, S).astype('float32')
        Symmetric_trait = correlated_traits[:, 0]
        Environmental_trait = correlated_traits[:, 1] 
        Hierarchic_trait = correlated_traits[:, 2]

        ## Build the interaction matrix given the species traits, and competitive regime specified by phi
        Aij =  model.Aij_matrix(S, Symmetric_trait, Hierarchic_trait, sigma_s, sigma_h, phi).astype('float32')

        ### II.3. Generate environmental grid
        Environment_matrix = model.Env_generation(n, structure, env_range, 10, auto_corr).astype('float32')

        ### II.4. Initialization
        Comm_matrix = model.init_commu(K, S, n)
        
        ### II.5. Run the simulation
        Results = model.Simulation_model(max_tick = 5,
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
        # Final_community = Results[0]
        # Space_occupation = Results[1]
        # Relative_abundance = Results[2]
        # ab_on_time = Results[3]         
        Traits = np.vstack((Symmetric_trait, Environmental_trait, Hierarchic_trait)).T        
        Results.append(Traits)
        param_res[sp] = Results
    return param_res
# tst = par_simul(2, S, n, auto_corr, structure, env_range, phi, We, Wc, correl, pool_sp, LHC)

#%% =============================================================================
# III. Run the simulation in parallel
# ===============================================================================    

#### /!\ If you are working on windows, for some (obscure) reason python may return:
    # ModuleNotFoundError: No module named  'model_function'.
    # It'sa bug from the parallelization process. If it's appen, run the script with parameters set for very small 
    # simulations (eg. n = 5, S = 10, max_tick = 3) in debug mode, with a break point somewhere in par_simul().
    # Then the script should run correctly in 'normal' mode, for this session/console.

## Simulation experiment can take from several minutes to several hours to run depending on :
    # Number of simulations 
    # n_jobs and your hardware

print(name)
n_jobs = joblib.cpu_count()
start_time = time.time()

out_simu = Parallel(n_jobs = n_jobs) (delayed(par_simul)(i,
                                                         S = S,
                                                         n = n,
                                                         auto_corr = auto_corr,
                                                         structure = structure,
                                                         env_range = env_range,
                                                         phi = phi,
                                                         We = We,
                                                         Wc = Wc,
                                                         correl = correl,
                                                         pool_sp = pool_sp,
                                                         LHC = LHC) for i in (range(len(LHC[:,1]))))

print("Simulation experiment %s minutes ---" % ((time.time() - start_time)/60))

## Output structure: A list with one item for each parameter combination of the LHC, 
## each item of the list is a dict with one item per simualtion realization (n = 16),
## for each realisation the results of the simulation is stored as a list with community structure and species traits.

file_name = "LHC_" + name + '.pkl'
open_file = open(file_name, "wb")
pickle.dump(LHC, open_file)
open_file.close()

file_name = "Results_" + name + '.pkl'
open_file = open(file_name, "wb")
pickle.dump(out_simu, open_file)
open_file.close()




















