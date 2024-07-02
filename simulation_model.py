# -*- coding: utf-8 -*-
"""
Created on May 2024

@author: Matthias Rohr
"""
# Import python modules
import os 
import numpy as np
import matplotlib.pyplot as plt
import time
import mkl
import pandas as pd
import seaborn as sns

## Set working directory
wd = 'C:\\Users\\rohrm\\Documents\\Thèse LECA\\articles\\papier modele assemblage\\Codes'
# wd = '//bettik/PROJECTS/pr-teembio/rohrm/simu_python/explo_param/test_model_norma/d=0'
os.chdir(wd)
# Import function required to run the model :
import model_function as model        
#%% =============================================================================
# I. Define the fix parameters, generates environment and species pool
# ===============================================================================

### I.1. Define the fix parameters for the simulations

## Species parameters
S = 100           # Number of species
mu = 0.2          # Mortality rate
Fecundity = 0.1   # Number of seed produced per individual
omega =  0.1      # Environmental niche breadth of the species
sigma_s = np.ones(S) * 0.1  # Symmetric competition breadth
sigma_h = np.ones(S) * 0.2  # Hierarchical competition breadth
correl = 0      # Traits correlations
Ext_seed_rain = 1  #  External migration rate

## Landscape parameters
n = 50               # Size of the landscape grid
auto_corr = 5        # Autocorrelation of the environment across the landscape
K = 10               # Carrying capacity of the cells
structure = 'mosaic' # Structure of the environmental grid, can be: homogeneous, autocorelated ('mosaic'), random or gradient
env_range = [0, 1]   # Range of the environmental values

# Simulation processes parameters
phi = .5           # Relative importance of symmetric competition vs hierarchical competition
We = 1            # Strength of the environmental filtering
Wc = 0            # Strength of the competition


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


### I.5 Graphical representation of the interaction matrix and environmental grid

f,(ax1,ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios':[1,1]}, figsize  = (12,5))
g1 = sns.heatmap(Aij,cmap="viridis",cbar=True,ax=ax1)
g1.set_title('Interaction matrix', fontsize = 15) 
g1.set_ylabel('')
g1.set_xlabel('')
g2 = sns.heatmap(Environment_matrix,cmap="viridis",cbar=True,ax=ax2)
g2.set_title('Environmental grid', fontsize = 15) 
g2.set_ylabel('')
g2.set_xlabel('')

#%% ===========================================================================
# II. Simulation execution
# =============================================================================

# Take between 2 and 5 minutes to run
mkl.set_dynamic(True)
start_time = time.time()
Results = model.Simulation_model(max_tick = 3000,
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
                                  
print("Simulation %s minutes ---" % ((time.time() - start_time)/60))
Final_community = Results[0]
Space_occupation = Results[1]
Relative_abundance = Results[2]
ab_on_time = Results[3]

#%% =============================================================================
# III. Results exploitation
# =============================================================================
### III.1. Primary visualization of the community 

Abundance_rank = np.sum(Final_community, (1,2))
Abundance_rank = np.sort(Abundance_rank)[::-1]

print ('Shannon diversity at gloabl scale: '  + str(model.shannon_div((np.sum(Final_community, (1,2))/np.sum(np.sum(Final_community, (1,2)))).reshape((1, S)))))
#Plot of the abundance rank graph:
plt.figure(figsize=(27, 10))
plt.subplot(2, 2, 1)
plt.bar(np.arange(S), Abundance_rank)
plt.xlabel('Species', size = 25)
plt.ylabel('Abundances', size = 25)
# Plot of the relative abundances of species over time:
plt.subplot(2, 2 ,2)
plt.plot(Relative_abundance)
plt.xlabel('Time steps', size = 25)
plt.ylabel('Relative abundance', size = 25)

#%% III.2. Plot trait distribution

plt.figure(figsize=(12, 7))
plt.subplot(1, 3, 1)
plt.bar(Environmental_trait, np.sum(Final_community, (1,2)), width= 0.005, label = 'E_filter')
plt.bar(Environmental_trait, 10, width= 0.005)
plt.title('Environmental trait', fontsize = 15)
plt.subplot(1, 3, 2)
plt.bar(Symmetric_trait, np.sum(Final_community, (1,2)), width=0.005,label = 'Niche_diff')
plt.bar(Symmetric_trait, 10, width= 0.005)
plt.title('Symmetric trait', fontsize = 15)
plt.subplot(1, 3, 3)
plt.bar(Hierarchic_trait, np.sum(Final_community, (1,2)), width=0.005,label = 'Niche_diff')
plt.bar(Hierarchic_trait, 10, width= 0.005)
plt.title('Hierarchic trait', fontsize = 15)

#%% III.3. Rao diversities on time

rao_Etal = model.RAO(ab_on_time, np.vstack((Environmental_trait)))
rao_Niche = model.RAO(ab_on_time, np.vstack((Symmetric_trait)))
rao_Hier = model.RAO(ab_on_time, np.vstack((Hierarchic_trait)))

plt.plot(rao_Etal, color = 'teal', label = 'Environmental_trait')
plt.plot(rao_Niche, color = 'darkred', label = 'Symmetric trait')
plt.plot(rao_Hier, color = 'dodgerblue', label = 'Hierarchic trait')
plt.legend(loc = 'upper left')
plt.ylabel('Rao quadratic entropy', fontsize = 15)
plt.xlabel('Time', fontsize = 15)

#%% III.4. Compute and plot the SES (Standardized Effect Size), to visualize the functionnal patterns

n_it = 200  # Number of randomizations for the null models
n_sample = 200  # Number of samples for each scale

# Reshape the community matrix with species as columns and sites as rows
Obs_out = Final_community.reshape(-1, n**2).T   

# Define the community matrix from the local sampling (9 10x10 plots evenly spaced in the landscape)
local_samples = model.local_sampling(n, S, Final_community, grain=10)  

# Total abundances of the species
sum_obs = np.sum(Obs_out, 0).reshape((1, S))  

# Reshape the traits array to compute functional diversities
trait_E = np.vstack((Environmental_trait))
trait_S = np.vstack((Symmetric_trait))
trait_H = np.vstack((Hierarchic_trait))
trait_Full = np.vstack((Environmental_trait, Symmetric_trait, Hierarchic_trait)).T

traits = [trait_E, trait_S, trait_H, trait_Full]
traits_name = ['Environmental_trait', 'Symmetric_trait', 'Hierarchic_trait', 'Multivariate_metric']

# Compute the SES for each individual trait for the three observation scales
start_time = time.time()
intra_SES = model.scale_RAO(Obs_out, traits, traits_name, 'intra', n_it, n_sample, K, S)
print("Intra" + " %s minutes ---" % ((time.time() - start_time) / 60))

local_SES = model.scale_RAO(local_samples, traits, traits_name, 'local', n_it, n_sample, K, S)
print("Local" + " %s minutes ---" % ((time.time() - start_time) / 60))

global_SES = model.scale_RAO(sum_obs, traits, traits_name, 'global', n_it, n_sample, K, S)
print("Global" + " %s minutes ---" % ((time.time() - start_time) / 60))

# Compile results in a DataFrame
dtf_SES = pd.concat((intra_SES, local_SES, global_SES))

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
data = dtf_SES.loc[dtf_SES['Traits'] != 'trait_B']
palette = sns.color_palette('Greys', n_colors=4)
sns.boxplot(data=data, x='scale', y='SES', hue='Traits', ax=ax, palette=palette)

yabs_max = abs(max(ax.get_ylim(), key=abs))
ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
ax.axhline(y=0, color='black', linestyle='dashed')
plt.show()








