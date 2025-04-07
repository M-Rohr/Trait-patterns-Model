# -*- coding: utf-8 -*-
"""
Created on May 2024

@author: Matthias Rohr
"""

# Import python modules
import os 
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pickle
import time
import itertools
import matplotlib.patheffects as path_effects
from matplotlib.colors import TwoSlopeNorm

## Set working directory
wd = 'YOUR_PATH' # Path to the directory containing: model_function.py and exploit_fun.py
os.chdir(wd)
# Import function required to run the model :
import model_function as model    
import exploit_fun as exploit

# =============================================================================
# I. Simulation parameters
# =============================================================================
wd = 'YOUR_PATH' # Path to the directory containing the results of simulation_experiment.py
os.chdir(wd)
## Repeat the parameters set for Simulation_experiment
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

# The parameters and range tested for the results
correl = [0, 0.5, 0.8, 1]    # Traits correlations
phi = [0, 0.5, 1]            # Relative importance of symmetric competition vs hierarchical competition
pool_sp = list(np.linspace(1,16,16).astype('int8')) # Number of iteration for each parameter combination

Set_of_param = np.array(list(itertools.product(correl, phi, pool_sp)))

name = ['Simulation_experiment']


var_list = ['mu', 'omega',  'sigma_s', 'sigma_h', 'Wc', 'We', 'correl', 'phi', 'pool_sp']
Trait_names = ['Symmetric_trait', 'Environmental_trait', 'Hierarchic_trait', 'Multivariate_metric']

## Parameters for model analysis
n_it = 200 # Number of randomization
n_sample = 80 # Number of subsamples for each observation scale

#%% ===========================================================================
# II. Compute the SES for each simulation, at the three different scale
# =============================================================================

### /!\ If you work on windows, python may return:
### "BrokenProcessPool: A task has failed to un-serialize. Please ensure that the arguments of the function are all picklable."
### It'sa bug from the parallelization process. If it's appen, run the script with parameters set for very small 
### null model  (eg. n_it = 2, n_sample = 1) in debug mode, with a break point on line 95.
### Then exot debug mode, the script should run properly.

# Take 5 to 10 minutes to run (depending on the number of simulation, njobs value and teh hardware)
results = list()
start_time = time.time()
SES_dtf = pd.DataFrame()
njobs = 19 # Number of jobs for the parallelization 
for s in range(len(name)) :
    results.append(exploit.Out_dtf(name[s], n, S, n_it, n_sample, Trait_names, K, njobs))
    SES_dtf = pd.DataFrame()
    for i in range(len(results[0][0])):
        tempo = results[0][0][i]
        tempo['corr'] = Set_of_param[i, 0]
        tempo['phi'] = Set_of_param[i, 1]
        tempo['pool_sp'] = Set_of_param[i, 2]
        tempo['simu'] = name[s]
        SES_dtf = pd.concat((SES_dtf, tempo))
    print("SES computing time: "  + "  %s minutes ---" % ((time.time() - start_time)/60))

SES_dtf['rank'] = SES_dtf['rank'] / n_it

#%% Boxplots for fig 3

corr = [0, 0.5, 0.8]
data = SES_dtf


def custom_boxplot(data, x, y, hue, ax):
    palette = sns.color_palette('Greys', n_colors=4)
    sns.boxplot(data=data, x=x, y=y, hue=hue, ax=ax, palette=palette)
    ax.legend(title=hue, loc='upper right')
    ax.axhline(y=0, color='black', linestyle='dashed', linewidth=4)
    ax.set_xlabel('', fontsize=15)
    ax.set_ylim(ymin=-4.5, ymax=4.5)

fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, corr_val in enumerate(corr):
    data_corr = data.loc[data['corr'] == corr_val]
    data_corr = data_corr.loc[data_corr['Traits'] != 'Multivariate_metric']
    data_corr = data_corr.loc[data_corr['phi'] == 1]
    custom_boxplot(data_corr, 'scale', 'SES', 'Traits', axs[i])
    axs[i].set_ylabel('SES', fontsize=15)
    axs[i].set_title(f'Correlation = {corr_val}', fontsize=15)


plt.tight_layout()
plt.savefig('fig3.png', format="png")
plt.show()


#%% Boxplots for fig 4
phi = [0, 0.5]
data = SES_dtf
fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

for i, phi_val in enumerate(phi):
    data_phi = data.loc[data['phi'] == phi_val]
    data_phi = data_phi.loc[data_phi['corr'] == 0]
    custom_boxplot(data_phi, 'scale', 'SES', 'Traits', axs[i])
    axs[i].set_ylabel('SES', fontsize=15)
    axs[i].set_title(f'Phi = {phi_val}', fontsize=15)

plt.tight_layout()
plt.savefig('fig_4' + '.png', format="png")
plt.show()

#%% Heatmaps for fig 5
trait_names = ['Environmental_trait', 'Symmetric_trait', 'Hierarchic_trait']
scale_names = ['intra', 'local', 'global']
SES = SES_dtf
heatmaps = dict()
for t in trait_names :
    for s in scale_names:
        ses = SES.loc[(SES['Traits'] == t) & (SES['scale'] == s)]
        corel = np.unique(ses['corr'])
        sym_c = np.unique(ses['phi'])
        h_map = np.zeros((len(corel), len(sym_c)))
        for i in range(4):
            for j in range(3):
                z = ses.loc[(ses['corr'] == corel[i]) & (ses['phi'] == sym_c[j])]
                a = np.mean(z['SES'])
                m = np.mean(z['SES'])
                ic = scipy.stats.sem(z['SES'])* scipy.stats.t.ppf((1 + 0.95) / 2., len(z['SES'])-1)
                h_map[3-i, j] = a
        heatmaps[t+s] = h_map


fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes_flat = axes.flatten()

# Define color map
cmap = 'vlag'
ylabs = [1, 0.8, 0.5, 0]
xlabs = [0, 0.5, 1]

# Initialize vmin and vmax for global colorbar
vmin = float('inf')
vmax = float('-inf')

for idx, (trait, scale) in enumerate([(t, s) for t in trait_names for s in scale_names]):
    heatmap_data = heatmaps[trait + scale]
    ax = axes_flat[idx]
    sns.heatmap(heatmap_data, ax=ax, annot=True, cbar = False, cmap=cmap, fmt='.2f', vmin=-2, vmax=1.5, center = 0,
                xticklabels=xlabs, yticklabels=ylabs)
    ax.set_title(trait + ' - ' + scale)
    # Update vmin and vmax for global colorbar
    vmin = min(vmin, np.min(heatmap_data))
    vmax = max(vmax, np.max(heatmap_data))

plt.tight_layout()

norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Colorbar Label')
plt.savefig('fig_5' + '.png', format="png")
plt.show()


#%% Species diversity Vs trait correlation (fig S4)
def shannon_index(abund):
    pi = abund / np.sum(abund)
    H = - np.nansum(pi * np.log(pi))
    return np.exp(H)

div_dtf = dict()
shan_div = np.zeros(len(Final_communities))
correl_lvl = []
cption = []

for i in range(len(Final_communities)):
    A = np.sum(Final_communities[key[i]], (1,2))
    shan_div[i] = shannon_index(A)
    match = re.search(r'corr=([\d\.]+)_phi=([\d\.]+)', key[i])
    corr_value = float(match.group(1))
    phi_value = float(match.group(2))
    correl_lvl.append(corr_value)
    cption.append(phi_value)

div_dtf['shannon'] = shan_div
div_dtf['corr_value'] = correl_lvl
div_dtf['phi_value'] = cption

div_dtf = pd.DataFrame(div_dtf)
fig, axs = plt.subplots(1, 1, figsize=(10, 6), sharey=True)
palette = sns.color_palette('Greys', n_colors=4)
sns.boxplot(data=div_dtf, x='corr_value', y='shannon', hue='phi_value', palette=palette)
plt.savefig('fig_S4' + '.svg', format="svg")

print('min:', np.min(shan_div))
print('max:', np.max(shan_div))
print('mean:', np.mean(shan_div))
print('std:', np.std(shan_div))

#%% Multivariate trait pattern (fig S5)

phi = [0, 0.5, 1]
data = SES_dtf
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phi_val in enumerate(phi):
    data_phi = data.loc[data['phi'] == phi_val]
    data_phi = data_phi.loc[data_phi['Traits'] == 'Multivariate_metric']
    custom_boxplot(data_phi, 'scale', 'SES', 'corr', axs[i])
    axs[i].set_ylabel('SES', fontsize=15)
    axs[i].set_title(f'Phi = {phi_val}', fontsize=15)

plt.tight_layout()
plt.savefig('fig_S5_100pool' + '.svg', format="svg")
plt.show()

















