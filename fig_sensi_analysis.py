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
import copy
import itertools
import matplotlib.patheffects as path_effects
from matplotlib.colors import TwoSlopeNorm

## Set working directory
wd = 'C:\\Users\\rohrm\\Documents\\Thèse LECA\\articles\\papier modele assemblage\\Codes'
os.chdir(wd)

# Import function required to run the model :
import model_function as model    
import exploit_fun as exploit

corr = 0 
phi = 0.5
name = 'Sensi_analysis_phi05_corr0'
Trait_names = ['Symmetric_trait', 'Environmental_trait', 'Hierarchic_trait', 'Multivariate_metric']

# Load Res sensi analysis
file_name = 'sensi_analysis_res\\SES_' + name + '.pkl'
open_file = open(file_name, "rb")
SES_dtf = pickle.load(open_file)
open_file.close()

# Load LHC parms combination
file_name = 'sensi_analysis_res\\LHC_' + name + '.pkl'
open_file = open(file_name, "rb")
LHC_dtf = pickle.load(open_file)
open_file.close()

# Load res simu exp
file_name = 'verif_result_consistency\\SES_Simulation_experiment.pkl'
open_file = open(file_name, "rb")
SES_paper = pickle.load(open_file)
open_file.close()

#%% Get mean SES paper

SES_paper_hier = SES_paper[SES_paper['Traits'] == 'Hierarchic_trait']
SES_paper_sym = SES_paper[SES_paper['Traits'] == 'Symmetric_trait']
SES_paper_env = SES_paper[SES_paper['Traits'] == 'Environmental_trait']

SES_paper_hier = SES_paper_hier[(SES_paper_hier['corr'] == corr) & (SES_paper_hier['phi'] == phi)]
SES_paper_sym = SES_paper_sym[(SES_paper_sym['corr'] == corr) & (SES_paper_sym['phi'] == phi)]
SES_paper_env = SES_paper_env[(SES_paper_env['corr'] == corr) & (SES_paper_env['phi'] == phi)]

SES_paper_hier_mean = SES_paper_hier.groupby('scale').mean().reset_index()
SES_paper_sym_mean = SES_paper_sym.groupby('scale').mean().reset_index()
SES_paper_env_mean = SES_paper_env.groupby('scale').mean().reset_index()

#%%
# for i in range(len(SES_dtf)):
tst = SES_dtf.iloc[:, 4:10].values
par_comb = np.where(np.all(tst[:, np.newaxis]==LHC_dtf, 2))[1]
SES_dtf['params_comb'] = par_comb

#%% Get mean SES sensi analysis
SES_sens_hier = SES_dtf[SES_dtf['Traits'] == 'Hierarchic_trait']
SES_sens_sym = SES_dtf[SES_dtf['Traits'] == 'Symmetric_trait']
SES_sens_env = SES_dtf[SES_dtf['Traits'] == 'Environmental_trait']

SES_sens_hier_mean = SES_sens_hier.groupby(['scale', 'params_comb']).mean().reset_index()
SES_sens_sym_mean = SES_sens_sym.groupby(['scale', 'params_comb']).mean().reset_index()
SES_sens_env_mean = SES_sens_env.groupby(['scale', 'params_comb']).mean().reset_index()
#%% Delta SES
def delta_fun(Delta_SES, paper_dtf, scale):
    a = np.array(Delta_SES[Delta_SES['scale'] == scale]['SES'])
    b = np.array(paper_dtf[paper_dtf['scale'] == scale]['SES'])
    c= a-b
    Delta_SES.loc[Delta_SES['scale'] == scale, 'Delta_SES'] = c
    return Delta_SES

Delta_SES_hier = copy.deepcopy(SES_sens_hier_mean)
Delta_SES_hier['Delta_SES'] = 0
Delta_SES_sym = copy.deepcopy(SES_sens_sym_mean)
Delta_SES_sym['Delta_SES'] = 0
Delta_SES_env = copy.deepcopy(SES_sens_env_mean)
Delta_SES_env['Delta_SES'] = 0
scales = ['intra', 'local', 'global']
for scale in scales:
    Delta_SES_hier = delta_fun(Delta_SES_hier, SES_paper_hier_mean, scale)
    Delta_SES_sym = delta_fun(Delta_SES_sym, SES_paper_sym_mean, scale)
    Delta_SES_env = delta_fun(Delta_SES_env, SES_paper_env_mean, scale)

#%% Plot the result
# Symmetric trait
fig, axs = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
def custom_boxplot(data, x, y, hue, ax):
    # palette = sns.color_palette('Greys', n_colors=4)
    sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax)
    ax.legend(title=hue, loc='upper right')
    ax.axhline(y=0, color='black', linestyle='dashed', linewidth=4)
    ax.axvline(x=np.mean(data[x]), color = 'red', linestyle='dashed', linewidth= 2)
    ax.set_xlabel('', fontsize=15)
    ax.set_ylim(ymin=-3, ymax=3)

params = Delta_SES_env.columns[4:10]
for i, ax in enumerate(axs.flat):  
    if i < len(params):
        custom_boxplot(data=Delta_SES_sym, x=params[i], y='Delta_SES', hue='scale', ax=ax)
        ax.set_xlabel(params[i], fontsize=15)
        if i == 1:
            ax.set_title('Symmetric trait patterns', fontsize=15)
            
plt.tight_layout()
plt.savefig('Sensi_analysis_sym_trait.svg', format="svg")
plt.show()
#%% Environmental traits
fig, axs = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
for i, ax in enumerate(axs.flat):  
    if i < len(params):
        custom_boxplot(data=Delta_SES_env, x=params[i], y='Delta_SES', hue='scale', ax=ax)
        ax.set_xlabel(params[i], fontsize=15)
        if i == 1:
            ax.set_title('Environmental trait patterns', fontsize=15)
plt.tight_layout()
plt.savefig('Sensi_analysis_env_trait.svg', format="svg")
plt.show()

#%% Hierarchic traits
fig, axs = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
for i, ax in enumerate(axs.flat):  
    if i < len(params):
        custom_boxplot(data=Delta_SES_hier, x=params[i], y='Delta_SES', hue='scale', ax=ax)
        ax.set_xlabel(params[i], fontsize=15)
        if i == 1:
            ax.set_title('Hierarchic trait patterns', fontsize=15)

plt.tight_layout()
plt.savefig('Sensi_analysis_hier_trait.svg', format="svg")
plt.show()

#%% PLot absolute value istead of deltas

# Symmetric trait
fig, axs = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
params = Delta_SES_env.columns[4:10]
for i, ax in enumerate(axs.flat):  
    if i < len(params):
        custom_boxplot(data=SES_sens_sym_mean, x=params[i], y= 'SES', hue='scale', ax=ax)
        ax.set_xlabel(params[i], fontsize=15)
        if i == 1:
            ax.set_title('Symmetric trait patterns', fontsize=15)

#%% Environment trait
fig, axs = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
params = Delta_SES_env.columns[4:10]
for i, ax in enumerate(axs.flat):  
    if i < len(params):
        custom_boxplot(data=SES_sens_env_mean, x=params[i], y= 'SES', hue='scale', ax=ax)
        ax.set_xlabel(params[i], fontsize=15)
        if i == 1:
            ax.set_title('Environment trait patterns', fontsize=15)

#%% Hierarchic trait
fig, axs = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
params = Delta_SES_env.columns[4:10]
for i, ax in enumerate(axs.flat):  
    if i < len(params):
        custom_boxplot(data=SES_sens_hier_mean, x=params[i], y= 'SES', hue='scale', ax=ax)
        ax.set_xlabel(params[i], fontsize = 15)
        if i == 1:
            ax.set_title('Hierarchic trait patterns', fontsize=15)




