# Trait-patterns-Model

## Fisrt install
The script run with Python 3
- Set your directory: `import os` ` os.chdir("YOUR_PATH/Trait-patterns-Model")` 
- upgrade pip : `pip install --upgrade pip`
- Install the requierments : `pip install -r requirements.txt`

## Description
A spatially explicit individual-based model that allows varying the relative importance of environmental filtering, and symmetric and hierarchical competition in community assembly.  

Scripts description:
 
 - model_function.py: Script with all the documented function required to initialize, run and make the primary analysis of the model.
 
 - simulation_model.py: Script to run a single simulation, with a set visualization.

 - simulation_experiment.py: Script to run a simulation experiment.
   
    #/!\ If you are working on windows, python may return: \
    #ModuleNotFoundError: No module named  'model_function'. \
    #It'sa bug from the parallelization process. If it's appen, run the script with parameters set for very small \
    #simulations (eg. n = 5, S = 10, max_tick = 3) in debug mode, with a break point somewhere in par_simul(). \
    #Then the script should the script should run properly.
    
 - exploit_fun.py: Script with all the documented function required to analyze the results from simulation_experiment.py.

 - Figs_simulation_experiment.py: Script to run the analysis of simulation experiment and generate the figure in the article.
   
     #/!\ If you work on windows, python may return: \
     #"BrokenProcessPool: A task has failed to un-serialize. Please ensure that the arguments of the function are all picklable." \
     #It'sa bug from the parallelization process. If it's appen, run the script with parameters set for very small \
     #null model  (eg. n_it = 2, n_sample = 1) in debug mode, with a break point on line 95. \
     #Then exit debug mode, the script should run properly.
   
 - sensitivity_analysis.py: Script to perform the sensitivity analysis of the model's fix parameter. Since the analysis is quite intensive, we recomend to use a cluster for this script.
 - exploit_sensitivity_analysis.py: Script to compute the results from the sensitivity analysis and save result as pkl file to be used to generate figures.
 - fig_sensi_analysis.py: Script to generate the figure of the sensitivity analysis (supplementary material, part 1).
 



