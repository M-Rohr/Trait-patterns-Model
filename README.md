# Trait-patterns-Model

## Fisrt install
The script run with Python 3

Activate the environment : source venv/bin/activate

upgrade pip : pip install --upgrade pip

Install the requierments : pip install -r requirements.txt


## Description
A spatially explicit individual-based model that allows varying the relative importance of environmental filtering, and symmetric and hierarchical competition in community assembly.  

Scripts description:
 
 - model_function.py: Script with all the documented function required to initialize, run and make the primary analysis of the model.
 
 - simulation_model.py: Script to run a single simulation, with a set visualization.

 - simulation_experiment.py: Script to run a simulation experiment.
   
    #/!\ If you are working on windows, for some (obscure) reason python may return: 
    #ModuleNotFoundError: No module named  'model_function'. 
    #It'sa bug from the parallelization process. If it's appen, run the script with parameters set for very small 
    #simulations (eg. n = 5, S = 10, max_tick = 3) in debug mode, with a break point somewhere in par_simul(). 
    #Then the script should run correctly in 'normal' mode, for this session/console.
    
 - exploit_fun.py: Script with all the documented function required to analyze the results from simulation_experiment.py.

 - Figs_simulation_experiment.py: Script to run the analysis of simulation experiment and generate the figure in the article. 



