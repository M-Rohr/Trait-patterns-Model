# -*- coding: utf-8 -*-
"""
Created on May 2024

@author: Matthias Rohr
"""
# Import python modules
import numpy as np
import copy
import scipy
import joblib
from joblib import Parallel, delayed
import pandas as pd

#%%
# =============================================================================
# I. Simulation set-up function (species pool and landscape)
# =============================================================================

def gen_corr_traits(r, S):
    """
    Generate correlated traits for species.
    
    Parameters:
    r (float): Correlation coefficient.
    S (int): Number of species.
    
    Returns:
    numpy.ndarray: Correlated traits matrix.
    """
    rho = 2 * np.sin(r * np.pi / 6)  # Convert correlation coefficient to appropriate scale
    P = scipy.linalg.toeplitz(np.array([1, rho, rho]))  # Create a Toeplitz matrix with the specified correlations
    d = len(P)
    U = scipy.stats.norm.cdf(np.dot(np.random.normal(size=(S, d)),
                                    scipy.linalg.cholesky(P))) # Generate correlated traits using the Cholesky decomposition 
    return U

def Aij_matrix(S, Symmetric_trait, Hierarchic_trait, sigma_s, sigma_h, phi):
    """
    Compute the interaction matrix.
    Interaction matrix adapted from Scheffer et.al., 2006.

    Parameters:
    S (int): Number of species.
    Symmetric_trait (numpy.ndarray): Symmetric traits of species.
    Hierarchic_trait (numpy.ndarray): Hierarchical traits of species.
    sigma_s (numpy.ndarray): Symmetric competition breadth.
    sigma_h (numpy.ndarray): Hierarchical competition breadth.
    phi (float): Relative importance of symmetric competition vs hierarchical competition.

    Returns:
    numpy.ndarray: Interaction matrix.
    """
    Aij_Sym = np.zeros((S, S), dtype='float32')
    Aij_Hie = np.zeros((S, S), dtype='float32')
    for i in range(S):
        for j in range(S): 
            expo = np.exp(-(Symmetric_trait[j] - Symmetric_trait[i])**2 / (4 * sigma_s[i]**2))
            numerator = scipy.special.erf((2 - Symmetric_trait[i] - Symmetric_trait[j]) / (2 * sigma_s[i])) + \
                                  scipy.special.erf((Symmetric_trait[i] + Symmetric_trait[j]) / (2 * sigma_s[i]))
            denominator = scipy.special.erf((1 - Symmetric_trait[i]) / sigma_s[i]) + \
                                    scipy.special.erf(Symmetric_trait[i] / sigma_s[i])
            
            Aij_Sym[j, i] = expo * (numerator / denominator)
            Aij_Hie[i, j] = np.exp(-1 / 2 * (((Hierarchic_trait[i] - Hierarchic_trait[j]) - 1) / sigma_h[j])**2)
            if i == j:
                Aij_Hie[i, j] = 1
                
    Aij = phi * Aij_Sym + (1 - phi) * Aij_Hie
    return Aij

def dist_torus(coord): 
    """
    Compute distance matrix for a torus 2D space.
    
    Parameters:
    coord (numpy.ndarray): Array of coordinates with shape (N, 2) where N is the number of points.
    
    Returns:
    numpy.ndarray: Distance matrix for the torus 2D space.
    """
    x = coord[:,0]
    y = coord[:,1]
    # Compute distance matrix using NumPy broadcasting
    dx = np.abs(x[:, np.newaxis] - x)
    dy = np.abs(y[:, np.newaxis] - y)
    # Compute distance matrix for a borderless 2D space
    max_x = np.max(dx) + 1 
    max_y = np.max(dy) + 1 
    mdx = max_x - dx
    mdy = max_y - dy
    dx = np.minimum(dx, mdx)
    dy = np.minimum(dy, mdy)
    d = np.sqrt(dx**2 + dy**2)
    return d

def env_set(n, R, type_, auto_corr):
    """
    Generate environmental settings for a landscape grid.

    Parameters:
    n (int): Side length of the grid.
    R (float): Range of environmental values.
    type_ (str): Type of environment ('mosaic' or 'random').
    cor_range (float): Autocorrelation of the environment across the landscape

    Returns:
    numpy.ndarray: 2D array representing the environmental values across the grid.
    """
    grid_size = n ** 2
    # Define lattice
    x = np.arange(n)
    y = np.arange(n)
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    if type_ == "mosaic":
        lyy = []
        # Generate dummy data with an exponential variogram
        z = np.random.normal(0, 0.025, grid_size)
        distance_matrix = dist_torus(xy)
        cov_matrix = 0.025 * np.exp(-distance_matrix / auto_corr)
        y_sim = np.random.multivariate_normal(z, cov_matrix)
        # Transform data from normal to uniform
        ranks = scipy.stats.rankdata(y_sim, method='ordinal')
        y_sim = scipy.stats.uniform.ppf(ranks / (grid_size + 1)) 
        # Rescale data to a given range (0 - R)
        y_sim = (y_sim - np.min(y_sim)) * R / (np.max(y_sim) - np.min(y_sim))
        lyy.append(y_sim)
    if type_ == "random":
        # Generate random uniform data
        lyy = np.linspace(0, R, grid_size)
        # Shuffle data
        np.random.shuffle(lyy)
    lyy = np.array(lyy)
    output = lyy.reshape((n, n))
    return output

def Env_generation(n, struct, env_range, n_moda, auto_corr):
    """
    Set up the generation of the Environment matrix.

    Parameters:
    n (int): Size of the landscape grid (n x n).
    struct (str): Structure of the environment ('uniform', 'random', 'gradient', 'mosaic').
    env_range (tuple or list): Range of environmental values.
    n_moda (int): Number of environmental values for the gradient structure
    auto_corr (float): Autocorrelation of the environment across the landscape for the 'mosaic' structure.

    Returns:
    numpy.ndarray: Environment matrix for the landscape grid.
    """
    if struct == 'homogeneous':
        Environment_matrix = np.ones((n, n)) * np.mean(env_range)
    elif struct == 'random':
        Environment_matrix = env_set(n, max(env_range), 'random', auto_corr)
    elif struct == 'gradient':
        dta_env = np.repeat(np.linspace(min(env_range), max(env_range), n_moda), n**2 / n_moda)  # Abrupt gradient
        Environment_matrix = dta_env.reshape((n, n)).astype(np.float32)
    elif struct == 'mosaic':
        Environment_matrix = env_set(n, max(env_range), 'mosaic', auto_corr)
    
    return Environment_matrix

def init_commu(K, S, n):
    """
    Set the initial state with equal abundance for all species and random positions for individuals.

    Parameters:
    K (int): Carrying capacity of the cells.
    S (int): Number of species.
    n (int): Size of the landscape grid (n x n).

    Returns:
    numpy.ndarray: 3D array representing the initial community with shape (S, n, n).
    """
    p = np.ones(S) / S  # Equal probability for all species
    commu = np.random.multinomial(K, p, size=n**2)  # Randomly distribute individuals
    return commu.T.reshape(S, n, n)  # Reshape to (S, n, n) format

# =============================================================================
# II. Simulation model function
# =============================================================================


def seed_competition(Community_matrix, Aij, S, n):
    """
    Compute seed competition.

    Parameters:
    Community_matrix (numpy.ndarray): Matrix with the position of each individual.
    Aij (numpy.ndarray): Interaction matrix.
    S (int): Number of species.
    n (int): Size of the space grid.

    Returns:
    numpy.ndarray: Seed competition values.
    """
    Commu = Community_matrix.reshape(-1, n**2).T.astype('float32')
    seed_comp = np.zeros((n**2, S), dtype='float32')
    seed_comp = np.dot(Commu, Aij)
    return seed_comp

    
def multiprocess_lottery(dispo, S, seed_proba, commu, Ext_seed_rain, Fecundity):
    """
    Perform lottery competition for seed germination.

    Parameters:
    dispo (numpy.ndarray): Array listing the cells with empty space, with the number of spaces in each cell. 
    S (int): Number of species.
    seed_proba (numpy.ndarray): Germination probabilities of seeds at different positions.
    commu (numpy.ndarray): Community matrix.
    Ext_seed_rain (float): External seed rain rate.
    Fecundity (float): Fecundity rate.

    Returns:
    numpy.ndarray: Results matrix containing the winners of the lottery competition.
    """
    out = np.zeros((len(dispo[0, :]), S), dtype='int8')  # Set up the results matrix
    for i in range(len(dispo[0, :])):
        location_seed = dispo[0, i]  # Compute the position research
        weights = seed_proba[location_seed, :]  # Get the seed's germination probabilities
        if (np.sum(weights) != 0) & (not np.isnan(weights).any()):  # Check if there is at least one non-zero probability
            Sp_colo = np.arange(0, S)  # Get the identity of the species
            weights = weights * (np.sum(commu, 0) * Fecundity + Ext_seed_rain)
            # weights = weights * (commu[location_seed,:] * Fecundity + Ext_seed_rain) # Uncomment if you want to test local dispersion only
            weights = weights / np.sum(weights)
            win_sp = np.random.choice(Sp_colo, int(min(dispo[1, i], len(Sp_colo[weights > 0]))), p=weights, replace=True)  # Select the winners given the relative probability
            Sp_ID, count = np.unique(win_sp, return_counts=True)  # Which species is winning and how many win per species?
            out[i, Sp_ID] = count  # Stock the result of the lottery
    return out


# def environmental_filter(Ti, Ek, omega, norm_env):
#     omega0 = 1 / (np.sqrt(2 * np.pi) * omega)
   
#     Etal_fit = omega0 * np.exp(- (Ti[np.newaxis,] - Ek[:, np.newaxis])**2 / (2 * omega**2))
#     return Etal_fit


def Colonization(Community_matrix, Environmental_trait, omega, Environment_matrix,
                 n, S, Aij, We, Wc, Fecundity, Ext_seed_rain, K):
    """
    Seed production and colonization function.

    Calculate the number of seeds produced per species, attribute to each seed a position in the space,
    and then compute the germination probability of each seed, updating the Community_matrix.

    Parameters:
    Community_matrix (numpy.ndarray): Matrix representing the community.
    Environmental_trait (numpy.ndarray): Environmental traits of species.
    omega (float): Width of the species niche.
    Environment_matrix (numpy.ndarray): Matrix representing the environmental conditions.
    n (int): Size of the space grid.
    S (int): Number of species.
    Aij (numpy.ndarray): Interaction matrix.
    We (float): Weight of the environmental filter.
    Wc (float): Weight of the competition.
    Fecundity (float): Number of seed produced per individual.
    Ext_seed_rain (float): External migration rate.
    K (int): Carrying capacity of the cells.

    Returns:
    list: Updated Community_matrix.
    """
    # Reshape the community matrix
    commu = Community_matrix.reshape(-1, n**2).T

    # Compute the environmental filtering
    Seed_filter = np.exp(- (Environmental_trait[np.newaxis,] - Environment_matrix.reshape(n**2, 1)[:, np.newaxis])**2 / (2 * omega**2))
    # Seed_filter =environmental_filter(Environmental_trait, Environment_matrix, omega)
    
    Seed_filter = Seed_filter.astype(np.float32).reshape((n**2, S))
    
    # Compute the competitive interaction
    seed_comp = np.zeros((n**2, S), dtype='float32')
    if Wc > 0:
        seed_comp = seed_competition(Community_matrix, Aij, S, n)

        # Normalization of the interaction by the number of effective competitors
        ab_per_cell = np.sum(Community_matrix.reshape(-1, n**2).T, 1)
        ab_per_cell[np.where(ab_per_cell == 0)] = 1  # Avoid NaN probability, the value of normalization is meaningless for cells with no competitors (seed_comp = 0)
        seed_comp = 1 - (seed_comp / ab_per_cell[:, np.newaxis])  # Set the experienced competition as a percentage of the maximum competition possible given the abundance in cell

        # Compute the germination probability, with the weight of each process
        with np.errstate(divide='ignore'):
            seed_proba = np.exp(We * np.log(Seed_filter) + Wc * np.log(seed_comp))
    else:
        # If W_comp=0 compute the germination probability without the competition effect
        seed_proba = np.exp(We * np.log(Seed_filter))
    # Get the number of available spaces for each cell
    dispo = K - np.sum(commu, 1)
    dispo = np.vstack((np.arange(0, n**2), dispo))
    dispo = dispo[:, dispo[1, :] > 0].astype('int32')
    # Perform the lottery
    Res_lottery = multiprocess_lottery(dispo, S, seed_proba, commu, Ext_seed_rain, Fecundity)
    # Add new individuals to the community matrix
    for i in range(len(dispo[0, :])):
        commu[dispo[0, i], :] += Res_lottery[i, :]
    Community_matrix = commu.T.reshape(S, n, n)

    return Community_matrix

def Simulation_model(max_tick, Community_matrix, S, omega,
                     Environmental_trait, Symmetric_trait, Hierarchic_trait,
                     Fecundity, Ext_seed_rain, mu, We, Wc, K, Aij,
                     Environment_matrix, n):
    """
    Run the community assembly simulation model.

    Parameters:
    max_tick (int): Maximum number of ticks
    Community_matrix (numpy.ndarray): Matrix containing the position of each individual of each species.
    S (int): Number of species.
    omega (float): Breatdh of the species niche.
    Environmental_trait (numpy.ndarray): Environmental trait values of species.
    Symmetric_trait (numpy.ndarray): Symmetric competition trait values of species.
    Hierarchic_trait (numpy.ndarray): Hierarchical competition trait values of species.
    Fecundity (float): Number of seed produced per individual.
    Ext_seed_rain (float): External migration rate.
    mu (float): Individual mortality rate.
    We (float): Weight on environmental filter.
    Wc (float): Weight on competition.
    K (int): Carrying capacity of the cells.
    Aij (numpy.ndarray): Interaction matrix (with the mix of symmetric and hierarchical competition).
    Environment_matrix (numpy.ndarray): Matrix containing the environmental value of each cell.
    n (int): Size of the space.

    Returns:
    list: Update of the community tensor, occupation, density, abundances.
    """
    Community_matrix = copy.deepcopy(Community_matrix)
    density = np.sum(Community_matrix, (1, 2)) / np.sum(Community_matrix)
    occupation = np.sum(np.sum(Community_matrix, (0)) > 0) / n**2
    Abundances = np.sum(Community_matrix, (1, 2))
    tick = 0

    while (tick <= max_tick):
        # Individual random mortality
        Community_matrix[np.where(Community_matrix > 0)] = np.random.binomial(Community_matrix[np.where(Community_matrix > 0)], 1 - mu,
                                                                              len(Community_matrix[np.where(Community_matrix > 0)]))

        # Seed production and colonization
        Community_matrix = Colonization(Community_matrix = Community_matrix,
                           Environmental_trait = Environmental_trait,
                           omega = omega,
                           Environment_matrix = Environment_matrix,
                           n = n,
                           S = S,
                           Aij = Aij,
                           We = We,
                           Wc = Wc,
                           Fecundity = Fecundity,
                           Ext_seed_rain = Ext_seed_rain,
                           K = K)

        density = np.vstack((density, np.sum(Community_matrix, (1, 2)) / np.sum(Community_matrix)))
        occupation = np.vstack((occupation, np.sum(np.sum(Community_matrix, (0)) > 0) / n**2))
        Abundances = np.vstack((Abundances, np.sum(Community_matrix, (1, 2))))

        tick = tick + 1

    return list((Community_matrix, occupation, density, Abundances))

# =============================================================================
# III. Model analysis function 
# =============================================================================

def shannon_div(rel_ab):
    """
    Compute the Shannon diversity index.

    Parameters:
    rel_ab (numpy.ndarray): Relative abundances of species.

    Returns:
    numpy.ndarray: Shannon diversity index for each time step.
    """
    # Compute log(pi) and set to 0 where pi = 0
    with np.errstate(divide='ignore'):
        log_rel = np.log(rel_ab)
        log_rel[np.isinf(log_rel)] = 0   
        Shannon_idx = np.exp(- np.sum(rel_ab * log_rel, axis=1))
    return Shannon_idx

def RAO(community_matrix, trait):
    """
    Calculate RAO quadratic entropy (a functional diversity metric).

    Parameters:
    community_matrix (array): Community matrix with samples as rows and species as columns.
    trait (array): An S by n matrix with the traits values of the S species in an n-dimensional space (1 to n traits).

    Returns:
    array: RAO's quadratic entropy of each sample of the community matrix.
    """
    # Ensure community_matrix is of type int64
    community_matrix = community_matrix.astype('int64')
    
    # Scale the trait data
    trait_sc = (trait - np.mean(trait, axis=0)) / np.std(trait, axis=0, ddof=1)
    
    # Compute pairwise Euclidean distances
    trait_mat = scipy.spatial.distance.pdist(trait_sc, 'euclidean')
    
    # Convert the condensed distance matrix to a square distance matrix
    mat_dist_square = scipy.spatial.distance.squareform(trait_mat)
    
    # Initialize the RAO array
    RAO = np.zeros(len(community_matrix[:, 0]), dtype='float32')
    
    # Calculate RAO for each sample
    for i in range(len(community_matrix[:, 0])):
        RAO[i] = np.dot(np.dot(community_matrix[i, :], mat_dist_square**2), 
                        community_matrix[i, :].T) / (2 * (np.sum(community_matrix[i, :])**2))
    
    return RAO

def par_RAO(i, community_matrix, trait):
    """
    Parallelisable version of the RAO function
    """
    community_matrix = community_matrix[i, :, :].astype('int64')
    # Scale the data
    trait_sc = (trait - np.mean(trait, axis=0)) / np.std(trait, axis=0, ddof=1)
    # Compute pairwise Euclidean distances
    trait_mat = scipy.spatial.distance.pdist(trait_sc, 'euclidean')

    # If you want to convert the condensed distance matrix to a square distance matrix
    mat_dist_square = scipy.spatial.distance.squareform(trait_mat)
    RAO = np.zeros(len(community_matrix[:,0]), dtype = 'float32')
    for i in range(len(community_matrix[:,0])):
        RAO[i] = np.dot(np.dot(community_matrix[i,:], mat_dist_square**2), 
                        community_matrix[i,:].T) / 2 / (np.sum(community_matrix[i,:])**2)
    return RAO

def EQ_R_cst_ric(community_matrix, iteration):
    """
    Null model C2 (Götzenberger et al. 2016), randomization of the rows of each column independently.

    Parameters:
    community_matrix (array): Community matrix with sites as rows and species as columns.
    iteration (int): Number of randomizations.

    Returns:
    array: Randomized community matrices, each slice is an iteration of the randomization process.
    """
    rng = np.random.default_rng()
    shuffled_matrix = np.zeros((iteration, community_matrix.shape[0], community_matrix.shape[1]), dtype='int32')
    for i in range(iteration):
        shuffled_matrix[i, :, :] = rng.permuted(community_matrix, axis=1)
    return shuffled_matrix

def local_sampling(n, S, commu_matrix, grain):
    """
    Sample local community data from the larger community matrix.

    Parameters:
    n (int): Size of the spatial grid (assumed to be square).
    S (int): Number of species.
    commu_matrix (array): The community matrix with dimensions (S, space_size, space_size).
    grain (int): The size of the local sampling grain.

    Returns:
    array: Sampled local community matrix with dimensions (9, S).
    """
    # Calculate the space between each sampling window
    x = int((n - 3 * grain) / 4)
    # Initialize the sampled matrix
    sampled_matrix = np.zeros((9, S), dtype='int32')
    count = 0
    for i in range(3):
        for j in range(3):
            # Determine the starting row and column for the current sampling window
            row = ((i + 1) * x + i * grain)
            col = ((j + 1) * x + j * grain)
            # Extract and sum the community data within the current sampling window
            window = commu_matrix[:, row : (row + grain), col : (col + grain)].reshape(-1, grain**2).T
            sampled_matrix[count, :] = np.sum(window, 0)
            count += 1
    
    return sampled_matrix

def obs_rnd_sampling(obs_commu, n_sample, n_it, S, k):
    """
    Perform observed and randomized sampling on the community matrix.

    Parameters:
    obs_commu (array): Observed community matrix with sites as rows and species as columns.
    n_sample (int): Number of samples to be drawn per site.
    n_it (int): Number of randomization iterations.
    S (int): Number of species.
    k (int): Number of individuals to be sampled.

    Returns:
    tuple: Observed samples and randomized samples.
    """
    n_site = obs_commu.shape[0]
    obs_sample = np.zeros((n_site * n_sample, S), dtype='int16')    
    row = 0
    for i in range(n_site):
        weights = obs_commu[i, :] / np.sum(obs_commu[i, :])
        for j in range(n_sample):
            samp = np.random.choice(range(S), k, p=weights, replace=True)
            sp_id, ind = np.unique(samp, return_counts=True)
            obs_sample[row, sp_id] = ind
            row += 1

    rnd_sample = EQ_R_cst_ric(obs_sample, n_it)
    return obs_sample, rnd_sample

def SES_calc(obs_RAO, rnd_RAO, traits_name):
    """
    Calculate Standardized Effect Size (SES) for RAO quadratic entropy.

    Parameters:
    obs_RAO (dict): Observed RAO values for different traits.
    rnd_RAO (dict): Randomized RAO values for different traits.
    traits_name (list): List of trait names.

    Returns:
    dict: SES values for each trait.
    """
    SES = dict()
    for t in traits_name:
        SES[t] = (obs_RAO[t] - np.mean(rnd_RAO[t], 1)) / np.std(rnd_RAO[t], 1)
    return SES

def dtf_fun(SES, traits_name):
    """
    Create a DataFrame-like structure from SES values for different traits.

    Parameters:
    SES (dict): SES values for each trait.
    traits_name (list): List of trait names.

    Returns:
    array: Array with SES values and corresponding trait names.
    """
    dtf = list()
    trait = list()
    for k in range(len(traits_name)):
        dtf.append(SES[traits_name[k]])
        trait.append(np.repeat(traits_name[k], len(SES[traits_name[k]])))
    dtf = np.hstack(dtf)
    trait = np.hstack(trait)
    out = np.vstack((dtf, trait)).T
    return out

def scale_RAO(community_matrix, traits, traits_name, scale, n_it, n_sample, k, S):
    """
    Calculate the SES (Standardized Effect Size) for RAO quadratic entropy at different scales.

    Parameters:
    community_matrix (array): Community matrix with sites as rows and species as columns.
    traits (list): List of trait matrices.
    traits_name (list): List of trait names.
    scale (str): Scale of analysis ('intra' or 'local' or 'global').
    n_it (int): Number of randomizations.
    n_sample (int): Number of samples.
    k (int): Number of individuals to sample.
    S (int): Number of species.

    Returns:
    DataFrame: DataFrame with SES values, traits, and scale.
    """
    # Get the sampled community matrix and its randomized mirror
    if scale == 'intra':
        sample_site = np.random.choice(range(len(community_matrix[:, 0])), n_sample, replace=False)
        obs_samp = community_matrix[sample_site, :]
        rnd_samp = EQ_R_cst_ric(community_matrix[sample_site, :], n_it)
    else:    
        obs_samp, rnd_samp = obs_rnd_sampling(community_matrix, n_sample, n_it, S, k)

    n_jobs = joblib.cpu_count() - 2
    # Compute the RAO diversity for all observed and random samples
    obs_RAO = dict()
    rnd_par_RAO = dict()
    for t in range(len(traits)):
        obs_RAO[traits_name[t]] = RAO(obs_samp, traits[t])
        rnd_par_RAO[traits_name[t]] = Parallel(n_jobs=n_jobs, backend = 'threading')(delayed(par_RAO)(i, rnd_samp, traits[t]) for i in range(n_it))

    rnd_RAO = dict()
    for t in range(len(traits)):
        rnd_RAO[traits_name[t]] = np.vstack(rnd_par_RAO[traits_name[t]]).T

    SES = SES_calc(obs_RAO, rnd_RAO, traits_name)
    SES = dtf_fun(SES, traits_name)
    dtf_SES = pd.DataFrame(SES, columns=['SES', 'Traits'])
    dtf_SES['SES'] = SES[:, 0].astype('float32')
    dtf_SES['scale'] = scale
    return dtf_SES

















