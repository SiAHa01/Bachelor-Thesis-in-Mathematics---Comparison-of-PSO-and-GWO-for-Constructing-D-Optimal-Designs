import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import Loc_D_Opt as ldo
from Loc_D_Opt import compress_design


"""
Goal:

    - Simulate responses from the Michaelis-Menten and Emax models using the D-optimal 
      design obtained from PSO:
                        y1* = η(x1*; θ) + ε1,   ε1 ~ N(0, σ²)
                        ...
                        yn* = η(xk*; θ) + εn,   εn ~ N(0, σ²)

                                        e.g. σ² = 0.5

    - Simulate responses from the Michaelis-Menten and Emax models using non-optimal 
      design (e.g., equally spaced doses) for comparison.
                        y1 = η(x1; θ) + ε1,   ε1 ~ N(0, σ²)
                        ...
                        yn = η(xk; θ) + εn,   εn ~ N(0, σ²)

                                        e.g. σ² = 0.5

    - Iteratively generate multiple datasets for each design and model, and analyze the 
      variability in the estimated parameters.
"""


# ============================================================
# Settings
# ============================================================


# Assumed variance of the noise in the simulated responses
sigma2 = 0.5 

# Convert to standard deviation, since random normal generation uses σ
sigma = np.sqrt(sigma2)

# Number of simulations
n_sims = 1000

# Sample size n for each study
sample_sizes = [24, 48, 72, 96]  # Different sample sizes to analyze the effect of sample size on estimation quality


### True parameter values:

# For the Michaelis-Menten model (V, K)
#theta = np.array([3, 0.5])  # Case 1: V=3, K=0.5 

# For the Emax model (V, K, h)
theta = np.array([3, 0.2, 2])    # Case 4: V=3, K=0.2, h=2, early saturation (half-max very early)


# Design space upper bound x_max = 20*K to ensure we cover the relevant range of doses
x_max = ldo.get_x_max(theta)  

# Maximum number of iterations for PSO optimization to obtain the design
n_iter_max = 1000



# ============================================================
# Helpers
# ============================================================


" Model helper for choosing correct model mean function based on theta "

def eta(x, theta):
    if len(theta) == 2:
        return ldo.eta_mm(x, theta)
    elif len(theta) == 3:
        return ldo.eta_emax(x, theta)
    else:
        raise ValueError("theta must have length 2 (MM) or 3 (Emax)")
    


"""
Given an approximate design, support points and weights are specified as follows:

    {(x1, w1), (x2, w2), ..., (xk, wk)}
    
where k is the number of support points, xi is the dose level of the i-th support point, 
and wi is the corresponding weight (proportion of total sample size allocated to that 
point). But to generate data , we need an exact design, that is, a list of the actual
x-values repeated a certain number of times according to the weights and total sample size.
"""


" Design helper for converting approximate weights into exact counts "

def w_to_counts(w, n):
    """
    Convert weights to counts for a given total sample size n.
    
    Parameters:
    w (array-like): Weights corresponding to each support point (should sum to 1).
    n (int): Total sample size.
    
    Returns:
    counts (array): Integer counts for each support point, summing to n.
    """
    
    # Copy weights as a new array
    w = np.array(w)

    # Normalize weights to ensure they sum to 1
    w = w / np.sum(w)

    # Initial non-integer counts at each support point
    raw = w * n

    # Round to nearest lower integer
    counts = np.floor(raw).astype(int)

    # Compute the decimal leftovers that were lost by flooring
    remainders = raw - counts

    # Compute how many observations are still needed to reach n
    shortfall = n - np.sum(counts)

    # If flooring already gave a total of n, we are done. Else:
    if shortfall > 0:
        # Get indices of the support points sorted from largest to smallest remainder
        order = np.argsort(-remainders)  # Sort in descending order

        # Distribute the remaining observations to the points with largest remainders
        for i in range(shortfall):
            counts[order[i]] += 1  # Give one more starting from the point with the largest remainder
    
    return counts



" Design helper (wrapper) for converting approximate design to exact design "

def approx_to_exact_design(x_star, w_star, n):
    """
    Convert an approximate design (support points and weights) to an exact design (list of x-values).
    
    Parameters:
    x_star (array-like): Support points of the design.
    w_star (array-like): Weights corresponding to each support point (should sum to 1).
    n (int): Total sample size.
    
    Returns:
    counts (array): Integer counts for each support point, summing to n.
    x_exact (array): List of x-values for the exact design, with repetitions according to weights.
    """

    # Get counts for each support point
    counts = w_to_counts(w_star, n)

    # Create the exact design by repeating each support point according to its count
    x_exact = []

    for x_i, count_i in zip(x_star, counts):
        x_exact += [x_i] * count_i  # Repeat x_i 'count_i' times

    return counts, np.array(x_exact)



" Design helper for getting and compressing approximate PSO design "

def PSO_design(theta, n_iter_max=n_iter_max, pso_seed=1):
    """
    Get the PSO design (support points and weights) for a given theta, and compress it.
    
    Parameters:
    theta (array-like): Parameter values for the model (length 2 for MM, length 3 for Emax).
    n_iter_max (int): Maximum number of iterations for PSO optimization.
    pso_seed (int): Random seed for reproducibility.

    Returns:
    x_star (array): Support points of the PSO design.
    w_star (array): Weights corresponding to each support point.
    """

    # Get PSO design for the given theta
    pso_result = ldo.PSO(theta, n_iter_max=n_iter_max, seed=pso_seed)  


    # Extract support points and weights from cleaned PSO result
    x_star, w_star = compress_design(pso_result['x_star'], pso_result['w_star'])

    return x_star, w_star



"""
The D-optimal design from PSO is compared to the following non-optimal designs with 
equidistant support points and homogeneous weights:

    - Non-optimal Scenario 1:   x = [0, x_max/2, x_max], w = [1/3, 1/3, 1/3]
    - Non-optimal Scenario 2:   x = [0, x_max/4, 3*x_max/4, x_max], w = [1/4, 1/4, 1/4, 1/4]
"""


" Design helper for getting an approximate non-optimal equidistant design "

def nonoptimal_design(x_max, scenario):
    """
    Get a non-optimal equidistant design based on the specified scenario.
    
    Parameters:
    x_max (float): Upper bound of the design space.
    scenario (int): Scenario number (1 or 2) to determine the design structure.
                    scenario = 1:
                        x = [0, x_max/2, x_max], w = [1/3, 1/3, 1/3]
                    scenario = 2:
                        x = [0, x_max/4, 3*x_max/4, x_max], w = [1/4, 1/4, 1/4, 1/4]

    Returns:
    x (array): Support points of the non-optimal design.
    w (array): Weights corresponding to each support point.
    """
    if scenario == 1:
        x = np.array([0, x_max/2, x_max])
        w = np.array([1/3, 1/3, 1/3])
    elif scenario == 2:
        x = np.array([0, x_max/3, 2*x_max/3, x_max])
        w = np.array([1/4, 1/4, 1/4, 1/4])
    else:
        raise ValueError("Scenario must be either 1 or 2.")

    return x, w



# ============================================================
# Data Generation
# ============================================================


"""
Given assumed true parameter θ and variance σ², we can simulate responses from the model 
for a given design (support points and weights):

                yj = η(xi; θ) + εj,   εj ~ N(0, σ²),   
                
                i = 1, ..., k where k is the number of support points, and
                j = 1, ..., n where n is the total sample size, and we 
                have n observations distributed across the k support points according 
                to the weights.

Step by step:

    1. Compute the true mean response at each observation of support points:
                μi = η(xi; θ)     i = 1, ..., k, where k is the number of support points
    
    2. Simulate random normal noise for each observation:
                εj ~ N(0, σ²)   j = 1, ..., n

    3. Generate the observed responses by adding the noise to the mean response:
                yj = μi + εj   j = 1, ..., n

This process is then repeated iteratively to generate multiple datasets for analysis.
"""


def simulate_responses(x_exact, theta, sigma, rng):
    """
    Simulate responses from the model for a given design (support points) and parameters.
    
    Parameters:
    x_exact (array-like): Support points of the design (exact design with repetitions).
    theta (array-like): True parameter values for the model.
    sigma (float): Standard deviation of the noise.
    rng (np.random.Generator): Random number generator.
    Returns:
    y (array): Simulated responses corresponding to each observation in x_exact.
    """

    # Compute the true mean responses using the appropriate model function
    mu = eta(x_exact, theta)

    # Use provided random number generator to simulate normal noise for each observation
    epsilon = rng.normal(loc=0, scale=sigma, size=len(x_exact))

    # Generate the observed responses by adding the noise to the mean response
    y = mu + epsilon

    return y



# ============================================================
# MLE / Non-Linear Least Squares Estimation
# ============================================================


"""
We assume:      yj = η(xi; θ) + εj,   εj ~ N(0, σ²),

with independent errors εj and known variance σ² = 0.5. Under this Gaussian noise 
assumption, estimation of the parameters θ by maximum likelihood is equivalent to
least squares estimation, in this case, non-linear least squares. In particular, we obtain 
the MLE θ_hat by minimizing the negative log-likelihood:

                - log L(θ) = n/2 * log(2πσ²) + 1/(2σ²) * ∑(yj - η(xi; θ))²

which is equivalent to minimizing the sum of squared residuals:

                θ_hat = argmin_θ ∑(yj - η(xi; θ))²

In practice, we can use the `scipy.optimize.minimize` function to perform this 
optimization, providing the sum of squared residuals as the objective function to minimize. 

Let     RSS(θ) = ∑rj² 

denote the residual sum of squares, where each rj = yj - η(xi; θ) corresponds to how much 
the observed response yj deviates from the model's predicted mean response at xi for a 
given θ. 
"""


" Objective function for optimization - the sum of squared residuals (RSS) "

def RSS(theta, x_exact, y):
    """
    Compute the residual sum of squares (RSS) for given parameters, design, and observed responses.

                        RSS(θ) = ∑(yj - η(xi; θ))²
    
    Parameters:
    theta (array-like): Parameter values to evaluate. Must be positive.
    x_exact (array-like): Support points of the design (exact design with repetitions).
    y (array-like): Observed responses corresponding to each observation in x_exact.

    Returns:
    rss (float or np.inf): Residual sum of squares for the given parameters. Returns 
                           infinity if parameters are invalid (non-positive), meaning 
                           “this parameter value is terrible, avoid it”.
    """

    # Store theta as an array
    theta = np.array(theta)

    # Check if theta is valid (positive parameters)
    if len(theta) == 2:  # Michaelis-Menten model
        if theta[0] <= 0 or theta[1] <= 0:
            return np.inf  # Return infinity for invalid parameters
    elif len(theta) == 3:  # Emax model
        if theta[0] <= 0 or theta[1] <= 0 or theta[2] <= 0:
            return np.inf  # Return infinity for invalid parameters
    
    # Compute the predicted mean responses using the appropriate model function
    mu = eta(x_exact, theta)

    # Compute residuals [r1, r2, ..., rn] where rj = yj - η(xi; θ)
    r = y - mu

    # Compute and return the sum of squared residuals
    rss = np.sum(r**2)
    
    return rss



"""
However, the model parameters must have the following constraints for validity:
    - For the Michaelis-Menten model: V > 0, K > 0
    - For the Emax model: V > 0, K > 0, h > 0

Instead of optimizing over θ directly, we can optimize over a transformed parameter space 
where the constraints are automatically satisfied. For example, we can optimize over 
log-transformed parameters. We introduce new parameters φ such that:
    - For the Michaelis-Menten model: φ1 = log(V), φ2 = log(K)
    - For the Emax model: φ1 = log(V), φ2 = log(K), φ3 = log(h)
and then recover the original parameters as:
    - V = exp(φ1), K = exp(φ2), h = exp(φ3)

Because the exponential function always returns positive values, e^φ > 0 for all φ, it 
ensures that V, K, and h are always positive regardless of the values of φ.

We are minimizing the same residual sum of squares, but using a different coordinate 
system, and now expressed in terms of φ instead of θ.
    - Original problem: min_{θ>0} RSS(θ)
    - Transformed problem (input variable): min_{φ} RSS(e^φ)
"""

" Objective function for optimization - RSS with parameter transformation for constraints "

def RSS_transformed(phi, x_exact, y):
    """
    Compute the residual sum of squares (RSS) for given transformed parameters, design, and observed responses.

                        RSS(e^φ) = ∑(yj - η(xi; e^φ))²
    
    Parameters:
    phi (array-like): Transformed parameter values (log-scale).
    x_exact (array-like): Support points of the design (exact design with repetitions).
    y (array-like): Observed responses corresponding to each observation in x_exact.

    Returns:
    rss (float or np.inf): Residual sum of squares for the given transformed parameters. Returns 
                           infinity if transformed parameters are invalid, meaning 
                           “this parameter value is terrible, avoid it”.
    """

    # Store phi as an array
    phi = np.array(phi)

    # Transform back to the original parameter space
    theta = np.exp(phi)

    # Compute the residual sum of squares for the transformed parameters
    rss = RSS(theta, x_exact, y)

    return rss



"""
Workflow for Optimization:
Start from an initial guess theta0 (e.g., the true parameters or some reasonable starting 
point) and transform to the unconstrained space (log-scale): phi0 = log(theta0). Then:

Minimize transformed RSS, with respect to φ, numerically starting from an initial guess phi0. 
The optimization already fullfills positivity constraints (θ > 0).
    1. Start from an initial guess phi0 (e.g., the true parameter values or a reasonable guess).
    2. Try many candidate values of φ 
    3. Evaluate RSS(e^φ) for each candidate
    4. Iteratively move towards the φ that minimizes RSS
    5. Stop when no more meaningful improvement is found.

This is exactly what the `scipy.optimize.minimize` function does when we provide it with 
the objective function (RSS) and an initial guess (phi0). L-BFGS-B is a general purpose 
numerical optimization algorithm. Lastly, we transform back to the original parameter space 
to get the estimated parameters:    theta_hat = exp(phi_hat)

The following function takes data and a starting guess, then asks the numerical optimizer 
to find the positive parameter vector θ that makes the residual sum of squares as small as 
possible. This will provide us with the estimated parameters θ_hat for each simulated 
dataset.
"""


" Optimization function for performing MLE / non-linear least squares estimation "

def MLE(x_exact, y, theta0):
    """
    Perform MLE / NLS estimation to find θ_hat that minimizes RSS.
    
    Parameters:
    x_exact (array-like): Support points of the design (exact design with repetitions).
    y (array-like): Observed responses corresponding to each observation in x_exact.
    theta0 (array-like): Initial guess for the parameters to start the optimization.

    Returns:
    dict: Contains the estimated parameters, value of the negative log-likelihood at the 
          optimum, success flag and optimization details.       
    """

    # Number of parameters
    p = len(theta0)

    # Store the starting guess as an array
    theta0 = np.array(theta0)

    # Optimize in (phi1, phi2, phi3), where V=exp(phi_1), K=exp(phi_2), h=exp(phi_3)

    # Transform the initial guess to the log scale
    phi0 = np.log(theta0)

    # Run the optimizer in the transformed space
    result = minimize(
        fun=RSS_transformed, # Objective function to minimize (with transformation)
        x0=phi0, # Initial guess for the transformed parameters
        args=(x_exact, y), # Additional arguments to pass to RSS_transformed
        method='L-BFGS-B', # Which optimization algorithm to use
        bounds=[(None, None)]*p # No bounds needed in the transformed space 
        )

    # Extract the estimated parameters φ_hat from the optimization result
    phi_hat = result.x

    # Transform back to the original parameter space to get θ_hat
    theta_hat = np.exp(phi_hat)

    # Extract the value of the minimized RSS at θ_hat (same as at φ_hat)
    rss_hat = result.fun

    return {
        'theta_hat': theta_hat, # Estimated parameters,
        'rss_hat': rss_hat, # Value of the residual sum of squares at the found optimum
        'success': result.success, # Whether the optimization was successful
        'message': result.message # Optimization message for diagnostics
    }



# ============================================================
# One-Simulation-Run Function for One Design
# ============================================================


"""
To perform one full experiment for one design (PSO or non-optimal), we can create a 
function that encapsulates the entire process:

    1) Start from one approximate design (support points and weights), obtained from PSO 
       or defined as non-optimal.

    2) Convert it to an exact design (list of x-values) of a given sample size n.

    3) Generate one dataset y by simulating responses from the model using the exact 
       design and true parameters.

    4) Choose a starting guess theta0 for the optimization:     
            theta0 = theta * exp(Z), Z ~ N_p(0, 0.15^2) 
       since starting too close to the true parameters may not reflect a realistic 
       estimation and may make the optimization artificially easy.

    5) Estimate the parameters θ_hat by performing MLE / non-linear least squares 
       estimation using the simulated data.

    6) Return the estimated parameters and any relevant information about the optimization.

The following function implements one such full Monte Carlo repetition for a given design 
and sample size. After repeating this function, say 1000 times, we obtain a distribution 
of estimated parameters θ_hat that we can analyze to compare the performance of the 
PSO design versus the non-optimal design.
"""


" Function to perform one full simulation run for a given design "

def one_simulation_run(theta, x_star, w_star, n, sigma, rng):
    """
    Perform one full simulation run for a given design and sample size.
    
    Parameters:
    theta (array-like): True parameter values for the model.
    x_star (array-like): Support points of the approximate design.
    w_star (array-like): Weights corresponding to each support point of the approximate design.
    n (int): Total sample size for the exact design.
    sigma (float): Standard deviation of the noise for simulating responses.
    rng (np.random.Generator): Random number generator.
    Returns:
    dict: Contains the estimated parameters θ_hat and optimization details.       
    """

    # Step 1: Convert approximate design to exact design
    counts, x_exact = approx_to_exact_design(x_star, w_star, n)

    # Step 2: Simulate responses from the model using the exact design and true parameters
    y = simulate_responses(x_exact, theta, sigma, rng)

    # Step 3: Choose a starting guess theta0 for the estimation
    theta0 = theta * np.exp(rng.normal(loc=0, scale=0.15, size=len(theta)))  # Perturb true parameters by up to ~15% for a reasonable starting guess

    # Step 4: Estimate parameters θ_hat by performing MLE / NLS estimation
    fit = MLE(x_exact, y, theta0)

    return {
        'x_exact': x_exact, # Exact design points used for this simulation
        'counts': counts, # Counts of observations at each support point
        'y': y, # Simulated responses
        'theta_hat': fit['theta_hat'], # Estimated parameters (V_hat, K_hat) or (V_hat, K_hat, h_hat)
        'rss_hat': fit['rss_hat'], # Residual sum of squares at the optimum
        'success': fit['success'], # Whether optimization was successful
        'message': fit['message'] # Optimization message for diagnostics
    }



# ============================================================
# Repeated Simulation Experiment for all Different Designs
# ============================================================


"""
This is where everything is put together. The following function runs the entire 
Monte Carlo study. It will repeat the one_simulation_run function for a specified number 
of simulations, and store the estimated parameters for each run. This will allow us to 
analyze the distribution of θ_hat across simulations for both the PSO design and the 
non-optimal design, and compare their performance in terms of bias, variance, and overall 
estimation quality.

The goal is to compare:
    - PSO optimal design: (x_star, w_star) obtained from PSO optimization
    - Non-optimal design Scenario 1: x = [0, x_max/2, x_max], w = [1/3, 1/3, 1/3]
    - Non-optimal design Scenario 2: x = [0, x_max/4, 3*x_max/4, x_max], w = [1/4, 1/4, 1/4, 1/4]

across many runs.
"""


" Main function to run the repeated simulation experiment for PSO and non-optimal designs "

def run_simulations(theta, n, n_sims, sigma2, n_iter_max, pso_seed=1, base_seed=2):
    """
    Run the repeated simulation experiment for PSO and non-optimal designs.
    
    Parameters:
    theta (array-like): True parameter values for the model.
    n (int): Total sample size for each simulation run.
    n_sims (int): Number of simulation runs to perform.
    sigma2 (float): Variance of the noise for simulating responses.
    n_iter_max (int): Maximum number of iterations for PSO optimization.
    pso_seed (int): Random seed for reproducibility of PSO design.
    base_seed (int): Base random seed for reproducibility of non-optimal designs.
    Returns:
    dict: Contains the estimated parameters across simulations for both designs.       
    """

    " Get Designs "
    
    # Get D-optimal PSO design
    x_star, w_star = PSO_design(theta, n_iter_max=n_iter_max, pso_seed=pso_seed)
    # Get non-optimal designs
    x_max = ldo.get_x_max(theta) # For safety, in case this theta is changed
    x_nonopt1, w_nonopt1 = nonoptimal_design(x_max, scenario=1)
    x_nonopt2, w_nonopt2 = nonoptimal_design(x_max, scenario=2)

    # Dictionary to store the designs for later use
    if len(theta) == 2:
        designs = {
            'D-Optimal': (x_star, w_star),
            'Non-Optimal 1': (x_nonopt1, w_nonopt1),
            'Non-Optimal 2': (x_nonopt2, w_nonopt2)
        }
    else:
        designs = {
            'D-Optimal': (x_star, w_star),
            'Non-Optimal 1': (x_nonopt1, w_nonopt1)
        }

    # Dictionary to store results across simulations for each design
    results = {}

    # Number of parameters to estimate
    p = len(theta)

    " Run simulations for each design "

    # Create a random number generator for reproducibility of the simulations
    rng = np.random.default_rng(base_seed)

    for design_name, (x, w) in designs.items():

        # Matrix to store estimated parameters for each run, n_sims rows and p columns
        theta_hats = np.zeros((n_sims, p))
        # Array to store RSS at the optimum for each run
        rss_hats = np.zeros(n_sims)
        # Array to store optimization success flags for each run
        successes = np.zeros(n_sims, dtype=bool)

        " Run Monte Carlo simulations for this design "

        print(f"\nRunning simulations for design: {design_name} with sample size n={n}...")

        # Do n_sims repetitions
        for s in range(n_sims):

            # Perform one simulation run for this design
            sim_result = one_simulation_run(theta, x, w, n, np.sqrt(sigma2), rng)

            # Store the estimated parameters, RSS at optimum, and success flag
            theta_hats[s, :] = sim_result['theta_hat']
            rss_hats[s] = sim_result['rss_hat']
            successes[s] = sim_result['success']

            # Print progress every 100 completed simulations
            if (s + 1) % 100 == 0 or (s + 1) == n_sims:
                print(f"  Completed {s + 1}/{n_sims} simulations for {design_name}.")

        # Store dictionary of results for this design in the results dictionary
        results[design_name] = {
            'support points': x, # Support points of the design
            'weights': w, # Weights of the design
            'theta_hats': theta_hats, # Estimated parameters across simulations
            'rss_hats': rss_hats, # Minimized RSS across simulations
            'successes': successes # Optimization success flags across simulations
        }

    return results



# ============================================================
# Summaries
# ============================================================

"""
After running the simulations, we can analyze the results to compare the performance of 
the PSO design versus the non-optimal designs. We can compute summary statistics such as:
- Standard Deviation (SD): The spread of the estimated parameter values across simulations.
- Mean: The average of the estimated parameter values across simulations.
- Bias: The difference between the estimated and true parameter values.
- Mean Squared Error (MSE): The average of the squared differences between the estimated 
  and true parameter values.
"""


" Function to compute and print summary statistics "

def summary(results, theta, n, n_sims):

    # Number of parameters
    p = len(theta)

    # Choose parameter names based on the model
    if p == 2:
        # V, K for Michaelis-Menten
        param_names = ['V', 'K']
    elif p == 3:
        # V, K, h for Emax
        param_names = ['V', 'K', 'h']
    else:
        # Fallback in case another model is used
        param_names = [f'theta{i+1}' for i in range(p)]

    # Print a block for the header
    print("\n" + "="*60)
    print(f"Summary of Simulation Results ")
    print(f" - Number of simulations: {n_sims}") 
    print(f" - Sample size n: {n}")
    print(f" - True theta: {theta}")
    print("="*60 + "\n")

    # Loop through each design
    for design_name, result in results.items():

        # Extract the estimated parameters across simulations for this design
        theta_hats = result['theta_hats']
        # Extract the RSS at the optimum across simulations for this design
        rss_hats = result['rss_hats']
        # Compute success rate of optimization across simulations for this design
        success_rate = 100 * np.mean(result['successes']) # True is treated as 1, False as 0

        # Print design information
        print("\n" + "-"*50)
        print(f"Design: {design_name}") # Create a header for this design
        print("-"*50 + "\n")
        print(f"    Support points: x = {np.round(result['support points'], 4)}")
        print(f"    Weights: w = {np.round(result['weights'], 4)}")
        print(f"    Success Rate of Optimization: {success_rate:.2f}%\n")

        # Loop through each parameter to compute and print summary statistics
        for i in range(p):

            # Mean of estimated ith parameter from this design across simulations
            theta_bar = np.mean(theta_hats[:, i])
            # Standard deviation of estimated ith parameter from this design across simulations
            sd_hat = np.std(theta_hats[:, i])
            # Bias of estimated ith parameter from this design across simulations
            bias_hat = theta_bar - theta[i]
            # Mean Squared Error of estimated ith parameter from this design across simulations
            MSE = np.mean((theta_hats[:, i] - theta[i]) ** 2)

            # Print summary statistics for the ith parameter
            print(f"    MLEs across simulations for {param_names[i]}:")
            print(f"        Mean: {theta_bar:.4f}")
            print(f"        SD: {sd_hat:.4f}")
            print(f"        Bias: {bias_hat:.4f}")
            print(f"        MSE: {MSE:.4f}\n")

            # Print mean RSS at the optimum for this design across simulations
        mean_rss = np.mean(rss_hats)
        print(f"     RSS at (optimum) estimate: {mean_rss:.4f}\n")



# ============================================================
# Plotting
# ============================================================


"""
We can also create visualizations to compare the distribution of the estimated parameters
across simulations for the PSO design versus the non-optimal designs. For example, we can 
create boxplots or histograms of the estimated parameters for each design, or scatter plots 
of the estimated parameters against the true parameter values. 
"""


" Function to create boxplots of estimated parameters across designs "

def boxplot_estimates(results, theta):
    """
    Create boxplots of estimated parameters across designs for comparison.
    
    Parameters:
    results (dict): Contains the estimated parameters across simulations for both designs.
    theta (array-like): True parameter values for the model.

    Returns:
    None: Displays boxplots of the estimated parameters for each design.       
    """

    # Number of parameters
    p = len(theta)

    # Choose parameter names based on the model
    if p == 2:
        # V, K for Michaelis-Menten
        param_names = ['V', 'K']
    elif p == 3:
        # V, K, h for Emax
        param_names = ['V', 'K', 'h']
    else:
        # Fallback in case another model is used
        param_names = [f'theta{i+1}' for i in range(p)]

    # Colors for each design
    design_colors = {
        'D-Optimal': 'blue',
        'Non-Optimal 1': 'darkorange',
        'Non-Optimal 2': '#E6A700'   # warm yellow-orange / amber
    }

    # Loop through each parameter to create boxplots
    for i in range(p):

        # Prepare data for boxplot: arrays of estimated ith parameter from each design (ith column of theta_hats)
        estimates_pso = results['D-Optimal']['theta_hats'][:, i]
        estimates_nonopt1 = results['Non-Optimal 1']['theta_hats'][:, i]
        if len(theta) == 2:
            estimates_nonopt2 = results['Non-Optimal 2']['theta_hats'][:, i]

        # Create side-by-side boxplot for the current parameter
        if len(theta) == 2:
            data = [estimates_pso, estimates_nonopt1, estimates_nonopt2]
            labels = ['D-Optimal', 'Non-Optimal 1', 'Non-Optimal 2']
            colors = [design_colors[label] for label in labels]
        elif len(theta) == 3:
            data = [estimates_pso, estimates_nonopt1]
            labels = ['D-Optimal', 'Non-Optimal 1']
            colors = [design_colors[label] for label in labels]

        plt.figure(figsize=(8, 6))

        bp = plt.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True
        )

        # Color each box according to design
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Optional: make lines look a bit nicer
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

        for whisker in bp['whiskers']:
            whisker.set_color('black')

        for cap in bp['caps']:
            cap.set_color('black')

        plt.axhline(y=theta[i], color='red', linestyle='--', label=f'True {param_names[i]}')
        plt.title(f'Boxplot of Estimated {param_names[i]} Across Designs')
        plt.ylabel(f'Estimated {param_names[i]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()



" Function to create histograms of estimated parameters across designs for comparison "

def histogram_estimates(results, theta, bins=30):
    """
    Create histograms of estimated parameters across designs for comparison.
    
    Parameters:
    results (dict): Contains the estimated parameters across simulations for both designs.
    theta (array-like): True parameter values for the model.
    bins (int): Number of bins for the histogram.

    Returns:
    None: Displays histograms of the estimated parameters for each design.       
    """

    # Number of parameters
    p = len(theta)

    # Choose parameter names based on the model
    if p == 2:
        # V, K for Michaelis-Menten
        param_names = ['V', 'K']
    elif p == 3:
        # V, K, h for Emax
        param_names = ['V', 'K', 'h']
    else:
        # Fallback in case another model is used
        param_names = [f'theta{i+1}' for i in range(p)]

    # Names of designs in the results dictionary
    design_names = list(results.keys())

    # Colors for each design
    design_colors = {
        'D-Optimal': 'blue',
        'Non-Optimal 1': 'darkorange',
        'Non-Optimal 2': '#E6A700'   # yellow-orange / amber
    }

    # Loop through each parameter to create histograms
    for i in range(p):

        # Extract estimates of the ith parameter from each design
        estimates_pso = results['D-Optimal']['theta_hats'][:, i]
        estimates_nonopt1 = results['Non-Optimal 1']['theta_hats'][:, i]
        if len(theta) == 2:
            estimates_nonopt2 = results['Non-Optimal 2']['theta_hats'][:, i]

        # Store estimates in a list for easier plotting
        if len(theta) == 2:
            all_estimates = [estimates_pso, estimates_nonopt1, estimates_nonopt2]
        elif len(theta) == 3:
            all_estimates = [estimates_pso, estimates_nonopt1]

        # Ensure all three histograms use the same x-axis limits for better comparison
        x_min = min(np.min(estimates) for estimates in all_estimates) # Smallest estimate across all three designs
        x_max = max(np.max(estimates) for estimates in all_estimates) # Largest estimate across all three designs
        margin = 0.05 * (x_max - x_min) # Add a small margin to the limits
        x_left = x_min - margin # Left limit of x-axis
        x_right = x_max + margin # Right limit of x-axis

        # Create one row with three subplots for the current parameter, one column per design
        if len(theta) == 2:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        elif len(theta) == 3:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        # Loop through each design to create a histogram in the corresponding subplot
        for ax, estimates, design_name in zip(axes, all_estimates, design_names):

            # Calculate the mean of the estimates for this design to display on the plot
            mean_estimate = np.mean(estimates)

            # Draw the histogram of the estimates for this design
            ax.hist(estimates, bins=bins, alpha=0.75, color=design_colors[design_name], edgecolor='black')
            # Add a vertical line for the true parameter value
            ax.axvline(x=theta[i], color='red', linestyle='-', linewidth=2, label=f'True {param_names[i]}')
            # Add a vertical line for the mean of the estimates
            ax.axvline(x=mean_estimate, color='green', linestyle='--', linewidth=2, label=f'Mean Estimate')

            # Set the title this subplot
            ax.set_title(design_name)
            # Label the x-axis
            ax.set_xlabel(f'Estimated {param_names[i]}')
            # Force common x-axis limits for all three histograms
            ax.set_xlim(x_left, x_right)
            # Add a grid for easier visualization
            ax.grid(True, alpha=0.3)

        # Label the y-axis on the first subplot only (since they share y-axis)
        axes[0].set_ylabel('Frequency')

        # Add shared title for the entire figure
        fig.suptitle(f'Histogram of Estimated {param_names[i]} Across Designs', fontsize=16)

        # Show the legend on the last subplot only (since they share the same reference lines)
        axes[-1].legend()

        # Adjust layout to prevent overlap and display the plot
        plt.tight_layout()
        plt.show()



# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    for n in sample_sizes:

        # Run the repeated simulation experiment for PSO and non-optimal designs
        results = run_simulations(theta, n, n_sims, sigma2=sigma2, n_iter_max=n_iter_max, pso_seed=1, base_seed=2)

        # Print summary statistics for the results
        summary(results, theta, n, n_sims)

        # Create boxplots of estimated parameters across designs
        boxplot_estimates(results, theta)

        # Create histograms of estimated parameters across designs
        histogram_estimates(results, theta, bins=30)