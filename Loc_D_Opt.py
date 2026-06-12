import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text


""" 
For the 2-parameter case, the mean-response function of interest is the standard 
Michaelis-Menten function: 

    η(x;θ) = (V * x) / (K + x), 

where θ = (V, K) and x∈[0, x_max] is the design space. 
"""


def eta_mm(x, theta):
    """
    Mean-response function η(x;θ) for Michaelis-Menten model.
    
    Parameters:
    x : array-like, shape (q,)
        Design points at which to evaluate the mean function.
    theta : array-like, shape (2,)
        Parameters θ = (V, K).
    
    Returns:
    eta : array, shape (q,)
        Predicted responses η(x;θ) evaluated at each x_j.
    """
    V, K = theta[0], theta[1]
    eta = (V * x) / (K + x)
    return eta



"""
For the 3-parameter case, we consider the more general Emax model:

    η(x;θ) = V*x^h / (K^h + x^h), 
    
where θ = (V, K, h).

However, numerically it can be unstable when h gets large, because 
x**h or K**h can become extremely large. An equivalent form is:

    η(x;θ) = V / (1 + (K/x)^h)

Now, using the identity: 

    (K/x)^h = exp(h * log(K/x)) = exp(h * (log(K) - log(x)))

we can rewrite the Emax function as:

    η(x;θ) = V / (1 + exp(h * (log(K) - log(x))))

which is a more robust way to compute the Emax function for large h.

OBS. Original formula:      η(0;θ) = V*0^h / (K^h + 0^h) = 0
"""


def eta_emax(x, theta):
    """
    Robust version of the mean-response function η(x;θ) for Emax model.
    
    Parameters:
    x : array-like, shape (q,)
        Design points at which to evaluate the predicted response.
    theta : array-like, shape (3,)
        Parameters θ = (V, K, h).
    
    Returns:
    eta : array, shape (q,)
        Predicted responses η(x;θ) evaluated at each x_j.
    """
    V, K, h = theta[0], theta[1], theta[2]

    # Ensure x is a numpy array for element-wise operations
    x = np.array(x, dtype=float)  

    # Output array to hold predicted responses
    eta = np.zeros(x.shape, dtype=float) # Initialize eta with zeros, shape (q,)

    # Identify which x-values are positive with Boolean array
    positive = x > 0

    # Since log(0) is undefined, we can compute η(x;θ) only for x > 0
    if np.any(positive):
        # Exponent terms for x > 0
        z = h * (np.log(K) - np.log(x[positive]))  # shape (q_positive,)
        # Compute η for x > 0 using the numerically stable form
        eta[positive] = V / (1 + np.exp(z))

    # and let η=0 for x=0 (since the response is 0 when x=0)

    return eta


# Example parameter values for different cases and vizualization of Michelis-Menten curves
# and Emax curves can be found under if __name__ == "__main__": block at the end of this file


""" 
 For Michaelis-Menten:  η(x;θ) = (V * x) / (K + x), where θ = (V, K),

 the gradient vector at a point x is: f(x;θ) = [∂η/∂V, ∂η/∂K] = [x/(K + x), -V*x/(K + x)^2].
 """


def gradient_mm(x, theta):
    """
    Gradient f(x;θ) for Michaelis-Menten model at each x_i.
    
    Parameters:
    x : array-like, x = (x_1, x_2, ..., x_k), shape (k,)
        Design (support) points at which to evaluate the gradient vector.
    theta : array-like, θ = (V, K), shape (2,)
    
    Returns:
    F : array, shape (k, 2)
        Matrix of gradients evaluated at each x_i, where each row 
        corresponds to f(x_i;θ) = [∂η(x_i;θ)/∂V, ∂η(x_i;θ)/∂K].
    """
    V, K = theta[0], theta[1]

    # Derivatives
    d_eta_d_V = x / (K + x)
    d_eta_d_K = -V * x / (K + x)**2

    # Combine into a matrix F where each row is f(x_i;θ)
    F = np.stack((d_eta_d_V, d_eta_d_K), axis=1) # stack vectors as columns, shape (k, 2)

    return F



"""
 For Emax:  η(x;θ) = V*x^h / (K^h + x^h), where θ = (V, K, h),
 the gradient vector at a point x is: 
 f(x;θ) =  [x^h/(K^h + x^h), -V*h*K^(h-1)*x^h/(K^h + x^h)^2, V*x^h*ln(x/K)*K^h/(K^h + x^h)^2].
 """


def gradient_emax(x, theta):
    """
    Gradient f(x;θ) for Emax model at each x_i.
    
    Parameters:
    x : array-like, x = (x_1, x_2, ..., x_k), shape (k,)
        Design (support) points at which to evaluate the gradient vector.
    theta : array-like, θ = (V, K, h), shape (3,)
    
    Returns:
    F : array, shape (k, 3)
        Matrix of gradients evaluated at each x_i, where each row 
        corresponds to f(x_i;θ) = [∂η(x_i;θ)/∂V, ∂η(x_i;θ)/∂K, ∂η(x_i;θ)/∂h].
    """
    V, K, h = theta[0], theta[1], theta[2]

    if K <= 0 or h <= 0:
        raise ValueError("Parameters K and h must be positive for the Emax model.")

    # Derivatives
    d_eta_d_V = x**h / (K**h + x**h)
    d_eta_d_K = -V * h * K**(h-1) * x**h / (K**h + x**h)**2
    # To avoid log(0) when x=0, we can clamp x for the log-term only
    x_safe = np.where(x > 0, x, 1e-12)  # Replace x=0 with a small positive value for log calculation
    d_eta_d_h = V * x**h * np.log(x_safe/K) * K**h / (K**h + x**h)**2

    # Combine into a matrix F where each row is f(x_i;θ)
    F = np.stack((d_eta_d_V, d_eta_d_K, d_eta_d_h), axis=1) # stack vectors as columns, shape (k, 3)

    return F



"""
The information matrix I(ξ;θ) for an approximate design ξ with support points x_i 
and weights w_i is given by:

    I(ξ;θ) = Σ w_i * f(x_i;θ) * f(x_i;θ)^T

Where:

- ξ = {(x_1, w_1), (x_2, w_2), ..., (x_k, w_k)} is the approximate design with 
  support points x_i and corresponding weights w_i.

- f(x_i;θ) is the gradient vector evaluated at x_i, which we can compute using 
  the function F defined above.

To compute M(ξ;θ), we can choose which gradient function to use based on the value of p 
(number of parameters). Then, we compute f(x_i;θ), iterate over each support point x_i,
and accumulate (sum) the weighted outer products w_i * f(x_i;θ) * f(x_i;θ)^T.
"""


def FIM(xi, theta):
    """
    Compute the Fisher Information Matrix I(ξ;θ).
    
    Parameters:
    xi : array-like, shape (2k,)
        The design vector containing support points and weights.
    theta : array-like, shape (p,), where p = 2 for Michaelis-Menten or p = 3 for Emax.
        Parameters θ = (V, K) for Michaelis-Menten or (V, K, h) for Emax.
    
    Returns:
    I + np.eye(p)* 1e-10 : array, shape (p, p)
        The Fisher Information Matrix I(ξ;θ) plus small term for numerical stability.
    """

    p = len(theta)  # number of parameters (2 for V, K)
    k = len(xi) // 2  # number of support points, should be same as len(w)

    " Extract x and w from the input design ξ "

    x = xi[:k] # First k elements are x_i
    w = xi[k:] # Next k elements are w_i

    w = np.array(w, dtype=float)  # Work with w as a copy to avoid modifying the input array

    " Validate weights w_i "
    
    if len(w) != k:
        raise ValueError("Number of weights w_i must match the number of support points x_i.")

    s = np.sum(w)

    if s == 0:
        raise ValueError("Weights must not all be zero (w must sum to a positive value).")
    
    for w_i in w:
        if w_i < 0:
            raise ValueError("Weights must be non-negative.")

    " Create the FIM M(ξ;θ) "

    # Initialize the information matrix
    I = np.zeros((p, p)) # given p = 2 or p = 3, then M is 2x2 or 3x3
    
    # Choose correct gradient and compute it once for all points
    if p == 2:
        F = gradient_mm(x, theta)  # shape (k, 2), F = [f1^T, f2^T, ..., fk^T]^T
    elif p == 3:
        F = gradient_emax(x, theta)  # shape (k, 3), F = [f1^T, f2^T, ..., fk^T]^T
    else:
        raise ValueError("Unsupported number of parameters p. Only p=2 (Michaelis-Menten) and p=3 (Emax) are supported.")
    
    # Accumulate weighted outer products (weighted contributions from each support x_i)
    for i in range(k):
        f_i = F[i, :]  # gradient at x_i, shape (2,) or (3,)
        I += w[i] * np.outer(f_i, f_i)

    # Add small term to ensure I is positive definite for numerical stability while returning I
    return I + np.eye(p)* 1e-10     # Handles both models



"""
Locally D-optimal design problem for Michaelis-Menten model:
The goal is to find the optimal design of the form ξ = {(x_1, x_2, ..., x_k), 
                                                        (w_1, w_2, ..., w_k)},
x_1, x_2, ..., x_k ∈ [0, x_max], w_1, w_2, ..., w_k ∈ [0, 1], and Σ w_i = 1,
that minimizes the negative log determinant of the information matrix I(ξ;θ) for a given θ.
We choose k ≤ P(P + 1)/2 support points, which in our case means k ≤ 3 for p = 2 or 
k ≤ 6 for p = 3. Then, we can choose e.g. k = 3 for a full-rank design, and optimize over 
the 3 support points and their corresponding weights (done inside PSO/GWO wrapper).
"""



"""
Our PSO particle or GWO position is a vector like this:     ξ = [ x_1, x_2, x_3, w_1, w_2, w_3 ]

However the algorithms will likely generate particles that do not satisfy the constraints, e.g.:
    - weights not summing to 1
    - negative weights
    - x_i outside of [0, x_max]

To handle this, we can implement a repair function that takes any particle and projects it 
back into the feasible space of designs. For example:
    - For x_i, we can clip them to [0, x_max].
    - For weights w_i, we can ensure non-negativity, and then normalize them to sum to 1.
"""


def repair_design(design, k, x_max):
    """
    Repair a PSO particle to ensure it represents a valid design ξ.
    
    Parameters:
    design : array-like, shape (2k,)
        The design vector ξ = [x_1, ..., x_k, w_1, ..., w_k].
    k : int
        The number of support points in the design.
    x_max : float
        The maximum value for the support points x_i.
    
    Returns:
    xi : array, shape (2k,)
        The repaired PSO particle that satisfies the constraints of a valid design.

        Design points are clipped into the design region [0, x_max].
        Weights are repaired by setting: w = w**2 / Σ w**2 
         - Squaring makes every weight non-negative
         - Dividing by the sum of squares normalizes the weights to sum to 1
    """
    # Split the proposed design into x and w components
    x_raw = np.array(design[:k])  # First k elements are x_i
    w_raw = np.array(design[k:])   # Next k elements are w_i

    # Repair x_i by clipping to [0, x_max]
    x = np.clip(x_raw, 0, x_max)

    # Square weights to make them non-negative
    w2 = w_raw**2  

    # Calculate the sum of squared weights for normalization
    s = np.sum(w2)

    # Fallback if all weights are 0
    if s == 0:
        # If all weights are zero, assign equal weights
        w = np.ones(k) / k

    # Else normalize weights to sum to 1
    else:
        w = w2 / s

    # Combine x and w back into a single repaired design vector ξ
    xi = np.concatenate((x, w))
    
    return xi



"""
The PSO procedure will involve locally optimizing the design ξ for each particle in the 
swarm. To do this, we define an objective function that takes a design ξ, computes the 
information matrix I(ξ;θ), and returns the negative log determinant of I(ξ;θ), which is 
the quantity we want to minimize for D-optimality: 

    Ψ(ξ) = -log(det(I(ξ;θ)))

Moreover, the design space for x is [0, x_max], and we can set 

    x_max = 19 * K 

to ensure we cover a wide range of substrate concentrations. When x = 19*K, we have 

    η(19*K;θ)/V = 19*K / (K + 19*K) = 19/20 ≈ 0.95, 

which is close to the maximum response V (95%). So setting x_max = 19*K ensures we explore 
the region where the response approaches its maximum.

For the Emax model, we can similarly set x_max = (19*K)^(1/h) to ensure we cover the relevant 
design space, since the Emax function also approaches its maximum as x becomes large relative to K.
"""


" Helper function to determine x_max based on the model and given saturation level (e.g. 95%) "

def get_x_max(theta, q=0.95):
    K = theta[1]

    if len(theta) == 2:
        return (q / (1 - q)) * K  # 19*K for q=0.95

    elif len(theta) == 3:
        h = theta[2]
        return K * (q / (1 - q))**(1 / h)  # (19*K)^(1/h) for q=0.95

    else:
        raise ValueError("theta must have length 2 or 3")


def objective_function(design, theta):
    """Objective function for D-optimal design, which computes -log(det(I(ξ;θ))). 

    Parameters:
    design : array-like, shape (2k,)
        The design vector ξ = [x_1, ..., x_k, w_1, ..., w_k].
    theta : tuple
        The parameter values (V, K) for the Michaelis-Menten model.

    Returns:
    float
        The value of -log(det(I(ξ;θ))) for the given design and parameters.
    """
    # Determine x_max
    x_max = get_x_max(theta)  # Set x_max based on K to ensure we cover the relevant design space

    # Number of parameters
    p = len(theta)  # V, K or V, K, h

    # Number of support points / weights in the design
    k = p*(p + 1) // 2  # k = 3 for p = 2, k = 6 for p = 3

    # Repair the proposed design to ensure it represents a valid design ξ
    xi = repair_design(design, k, x_max)

    # Compute the information matrix I(ξ;θ) for the repaired design ξ and parameter θ
    I = FIM(xi, theta)

    # Calculate the determinant of the information matrix I(ξ;θ)
    det_I = np.linalg.det(I)

    # Return the negative log of the determinant for minimization by PSO
    if det_I > 0:
        return -np.log(det_I)
    else:
        return 1e6  # Return a large penalty value if determinant is non-positive
    
"""
Returning 1e6 is a penalty value so PSO treats that particle as very bad and moves away 
from it. This keeps the objective finite (instead of nan/-inf) and helps optimization 
remain stable. So the else: return 1e6 is mainly a safeguard: 
    “reject infeasible or degenerate designs.”
"""


""" Swarm evaluable objective function:
     wraps the objective function and applies it to each particle in the swarm """

def swarm_objective_function(designs, theta):
    """
    Wrapper function to evaluate the local D-optimal objective for each 
    design in the swarm.
    
    Parameters:
    designs : array-like, shape (n_designs, 2k)
        The swarm of designs, where each design is a design vector 
        ξ= [x_1, ..., x_k, w_1, ..., w_k].
    theta : tuple
        The parameter values (V, K) for the Michaelis-Menten model.
    
    Returns:
    objectives : array, shape (n_designs,)
        The objective function values for each design in the swarm.
    """
    # Number of designs in the swarm
    n_designs = designs.shape[0]  

    # Initialize an array to hold the objective values for each design
    objectives = np.zeros(n_designs)

    # Evaluate the objective function for each design in the swarm and store it
    for i in range(n_designs):
        # Objective function value for ith design
        objectives[i] = objective_function(designs[i], theta) 

    return objectives



"""
PSO will minimize the swarm_objective_function, which evaluates the local D-optimal 
objective for each particle in the swarm. The PSO algorithm will iteratively update 
the particles' positions in the design space to find the design ξ that minimizes 
-log(det(I(ξ;θ))), which corresponds to maximizing the determinant of the information 
matrix and thus finding a locally D-optimal design for the given parameter values θ.
"""

"Hyperparameters"

# Classic parameter values for PSO
w = 0.7     # inertia weight 
c1 = 2.0    # cognitive coefficient
c2 = 2.0    # social coefficient

# Maximum number of iterations
n_iter_max = 1000

# Seed for reproducibility of PSO runs
seed = 1

# Number of particles in the swarm
n_particles = 30  # Number of particles in the swarm

# Threshold for convergence criterion: |Gbest_score - current best score in swarm| ≤ conv_threshold
conv_tol = 1e-6

"Particle Swarm Optimization (PSO) implementation"

def PSO(theta, w=w, c1=c1, c2=c2, n_particles=n_particles, n_iter_max=n_iter_max, conv_tol=conv_tol, seed=seed):
    """
    Find the locally D-optimal design ξ* for the Michaelis-Menten/Emax model.
    """

    " Define arguments "
    
    # Random number generator
    rng = np.random.default_rng(seed)

    # Determine x_max based on K to ensure we cover the relevant design space
    x_max = get_x_max(theta)

    # Number of parameters
    p = len(theta)  # V, K or V, K, h

    # Number of support points / weights in the design
    k = p*(p + 1) // 2  # k = 3 for p = 2, k = 6 for p = 3

    # Dimension of each particle in the swarm
    d_particle = 2 * k  # Each particle represents a design ξ with k support points and k weights

    " Bounds for support points and raw weights in the particle representation "

    # Lower bounds:
    #   First k entries: x_i >= 0
    #   Last k entries: raw weights >= -1 (will be squared and normalized)
    lb = np.concatenate((np.zeros(k), -np.ones(k)))  # shape (2k,)

    # Upper bounds:
    #   First k entries: x_i <= x_max
    #   Last k entries: raw weights <= 1 
    ub = np.concatenate((x_max * np.ones(k), np.ones(k)))  # shape (2k,)

    " Initialize particle positions and velocities randomly within the bounds "

    # Create n 2k-dim. particles within bounds for both x_i and raw weights
    Positions = rng.uniform(lb, ub, size=(n_particles, d_particle))  # shape (n_particles, 2k)
    # Initialize Velocities[i] = [v_i1, ..., v_i(2k)], each with random step size in [-1, 1]
    Velocities = rng.uniform(-1, 1, size=(n_particles, d_particle))  # shape (n_particles, 2k)

    " Initialize best positions and scores (objective function values) for each particle "

    # Personal best positions (initially set to the initial positions)
    Pbest = Positions.copy()  # shape (n_particles, 2k)
    # Personal best scores (initially set to the objective function values at initial positions)
    Pbest_scores = swarm_objective_function(Pbest, theta)  # shape (n_particles,)
    # How good each Pbest position is, lower = better (since we minimize negative log of determinant)

    # Global best position (initially set to the best of the initial positions)
    Gbest_index = np.argmin(Pbest_scores)  # Index of the best personal best score
    Gbest = Pbest[Gbest_index].copy()  # Global best position, shape (2k,)
    # Global best score (the best objective function value found so far)
    Gbest_score = Pbest_scores[Gbest_index]  # scalar

    # List to store global best objective values over iterations
    Gbest_scores_history = [Gbest_score]

    # Variable to store difference Gbest_score and current best score over the swarm for convergence monitoring
    diff = np.inf  # Initialize to infinity for the first iteration

    # Variable to track the number of iterations 
    n_iter = 0

    " PSO main loop: iteratively update velocities and particles to find optimal design "

    # While max nr. of iterations not reached and not converged (|diff| >= threshold)
    while ( abs(diff) >= conv_tol ) and ( n_iter < n_iter_max ):

        # Increment iteration counter
        n_iter += 1  

        # Coefficients for random exploration
        r1 = rng.random((n_particles, d_particle))  # [r11, ..., r1(2k)] for updating each particle's coordinates
        r2 = rng.random((n_particles, d_particle))  # [r21, ..., r2(2k)] for updating each particle's coordinates

        # Update velocities: V = inertia + cognitive + social
        Velocities = (w * Velocities 
                      + c1 * r1 * (Pbest - Positions) 
                      + c2 * r2 * (Gbest - Positions))
        
        # Move particles: X = X + V
        Positions = Positions + Velocities

        # Keep particles within bounds
        Positions = np.clip(Positions, lb, ub)

        # Evaluate the objective function for the new positions of all particles
        objectives = swarm_objective_function(Positions, theta)  # shape (n_particles,)

        # Current best score in the swarm after moving particles (for convergence monitoring)
        Cbest_score = np.min(objectives)  # scalar

        # Calculate and update the difference (for convergence monitoring)
        diff = Gbest_score - Cbest_score
        # positive if Gbest is better than current best in swarm, negative if current best in swarm is better than Gbest

        # Compare new values with current personal bests
        obj_better = objectives < Pbest_scores  # Boolean array: True where new position/objective is better

        # Update personal best values and positions with Boolean indexing
        Pbest_scores[obj_better] = objectives[obj_better]  # Update Pbest scores where obj_better = True
        Pbest[obj_better] = Positions[obj_better]  # Update Pbest positions where obj_better = True

        # Update global best if any personal best is better than current global best
        Gbest_index = np.argmin(Pbest_scores)  # Index of potential new global best
        
        # If it is better, update global best position and value
        if Pbest_scores[Gbest_index] < Gbest_score:
            Gbest_score = Pbest_scores[Gbest_index]  # Update global best score
            Gbest = Pbest[Gbest_index].copy()  # Update global best position

        # Store global best score for this iteration
        Gbest_scores_history.append(Gbest_score)
        

    # Convert the final global best particle into a valid design ξ* 
    xi_star = repair_design(Gbest, k, x_max)  # shape (2k,)
    x_star = xi_star[:k]  # Extract support points from the design vector
    w_star = xi_star[k:]  # Extract weights from the design vector

    # Compute final det(I(ξ*;θ)) and -log[det(I(ξ*;θ))] for the best design found
    I_star = FIM(xi_star, theta)  # Information matrix for the best design
    detI_star = np.linalg.det(I_star)  # Determinant of the information
    neg_log_detI_star = -np.log(detI_star)  # Objective function value for the best design

    # Return the results as a dictionary for easy access
    return {
        'theta': theta,
        'x_max': x_max,
        'xi_star': xi_star,
        'x_star': x_star,
        'w_star': w_star,
        'detI_star': detI_star,
        'neg_log_detI_star': neg_log_detI_star,
        'Gbest_scores_history': Gbest_scores_history,
        'n_stop': n_iter,
        'stop_diff': abs(diff)
    }



""""
The next metaheuristic to be implemented is the Grey Wolf Optimizer (GWO), which is another 
population-based optimization algorithm inspired by the social hierarchy and hunting 
behavior of grey wolves. The GWO algorithm will also be applied to find locally D-optimal 
designs for the Michaelis-Menten and Emax models.

Similar to PSO, we will initialize a population of candidate designs (wolves), evaluate their 
fitness using the same objective function (negative log determinant of the information 
matrix), and iteratively update the positions of the wolves based on the positions of the 
best wolves in the hierarchy (alpha, beta, delta) to converge towards the optimal design. 
"""

" Grey Wolf Optimizer (GWO) implementation "

def GWO(theta, n_wolves=n_particles, n_iter_max=n_iter_max, conv_tol=conv_tol, seed=seed):
    """
    Find the locally D-optimal design ξ* for the Michaelis-Menten/Emax model using GWO.
    
    Parameters:
    theta : array-like, shape (p,)
        Parameters θ = (V, K) for Michaelis-Menten or (V, K, h) for Emax.
    n_wolves : int, optional
        Number of wolves (candidate solutions) in the population. Default is n_particles.
    n_iter_max : int, optional
        Maximum number of iterations for the GWO algorithm. Default is n_iter_max.
    conv_tol : float, optional
        Convergence tolerance for the GWO algorithm. Default is conv_tol.
    seed : int, optional
        Random seed for reproducibility. Default is seed.

    Returns:
    dict
        A dictionary containing the results of the optimization, in particular 
        the best design found.
    """

    " Define arguments "
    
    # Random number generator
    rng = np.random.default_rng(seed)

    # Determine x_max based on K to ensure we cover the relevant design space
    x_max = get_x_max(theta) 

    # Number of parameters
    p = len(theta)  # V, K or V, K, h

    # Number of support points / weights in the design
    k = p*(p + 1) // 2  # k = 3 for p = 2, k = 6 for p = 3

    # Dimension of each particle in the swarm
    d_particle = 2 * k  # Each particle represents a design ξ with k support points and k weights

    " Bounds for support points and raw weights in the particle representation "

    # Lower bounds:
    #   First k entries: x_i >= 0
    #   Last k entries: raw weights >= -1 (will be squared and normalized)
    lb = np.concatenate((np.zeros(k), -np.ones(k)))  # shape (2k,)

    # Upper bounds:
    #   First k entries: x_i <= x_max
    #   Last k entries: raw weights <= 1 
    ub = np.concatenate((x_max * np.ones(k), np.ones(k)))  # shape (2k,) 

    " Initialize positions, evaluate fitness, and identify alpha, beta, delta wolves "  

    # Initialize wolf positions randomly and uniformly within bounds for both x_i and raw weights
    Positions = rng.uniform(lb, ub, size=(n_wolves, d_particle))  # shape (n_wolves, 2k)

    # Evaluate initial population fitness using the objective function (negative log determinant of M)
    scores = swarm_objective_function(Positions, theta)  # shape (n_wolves,), Lower = better

    # Sort wolves by fitness to identify alpha, beta, delta
    sorted_indices = np.argsort(scores)  # Indices that would sort the scores

    # Identify and store the best wolf (alpha)
    alpha_pos = Positions[sorted_indices[0]].copy()  # Position of the best wolf (alpha)
    alpha_score = scores[sorted_indices[0]]  # Score of the best wolf (alpha), lowest score

    # Identify and store the second best wolf (beta)
    beta_pos = Positions[sorted_indices[1]].copy()  # Position of the second best wolf (beta)
    beta_score = scores[sorted_indices[1]]  # Score of the second best wolf (beta), second lowest score

    # Identify and store the third best wolf (delta)
    delta_pos = Positions[sorted_indices[2]].copy()  # Position of the third best wolf (delta)
    delta_score = scores[sorted_indices[2]]  # Score of the third best wolf (delta), third lowest score

    # Store alpha scores history for convergence plot
    alpha_scores_history = [alpha_score]

    # Variable to track the difference between old alpha score and new best score for convergence monitoring
    diff = np.inf  # Initialize to infinity for the first iteration

    # Variable to track the number of iterations
    n_iter = 0

    " Main GWO loop "

    # While max nr. of iterations not reached and not converged (|diff| >= threshold)
    while ( abs(diff) >= conv_tol ) and ( n_iter < n_iter_max ):
        
        # Increment iteration counter
        n_iter += 1

        # Store the old alpha score for convergence monitoring
        old_alpha_score = alpha_score

        # linearly decrease a from 2 to 0 over iterations
        a = 2 - n_iter * (2 / n_iter_max) 

        # Array to store new positions of wolves after update, alt. np.zeros_like(Positions)
        New_Positions = np.zeros((n_wolves, d_particle))  

        " Update each wolf's position based on the positions of alpha, beta, and delta wolves "

        for i in range(n_wolves):

            # Current wolf
            X = Positions[i].copy()  # shape (2k,)

            # Random coefficient vectors for movement towards alpha
            r1_alpha = rng.random(d_particle)  # shape (2k,)
            r2_alpha = rng.random(d_particle)  # shape (2k,)
            A1 = 2 * a * r1_alpha - a  # shape (2k,)
            C1 = 2 * r2_alpha  # shape (2k,)

            # Random coefficient vectors for movement towards beta
            r1_beta = rng.random(d_particle)  # shape (2k,)
            r2_beta = rng.random(d_particle)  # shape (2k,)
            A2 = 2 * a * r1_beta - a  # shape (2k,)
            C2 = 2 * r2_beta  # shape (2k,)

            # Random coefficient vectors for movement towards delta
            r1_delta = rng.random(d_particle)  # shape (2k,)   
            r2_delta = rng.random(d_particle)  # shape (2k,)
            A3 = 2 * a * r1_delta - a  # shape (2k,)
            C3 = 2 * r2_delta  # shape (2k,)

            # Compute distance to alpha, beta, delta wolves
            D_alpha = np.abs(C1 * alpha_pos - X)  # shape (2k,)
            D_beta = np.abs(C2 * beta_pos - X)  # shape (2k,)
            D_delta = np.abs(C3 * delta_pos - X)  # shape (2k,)

            # Compute candidate positions from alpha, beta, delta guidance
            X1 = alpha_pos - A1 * D_alpha  # shape (2k,)
            X2 = beta_pos - A2 * D_beta  # shape (2k,)
            X3 = delta_pos - A3 * D_delta  # shape (2k,)

            # Wolf's new position is the average of the three candidates
            X_new = (X1 + X2 + X3) / 3  # shape (2k,)

            # Optional but clean: Ensure new position is within bounds (for evaluation stability)
            X_new = np.clip(X_new, lb, ub)

            New_Positions[i] = X_new

        # Update the whole population of wolves to their new positions
        Positions = New_Positions

        " Re-evaluate population fitness and update alpha, beta, delta wolves "

        # New fitness scores for the updated positions of all wolves
        scores = swarm_objective_function(Positions, theta)  # shape (n_wolves,)
        # Indices that would sort the scores
        sorted_indices = np.argsort(scores)  

        # Update alpha position
        alpha_pos = Positions[sorted_indices[0]].copy()  
        # Update alpha score
        alpha_score = scores[sorted_indices[0]]  

        # Update beta position
        beta_pos = Positions[sorted_indices[1]].copy()
        # Update beta score
        beta_score = scores[sorted_indices[1]]

        # Update delta position
        delta_pos = Positions[sorted_indices[2]].copy()
        # Update delta score
        delta_score = scores[sorted_indices[2]]

        # Calculate and update the difference for convergence monitoring
        diff = old_alpha_score - alpha_score

        # Store alpha score for this iteration for convergence plot
        alpha_scores_history.append(alpha_score)

    " Convert the final alpha wolf's position into a valid design ξ* and compute final metrics "

    # Repair the alpha wolf's position to ensure it represents a valid design ξ*
    xi_star = repair_design(alpha_pos, k, x_max)  # shape (2k,)
    # Extract support points and weights from the design vector
    x_star = xi_star[:k]  # shape (k,)
    w_star = xi_star[k:]  # shape (k,)

    # Compute the information matrix I(ξ*;θ) for the best design found
    I_star = FIM(xi_star, theta)  # shape (p, p)
    # Compute the determinant of the information matrix I(ξ*;θ)
    detI_star = np.linalg.det(I_star)  # scalar
    # Compute the negative log of the determinant for the best design found
    neg_log_detI_star = -np.log(detI_star)  # scalar

    # Return the results as a dictionary for easy access
    return {
        'theta': theta,
        'x_max': x_max,
        'xi_star': xi_star,
        'x_star': x_star,
        'w_star': w_star,
        'detI_star': detI_star,
        'neg_log_detI_star': neg_log_detI_star,
        'alpha_scores_history': alpha_scores_history,
        'alpha_score': alpha_score,
        'beta_score': beta_score,
        'delta_score': delta_score,
        'n_stop': n_iter,
        'stop_diff': abs(diff)
    }



""" The following function cleans the PSO/GWO design (drop near-zero weights, merge duplicate support points, 
       renormalize weights)."""

" Implemantation of helper for cleaning the PSO design "

def compress_design(x_star, w_star, x_tol=1.5e-3, w_tol=1.5e-3):
    """ Clean the PSO/GWO design by merging duplicate support points and removing near-zero weights. 
    
    Parameters:
    x_star (array-like): Support points of the design.
    w_star (array-like): Corresponding weights of the support points.
    x_tol (float): Tolerance for merging support points (default: 1.5e-3).
    w_tol (float): Tolerance for removing near-zero weights (default: 1.5e-3).
    
     Returns:
     tuple: A tuple containing the cleaned support points and their corresponding weights.
     """
    # Convert to numpy arrays for easier manipulation
    x_star = np.array(x_star, dtype=float).copy()
    w_star = np.array(w_star, dtype=float).copy()

    # Remove near-zero weights
    large_w = w_star > w_tol
    x_star, w_star = x_star[large_w], w_star[large_w]

    if len(x_star) == 0:
        raise ValueError("All weights are ~0 after filtering. No valid design points remain.")
    
    # Sort by support points (helps with merging duplicates)
    sort_idx = np.argsort(x_star)   # Sort indices based on support points
    x_star, w_star = x_star[sort_idx], w_star[sort_idx] # Sort support points and weights accordingly

    # Merge x-values that are "the same" numerically
    
    unique_x = [x_star[0]]
    unique_w = [w_star[0]]

    for x_i, w_i in zip(x_star[1:], w_star[1:]):
        if np.abs(x_i - unique_x[-1]) < x_tol:  # If x_i is close to the last unique x
            unique_w[-1] += w_i  # Merge weights
        else:
            unique_x.append(x_i)  # Add new unique x
            unique_w.append(w_i)  # Add corresponding weight

    # Convert back to numpy arrays
    unique_x = np.array(unique_x)
    unique_w = np.array(unique_w)

    # Renormalize weights to sum to 1
    unique_w = unique_w / np.sum(unique_w)

    return unique_x, unique_w



""" Optional: Implementation of function that plots the curve η(x;θ) of the model for a 
given θ. Useful for visualization of the mean respons function together with the optimal 
design. """

def plot_curve_and_design(theta, x_star, w_star,x_max, eta_func, theta_case, title):
    """ Plot the curve of η(x;θ) along with the support points of the design. """
    x_plot = np.linspace(0, x_max, 500)
    eta_values = eta_func(x_plot, theta)

    # Evaluate η(x;θ) at the support points
    eta_at_support = eta_func(x_star, theta)

    # Plot η(x;θ) curve
    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, eta_values, label=r"$\eta(x; \theta)$", color = f"C{theta_case-1}")

    # Bars proportional to weights (scaled for visibility) at the support points
    bar_heights = w_star * np.max(eta_values) * 0.4  # Scale weights to be visible on the plot
    bar_width = 0.03 * x_max  # Width of bars proportional to x_max

    # Plot vertical bars on x-axis
    plt.bar(x_star, bar_heights,
            width=bar_width,    # Bars placed at support points
            bottom=0,   # Bars start from y=0, so they appear like probability masses on the x-axis
            color='black',
            alpha=0.4,
            label='Optimal Support Points (height ∝ weight)')

    # Mark the optimal support points with dots at their corresponding η(x;θ) values
    plt.scatter(x_star, eta_at_support,
        color='red', s=30, zorder=5, label='η(x;θ) at Optimal Support Points')
    
    texts = []
    # Label bars
    for i in range(len(x_star)):
        texts.append(plt.text(x_star[i], 
                              bar_heights[i] + 0.02*max(eta_values),   # Labels appear slightly above the bars
                              f'w{i+1}={w_star[i]:.2f}',
                              ha='center',
                              fontsize=10))
    adjust_text(texts)
        
    plt.xlim(0, x_max)
    if len(theta) == 2:
        plt.title(title if title else f'Curve η(x;θ) of Michaelis-Menten Model and Optimal Support Points' 
                  f' for Case {theta_case}: θ = (V={theta[0]}, K={theta[1]}) \n \n')
    elif len(theta) == 3:
        plt.title(title if title else f'Curve η(x;θ) of Emax Model and Optimal Support Points' 
                  f' for Case {theta_case}: θ = (V={theta[0]}, K={theta[1]}, h={theta[2]}) \n \n')
    plt.xlabel('x')
    plt.ylabel(r'$\eta(x; \theta)$')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()   




if __name__ == "__main__":

    # -------------------------------------------
    # Michaelis-Menten model: visualization and results
    # -------------------------------------------

    " Example parameter values for different cases (Michaelis-Menten model) "
    
    # Assumed parameter values for different cases
    theta_case1_mm = np.array([3, 0.5])  # Case 1: V=3, K=0.5
    theta_case2_mm = np.array([7, 2]) # Case 2: V=7, K=2
    theta_case3_mm = np.array([5, 8]) # Case 3: V=5, K=8
    theta_case4_mm = np.array([5, 4]) # Case 4: V=5, K=4

    " Visualize the Michaelis-Menten mean function η(x;θ) for different parameter values θ "

    # x range
    x_plot = np.linspace(0, 100, 500)

    # Parameter cases to visualize
    theta_cases_mm = [theta_case1_mm, theta_case2_mm, theta_case3_mm, theta_case4_mm]

    # Plot curves for each case
    plt.figure(figsize=(8, 6)) # Create a larger figure for better visibility
    for i, theta in enumerate(theta_cases_mm): # i = 0, 1, 2, 3 for the four cases, theta = (V, K) for each case
        # Compute η(x;θ) for the current case across the x range
        eta_values = eta_mm(x_plot, theta) 
        # Plot the curve for the current case, labeling it with the parameter values
        plt.plot(x_plot, eta_values, label=f'Case {i+1}: V={theta[0]}, K={theta[1]}')

    # Finalize the plot with title, labels, legend, and grid
    plt.title('Mean Function η(x;θ) for Different Parameters (Michaelis-Menten Model)')
    plt.xlabel('x')
    plt.ylabel('η(x;θ)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------------------------------------------
    # PSO on Michaelis-Menten model
    # -------------------------------------------

    " Example usage of PSO to find locally D-optimal design for each fixed θ (Michaelis-Menten model) "

    results_mm_pso = []

    print(" \n--- PSO Results for Michaelis-Menten Model ---\n")
    for theta in theta_cases_mm:
        result_mm = PSO(theta, w, c1, c2, n_particles, n_iter_max, conv_tol, seed)
        results_mm_pso.append(result_mm)
        print(f"\nResults for θ = {theta}")
        print(f"\n  Determinant of Fisher Information Matrix I(ξ*;θ):")
        print(f"      det(I(ξ*;θ)) = {result_mm['detI_star']}")
        print(f"\n  Negative log determinant of I(ξ*;θ):")
        print(f"      -log(det(I(ξ*;θ))) = {result_mm['neg_log_detI_star']}")
        print(f"\n  Number of iterations until convergence: ")
        print(f"      n_stop = {result_mm['n_stop']}")
        print(f"\n  Convergence difference: ")
        print(f"      |diff| = {result_mm['stop_diff']}\n")

        # Cleaned support points and weights and print cleaned design
        x_star_clean, w_star_clean = compress_design(result_mm['x_star'], result_mm['w_star'])
        print(f"  Cleaned support points x* (merged duplicates, removed near-zero weights and renormalized):")
        print(f"      x* = {x_star_clean}")
        print(f"\n  Cleaned weights w* (corresponding to cleaned support points):")
        print(f"      w* = {w_star_clean}\n")


    " Plot Michaelis-Menten curves and mark optimal support points for cleaned PSO results "

    # Clean the PSO design
    for res in results_mm_pso:
        res['x_star'], res['w_star'] = compress_design(res['x_star'], res['w_star'])

    # Counter for cases to label them in the plot
    theta_case = 1  

    # Plot curves and optimal support points for each case
    for res in results_mm_pso:
        x_max = res['x_max']
        plot_curve_and_design(res['theta'],
                               res['x_star'], 
                               res['w_star'], 
                               x_max, 
                               eta_mm, 
                               theta_case, 
                               title=None)
        theta_case += 1


    # -------------------------------------------
    # GWO on Michaelis-Menten model
    # -------------------------------------------

    " Example usage of GWO to find locally D-optimal design for each fixed θ (Michaelis-Menten model) "

    results_mm_gwo = []

    print("\n--- GWO Results for Michaelis-Menten Model ---\n")
    for theta in theta_cases_mm:
        result_mm = GWO(theta, 
                        n_wolves=n_particles, 
                        n_iter_max=n_iter_max, 
                        conv_tol=conv_tol, 
                        seed=seed)
        results_mm_gwo.append(result_mm)
        print(f"\nResults for θ = {theta}")
        print(f"\n  Determinant of Fisher Information Matrix I(ξ*;θ):")
        print(f"      det(I(ξ*;θ)) = {result_mm['detI_star']}")
        print(f"\n  Negative log determinant of I(ξ*;θ):")
        print(f"      -log(det(I(ξ*;θ))) = {result_mm['neg_log_detI_star']}")
        print(f"\n  Number of iterations until convergence: ")
        print(f"      n_stop = {result_mm['n_stop']}")
        print(f"\n  Convergence difference: ")
        print(f"      |diff| = {result_mm['stop_diff']}\n")

        # Clean support points and weights and print cleaned design
        x_star_clean, w_star_clean = compress_design(result_mm['x_star'], result_mm['w_star'])
        print(f"  Cleaned support points x* (merged duplicates, removed near-zero weights and renormalized):")
        print(f"      x* = {x_star_clean}")
        print(f"\n  Cleaned weights w* (corresponding to cleaned support points):")
        print(f"      w* = {w_star_clean}\n")


    " Plot Michaelis-Menten curves and mark optimal support points for GWO results "

    # Clean the GWO design
    for res in results_mm_gwo:
        res['x_star'], res['w_star'] = compress_design(res['x_star'], res['w_star'])

    # Counter for cases to label them in the plot
    theta_case = 1

    # Plot curves and optimal support points for each case
    for res in results_mm_gwo:
        x_max = res['x_max']
        plot_curve_and_design(res['theta'],
                               res['x_star'], 
                               res['w_star'], 
                               x_max, 
                               eta_mm, 
                               theta_case, 
                               title=None)
        theta_case += 1


    # -------------------------------------------
    # Emax model: visualization and results
    # -------------------------------------------

    " Example parameter values for different cases (Emax model) "

    # Assumed parameter values for different cases
    theta_case1_emax = np.array([1, 1, 0.5])  # Case 1: V=1, K=1, h=0.5, very gradual response 
    theta_case2_emax = np.array([2.5, 2, 6])    # Case 2: V=3, K=1, h=6, very steep response
    theta_case3_emax = np.array([2, 6, 2])    # Case 3: V=2, K=6, h=2, late saturation (needs large x to approach V)
    theta_case4_emax = np.array([3, 0.2, 2])    # Case 4: V=3, K=0.2, h=2, early saturation (half-max very early)

    " Visualize the Emax mean function η(x;θ) for different parameter values θ "

    # x range
    x_plot_emax = np.linspace(0, 100, 500)

    # Parameter cases to visualize
    theta_cases_emax = [theta_case1_emax, theta_case2_emax, theta_case3_emax, theta_case4_emax]

    # Plot curves for each case
    plt.figure(figsize=(8, 6)) # Create a larger figure for better visibility
    for i, theta in enumerate(theta_cases_emax): # i = 0, 1, 2, 3 for the four cases, theta = (V, K, h) for each case
        # Compute η(x;θ) for the current case across the x range
        eta_values = eta_emax(x_plot_emax, theta) 
        # Plot the curve for the current case, labeling it with the parameter values
        plt.plot(x_plot_emax, eta_values, label=f'Case {i+1}: V={theta[0]}, K={theta[1]}, h={theta[2]}')

    plt.title('Mean Function η(x;θ) for Different Parameter Values (Emax Model)')
    plt.xlabel('x')
    plt.ylabel('η(x;θ)')
    plt.legend()
    plt.grid(True)
    plt.show()


    # -------------------------------------------
    # PSO on Emax model
    # -------------------------------------------

    " Example usage of PSO to find locally D-optimal design for each fixed θ (Emax model) "

    results_emax = []

    print("\n--- PSO Results for Emax Model ---\n")
    for theta in theta_cases_emax:
        result_emax = PSO(theta, w, c1, c2, n_particles, n_iter_max, conv_tol, seed)
        results_emax.append(result_emax)
        print(f"\nResults for θ = {theta}")
        print(f"\n  Determinant of Fisher Information Matrix I(ξ*;θ):")
        print(f"      det(I(ξ*;θ)) = {result_emax['detI_star']}")
        print(f"\n  Negative log determinant of I(ξ*;θ):")
        print(f"      -log(det(I(ξ*;θ))) = {result_emax['neg_log_detI_star']}")
        print(f"\n  Number of iterations until convergence: ")
        print(f"      n_stop = {result_emax['n_stop']}")
        print(f"\n  Convergence difference: ")
        print(f"      |diff| = {result_emax['stop_diff']}\n")

        # Clean support points and weights and print cleaned design
        x_star_clean, w_star_clean = compress_design(result_emax['x_star'], result_emax['w_star'])
        print(f"  Cleaned support points x* (merged duplicates, removed near-zero weights and renormalized):")
        print(f"      x* = {x_star_clean}") 
        print(f"\n  Cleaned weights w* (corresponding to cleaned support points):")
        print(f"      w* = {w_star_clean}\n")


    " Plot Emax curves and mark optimal support points "

    # Clean the PSO design
    for res in results_emax:
        res['x_star'], res['w_star'] = compress_design(res['x_star'], res['w_star'])

    # Counter for cases to label them in the plot
    theta_case = 1

    for res in results_emax:
        x_max = res['x_max']
        plot_curve_and_design(res['theta'], 
                              res['x_star'], 
                              res['w_star'], 
                              x_max, 
                              eta_emax, 
                              theta_case, 
                              title=None)
        theta_case += 1


    # -------------------------------------------
    # GWO on Emax model
    # -------------------------------------------

    " Example usage of GWO to find locally D-optimal design for different θ (Emax model) "

    results_emax_gwo = []

    print(" \n--- GWO Results for Emax Model ---\n")
    for theta in theta_cases_emax:
        result_emax = GWO(theta, 
                          n_wolves=n_particles, 
                          n_iter_max=n_iter_max, 
                          conv_tol=conv_tol, 
                          seed=seed)
        results_emax_gwo.append(result_emax)
        print(f"\nResults for θ = {theta}")
        print(f"\n  Determinant of Fisher Information Matrix I(ξ*;θ):")
        print(f"      det(I(ξ*;θ)) = {result_emax['detI_star']}")
        print(f"\n  Negative log determinant of I(ξ*;θ):")
        print(f"      -log(det(I(ξ*;θ))) = {result_emax['neg_log_detI_star']}")
        print(f"\n  Number of iterations until convergence: ")
        print(f"      n_stop = {result_emax['n_stop']}")
        print(f"\n  Convergence difference: ")
        print(f"      |diff| = {result_emax['stop_diff']}\n")

        # Clean support points and weights and print cleaned design
        x_star_clean, w_star_clean = compress_design(result_emax['x_star'], result_emax['w_star'])
        print(f"  Cleaned support points x* (merged duplicates, removed near-zero weights and renormalized):")
        print(f"      x* = {x_star_clean}")
        print(f"\n  Cleaned weights w* (corresponding to cleaned support points):")
        print(f"      w* = {w_star_clean}\n")


    " Plot Emax curves and mark optimal support points for GWO results "

    # Clean the GWO design
    for res in results_emax_gwo:
        res['x_star'], res['w_star'] = compress_design(res['x_star'], res['w_star'])

    # Counter for cases to label them in the plot
    theta_case = 1

    for res in results_emax_gwo:
        x_max = res['x_max']
        plot_curve_and_design(res['theta'],
                              res['x_star'], 
                              res['w_star'], 
                              x_max, 
                              eta_emax, 
                              theta_case, 
                              title=None)
        theta_case += 1





    





