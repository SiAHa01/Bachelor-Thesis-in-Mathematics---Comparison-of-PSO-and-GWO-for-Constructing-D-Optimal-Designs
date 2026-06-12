import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from Loc_D_Opt import compress_design

"""
Let the directional derivative Φ(x, ξ*) be defined as:

    Φ(x, ξ*) = f(x)^T M(ξ*)^-1 f(x) - p

For a nonlinear regression model with parameter dimension p, the General Equivalence 
Theorem states that a design ξ* is locally D-optimal if and only if the following 
condition holds for all x in the design space X:

    Φ(x, ξ*) ≤ 0     for all x ∈ X
and 
    Φ(x_i, ξ*) = 0     for all support points x_i of ξ*

I.e., Φ(x, ξ*) must be non-positive for all x in the design space, and it must equal zero 
at the support points with positive weights. 

Let d be the sensitivity function for a design ξ* defined as:

    d(x, ξ*) = f(x)^T M(ξ*)^-1 f(x)

Then,

    Φ(x, ξ*) = d(x, ξ*) - p
"""


"""
This module contains functions to verify the General Equivalence Theorem. It defines functions
for the following:

    1. Compute Φ(x, ξ*) across a dense grid of x values in the design space, using the 
       cleaned design and the model's sensitivity function.

    2. Print results and plot Φ(x, ξ*) against x, highlighting the support points of the 
       design. The plot should show that Φ(x, ξ*) is non-positive across the design space 
       and equals zero at the support points, confirming that the design satisfies the 
       General Equivalence Theorem for D-optimality.
"""


" Implementation of helper for computing the sensitivity function d(x, ξ*) across a grid of x values. "

def d(x_grid, theta, x_star, w_star, gradient, FIM):
    """ Compute the sensitivity function d(x, ξ*) across a grid of x values. 
    
    Parameters:
    x_grid (array-like): Grid of x values in the design space.
    theta (array-like): Current parameter estimates.
    x_star (array-like): Support points of the design.
    w_star (array-like): Corresponding weights of the support points.
    gradient (callable): Function to compute the gradient f(x) of the model at given x and theta.
    fim (callable): Function to compute the Fisher Information Matrix M(ξ*) for the design.
    
    Returns:
    array: Sensitivity function values d(x, ξ*) for each x in the grid.
    """
    # Construct the design ξ* from the support points and weights
    xi = np.concatenate([x_star, w_star])  

    # Compute the Fisher Information Matrix M(ξ*) for the design
    M = FIM(xi, theta)

    # Invert M(ξ*)
    M_inv = np.linalg.inv(M)

    p = len(theta)

    # Compute d(x, ξ*) for each x in the grid
    d_values = []
    for x in x_grid:
        # Compute the gradient f(x) at the current x as an array (returns shape (1,p))  
        f_x = np.asarray(gradient(np.array([x], dtype=float), theta), dtype=float)
        # Valid shapes for the formula are (p,) or (p,1)
        if f_x.shape == (1, p):
            f_x = f_x.T 

        # Or squeeze to convert shape (1,p) to (p,)
        # f_x = np.squeeze(f_x_raw)

        # Safety check for correct shape
        if f_x.size != p:
            raise ValueError(
                f"gradient(x, theta) must return shape (p,), (p,1), or (1,p) with p={p}. "
                f"Got shape {f_x.shape} for x={x}."
            )
        
        # Compute d(x, ξ*) as scalar
        d_x = float(f_x.T @ M_inv @ f_x) 

        # For squeezed f_x, the line above is equivalent to:
        # d_x = float(f_x @ M_inv @ f_x)  # (p,) transposed is still (p,)

        d_values.append(d_x)

    return np.array(d_values)



" Implementation of helper for computing the directional derivative d(x, ξ*) across a grid of x values."

def Phi(x_grid, theta, x_star, w_star, gradient, FIM):
    """ Compute the directional derivative d(x, ξ*) across a grid of x values. 
    
    Parameters:
    x_grid (array-like): Grid of x values in the design space.
    theta (array-like): Current parameter estimates.
    x_star (array-like): Support points of the design.
    w_star (array-like): Corresponding weights of the support points.
    gradient (callable): Function to compute the gradient f(x) of the model at given x and theta.
    fim (callable): Function to compute the Fisher Information Matrix M(ξ*) for the design.
    
    Returns:
    array: Directional derivative values Φ(x, ξ*) for each x in the grid.
    """
    d_values = d(x_grid, theta, x_star, w_star, gradient, FIM)
    p = len(theta)
    Phi_values = d_values - p
    return Phi_values



" Implementation of silent equivalence check "

def equivalence_check(theta,
                               x_star,
                               w_star,
                               x_max,
                               gradient,
                               FIM,
                               n_grid=2000, 
                               w_tol=1.5e-3, 
                               x_tol=1.5e-3, 
                               tol=5e-3):
    """ 
    Silent computation of the General Equivalence Theorem diagnostics.

    Returns a dictionary containing:
        - cleaned design
        - grid and Φ(x, ξ*) values
        - Φ(x, ξ*) at support points
        - max_Phi
        - max_abs_Phi_at_support
        - converged (Boolean)
    """
    # Clean the design
    x_star, w_star = compress_design(x_star, w_star, x_tol=x_tol, w_tol=w_tol)

    # Create a dense grid of x values in the design space
    x_grid = np.linspace(0, x_max, n_grid)

    # Compute Φ(x, ξ*) across the grid
    Phi_values = Phi(x_grid, theta, x_star, w_star, gradient, FIM)

    # Evaluate Φ(x, ξ*) at the support points (should be close to 0)
    Phi_at_support = Phi(x_star, theta, x_star, w_star, gradient, FIM)

    " Diagnostics "

    # Find the maximum value of Φ(x, ξ*) across the grid 
    max_Phi = float(np.max(Phi_values))

    # Minimum value of Φ(x, ξ*) across the grid (should be ≤ 0)
    min_Phi = float(np.min(Phi_values))

    # Maximum absolute value of Φ(x, ξ*) at the support points (should be close to 0)
    max_abs_Phi_at_support = float(np.max(np.abs(Phi_at_support)))

    # Boolean indicating whether the design satisfies the General Equivalence Theorem for D-optimality
    converged = (max_Phi <= tol) and np.all(np.abs(Phi_at_support) <= tol)

    # Return results in a dictionary
    return {
        "theta": np.array(theta, dtype=float),
        "p": len(theta),
        "x_max": float(x_max),
        "tol": float(tol),
        "n_grid": int(n_grid),
        "x_star_cleaned": x_star,
        "w_star_cleaned": w_star,
        "x_grid": x_grid,
        "Phi_values": Phi_values,
        "Phi_at_support": Phi_at_support,
        "Phi_max": max_Phi,
        "Phi_min": min_Phi,
        "max_abs_Phi_at_support": max_abs_Phi_at_support,
        "converged": converged,
    }


""" Wrapper for interactive use: compute + print + plot """

def check_and_plot_equivalence(theta,
                               x_star,
                               w_star,
                               x_max,
                               gradient,
                               FIM,
                               n_grid=2000,
                               w_tol=1.5e-3,
                               x_tol=1.5e-3,
                               tol=5e-3,
                               title="Equivalence Theorem Check for Local D-Optimality \n \n"):
    """
    Backward-compatible wrapper:
        - runs equivalence_check()
        - prints diagnostics
        - plots the result
        - returns the same dictionary
    """
    # Run the silent equivalence check to get all diagnostics and results
    eq_result = equivalence_check(theta, x_star, w_star, x_max, gradient, FIM, n_grid, w_tol, x_tol, tol)

    # Extract results for easier access
    p = eq_result["p"]
    x_star_cleaned = eq_result["x_star_cleaned"]
    w_star_cleaned = eq_result["w_star_cleaned"]
    Phi_at_support = eq_result["Phi_at_support"]
    max_Phi = eq_result["Phi_max"]
    converged = eq_result["converged"]
    x_grid = eq_result["x_grid"]
    Phi_values = eq_result["Phi_values"]

    # Print summary diagnostics
    print(f"\n---{title}---\n" if title else f"\n---Equivalence Theorem Check for Local D-Optimality---\n")
    print(f"θ = {theta}, p = {p}, x_max = {x_max}\n")
    for i, (x_i, w_i, Phi_i) in enumerate(zip(x_star_cleaned, w_star_cleaned, Phi_at_support)):
        print(f"Support point {i+1}: x{i+1} = {x_i:.10g}, w{i+1} = {w_i:.6f}, Φ(x{i+1}, ξ*) at support = {Phi_i:.6e}")
    print(f"\nMaximum Φ(x, ξ*) across the grid: {max_Phi:.6e}")
    print(f"\nFor tol = {tol} (anything ≤{tol} is considered ~0):")
    if max_Phi <= tol and np.all(np.abs(Phi_at_support) <= tol):
        print(f"SUCCESS (max Φ(x, ξ*) ≤ {tol} and all |Φ(x, ξ*)| at support points ≤ {tol}):\n"
              f"    The design satisfies the General Equivalence Theorem for D-optimality.\n")
    else:
        print(f"FAILURE (max Φ(x, ξ*) > {tol} or some |Φ(x, ξ*)| at support points > {tol}):\n"
              f"    The design does NOT satisfy the General Equivalence Theorem for D-optimality.\n"
              f"    Possible reasons: PSO not fully converged, numerical issues (like rounding errors),\n"
              f"    grid too coarse, design not truly optimal, or tol too strict.\n")
    
    # Plot Φ(x, ξ*) against x
    plt.figure(figsize=(8, 6))
    plt.plot(x_grid, Phi_values, label=r"$\Phi(x, \xi^*) = d(x, \xi^*) - p$", color='blue')
    # Horizontal line at y=0 for reference
    plt.axhline(0, color='gray', linestyle='--', label="Φ(x, ξ*) = 0")
    # Highlight support points as red dots at given coordinates (Φ should be ~0 at these points)
    plt.scatter(x_star_cleaned, Phi_at_support, color='red', zorder=5, label="Actual support points (Φ at support)")

    # Optional: vertical ticks at support points for clarity
    for x_i, w_i in zip(x_star_cleaned, w_star_cleaned):
        # Draw a vertical line of height ymax-ymin=0.08 at each support point
        plt.axvline(x_i, ymin=0.0, ymax=0.08, alpha=0.5)
        # Label weight vertically above the support point
        plt.text(x_i, 0.02, f"w={w_i:.2f}", rotation=90, va='bottom', ha='center', fontsize=8)
    
    plt.xlim(0, x_max) # Sets visible range of x-axis to [0, x_max]
    plt.xlabel("x")
    plt.ylabel(r"$Φ(x, \xi^*)$")
    plt.title(f"{title}\n\n" if title else "Equivalence Theorem Check for Local D-Optimality")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return {
        "x_star_cleaned": x_star_cleaned,
        "w_star_cleaned": w_star_cleaned,
        "Phi_max": max_Phi,
        "Phi_at_support": Phi_at_support,
    }












