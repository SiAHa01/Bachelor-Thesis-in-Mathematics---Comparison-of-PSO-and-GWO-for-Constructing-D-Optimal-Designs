import numpy as np
import Loc_D_Opt as ldo
from equivalence import check_and_plot_equivalence

# -------------------------------------------
# Settings
# -------------------------------------------

# Maximum number of iterations for each optimization run (PSO and GWO)
n_iter_max = 1000

# Assumed parameter values for different Michaelis-Menten cases
theta_case1_mm = np.array([3, 0.5])  # Case 1: V=3, K=0.5
theta_case2_mm = np.array([7, 2]) # Case 2: V=7, K=2
theta_case3_mm = np.array([5, 8]) # Case 3: V=5, K=8
theta_case4_mm = np.array([5, 4]) # Case 4: V=5, K=4

theta_cases_mm = [theta_case1_mm, theta_case2_mm, theta_case3_mm, theta_case4_mm]

# Assumed parameter values for different Emax cases
theta_case1_emax = np.array([1, 1, 0.5])  # Case 1: V=1, K=1, h=0.5, very gradual response 
theta_case2_emax = np.array([2.5, 2, 6])    # Case 2: V=3, K=1, h=6, very steep response
theta_case3_emax = np.array([2, 6, 2])    # Case 3: V=2, K=6, h=2, late saturation (needs large x to approach V)
theta_case4_emax = np.array([3, 0.2, 2])    # Case 4: V=3, K=0.2, h=2, early saturation (half-max very early)

theta_cases_emax = [theta_case1_emax, theta_case2_emax, theta_case3_emax, theta_case4_emax]


# -------------------------------------------
# Optimality check on PSO
# -------------------------------------------

" Local D-optimality check of PSO design for Michaelis-Menten model"

# Compute PSO results for each case
pso_results_mm = []

for theta in theta_cases_mm:
    pso_result_mm = ldo.PSO(theta, n_iter_max=n_iter_max) 
    pso_results_mm.append(pso_result_mm)

# Check and plot equivalence for each case
eq_results_mm_pso = []

for i, (theta, res) in enumerate(zip(theta_cases_mm, pso_results_mm), start=1):
    x_max = ldo.get_x_max(theta)  # Set x_max based on K to ensure we cover the relevant design space
    print(f"\nCase {i}: θ = {theta}")
    eq_result_mm_pso = check_and_plot_equivalence(theta, 
                                              res['x_star'], 
                                              res['w_star'], 
                                              x_max, 
                                              ldo.gradient_mm, 
                                              ldo.FIM, 
                                              title=(f"Michaelis-Menten Case {i}: Directional " 
                                              "Derivative for PSO Design"))
    eq_results_mm_pso.append(eq_result_mm_pso)



" Local D-optimality check of PSO design for Emax model "

# Compute PSO results for each case
pso_results_emax = []

for theta in theta_cases_emax:
    pso_result_emax = ldo.PSO(theta, n_iter_max=n_iter_max)
    pso_results_emax.append(pso_result_emax)

# Check and plot equivalence for each case
eq_results_emax_pso = []

for i, (theta, res) in enumerate(zip(theta_cases_emax, pso_results_emax), start=1):
    x_max = ldo.get_x_max(theta)  # Set x_max based on K to ensure we cover the relevant design space
    print(f"\nCase {i}: θ = {theta}")
    eq_result_emax_pso = check_and_plot_equivalence(theta, 
                                               res['x_star'], 
                                               res['w_star'], 
                                               x_max, 
                                               ldo.gradient_emax, 
                                               ldo.FIM, 
                                               title=(f"Emax Case {i}: Directional " 
                                              "Derivative for PSO Design"))
    eq_results_emax_pso.append(eq_result_emax_pso)


# -------------------------------------------
# Optimality check on GWO
# -------------------------------------------

" Local D-optimality check of GWO design for Michaelis-Menten model"

# Compute GWO results for each case
gwo_results_mm = []

for theta in theta_cases_mm:
    gwo_result_mm = ldo.GWO(theta, n_iter_max=n_iter_max)
    gwo_results_mm.append(gwo_result_mm)

# Check and plot equivalence for each case
eq_results_mm_gwo = []

for i, (theta, res) in enumerate(zip(theta_cases_mm, gwo_results_mm), start=1):
    x_max = ldo.get_x_max(theta)  # Set x_max based on K to ensure we cover the relevant design space
    print(f"\nCase {i}: θ = {theta}")
    eq_result_mm_gwo = check_and_plot_equivalence(theta, 
                                              res['x_star'], 
                                              res['w_star'], 
                                              x_max, 
                                              ldo.gradient_mm, 
                                              ldo.FIM, 
                                              title=(f"Michaelis-Menten Case {i}: Directional " 
                                              "Derivative for GWO Design"))
    eq_results_mm_gwo.append(eq_result_mm_gwo)


" Local D-optimality check of GWO design for Emax model "

# Compute GWO results for each case
gwo_results_emax = []

for theta in theta_cases_emax:
    gwo_result_emax = ldo.GWO(theta, n_iter_max=n_iter_max)  
    gwo_results_emax.append(gwo_result_emax)

# Check and plot equivalence for each case
eq_results_emax_gwo = []

for i, (theta, res) in enumerate(zip(theta_cases_emax, gwo_results_emax), start=1):
    x_max = ldo.get_x_max(theta)  # Set x_max based on K to ensure we cover the relevant design space
    print(f"\nCase {i}: θ = {theta}")
    eq_result_emax_gwo = check_and_plot_equivalence(theta, 
                                               res['x_star'], 
                                               res['w_star'], 
                                               x_max, 
                                               ldo.gradient_emax, 
                                               ldo.FIM, 
                                               title=(f"Emax Case {i}: Directional " 
                                              "Derivative for GWO Design"))
    eq_results_emax_gwo.append(eq_result_emax_gwo)




