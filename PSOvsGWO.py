import numpy as np
from time import perf_counter as pc
import concurrent.futures as future

import Loc_D_Opt as ldo
from equivalence import equivalence_check



# ============================================================
# Settings
# ============================================================


### Choose ONE case to test

# Example: Michaelis-Menten case 1
#theta = np.array([3.0, 0.5])  # Case 1: V=3, K=0.5
#theta = np.array([7.0, 2.0]) # Case 2: V=7, K=2
#theta = np.array([5.0, 8.0]) # Case 3: V=5, K=8
#theta = np.array([5.0, 4.0]) # Case 4: V=5, K=4

# Example: Emax case 1
#theta = np.array([1.0, 1.0, 0.5])  # Case 1: V=1, K=1, h=0.5, very gradual response
theta = np.array([2.5, 2.0, 6.0])    # Case 2: V=3, K=1, h=6, very steep response
#theta = np.array([2.0, 6.0, 2.0])    # Case 3: V=2, K=6, h=2, late saturation (needs large x to approach V)
#theta = np.array([3.0, 0.2, 2.0])    # Case 4: V=3, K=1, h=2, early saturation (half-max very early)


### Number of runs for each algorithm
n_runs = 1000


### Maximum number of iterations for each optimization run (PSO and GWO)
n_iter_max = 1000


### Parallel workers:

# Set how many worker processes the process pool uses.
""" 
Leave as None to let Python's ProcessPoolExecutor decide,
or set e.g. max_workers=8, then, at most 8 runs will be executed simultaneously.
"""
max_workers = None

"""
Each worker process handles one run at a time,
after a worker finishes one run, it can take the next submitted run 
"""


### Equivalence check settings:

# tol for optimality check: max Φ(x, ξ*) ≤ tol and |Φ(x, ξ*)| ≤ tol at support points
tol = 5e-3

# Number of grid points to evaluate Φ(x, ξ*) on for the optimality check
n_grid = 2000

# tol for merging support points close to each other when cleaning design in equivalence check
x_tol = 1e-3

# tol for removing near-zero weights when cleaning design in equivalence check
w_tol = 1e-3



# ============================================================
# Helpers
# ============================================================


" Helper function to choose correct gradient "

def get_gradient(theta):
    if len(theta) == 2:
        return ldo.gradient_mm
    elif len(theta) == 3:
        return ldo.gradient_emax
    else:
        raise ValueError("theta must have length 2 (MM) or 3 (Emax)")



"""
The following function takes one list of numbers and returns the exact summary statistics 
of interest.

For example, if by giving it the 1000 PSO runtimes, it returns:

    - max runtime
    - Q3 runtime
    - median runtime
    - Q1 runtime
    - min runtime
    - mean runtime

The same function can then be reused for:

    - PSO n_stop
    - PSO runtime
    - PSO max_abs_Phi_at_support
    - GWO n_stop
    - GWO runtime
    - GWO max_abs_Phi_at_support

n_stop: number of iterations until stopping criterion is met 
        (can be different for each run)
runtime: time in seconds until stopping criterion is met (can be different for each run)
max_abs_Phi_at_support: the maximum absolute value of Φ(x, ξ*) at the support points of the 
                      design found by the algorithm (should be ≤ tol for a successful run, 
                      can be different for each run)
"""


" Helper function to get list of actual summary statistics "

def summarize(values):
    """
    Compute max, Q3, median, Q1, min, mean for a 1D array-like.
    """
    # Convert input values to numpy array (easy computation with built-in numpy functions)
    arr = np.array(values)

    # Return dictionary with summary statistics as floats (better readability when printed)
    return {
        'max': float(np.max(arr)),
        'Q3': float(np.quantile(arr, 0.75)),
        'median': float(np.median(arr)),
        'Q1': float(np.quantile(arr, 0.25)),
        'min': float(np.min(arr)),
        'mean': float(np.mean(arr))
    }



" Helper function to print summary statistics in a nice format "

def print_summary(title, summary):

    def format_value(val):
        # If close to zero, use scientific notation
        if abs(val) < 1e-6:
            return f"{val:.2e}"   # scientific notation
        # Otherwise just show 4 decimals
        else:
            return f"{val:.4f}"   # normal decimal

    print(f"\n{title}:")
    print(f"  max    = {format_value(summary['max'])}")
    print(f"  Q3     = {format_value(summary['Q3'])}")
    print(f"  median = {format_value(summary['median'])}")
    print(f"  Q1     = {format_value(summary['Q1'])}")
    print(f"  min    = {format_value(summary['min'])}")
    print(f"  mean   = {format_value(summary['mean'])}")



# ============================================================
# Single-run worker functions
# ============================================================


"""
These are the core worker functions. Each one performs one single run of the algorithm and 
returns all the values of interest. Then a parallel setup is used to run many copies of 
these functions in parallel. Each single-run worker function will be executed by one worker 
process independently of other runs, and after a worker finishes one run, it can take the 
next submitted run.
"""


" Worker function for PSO "

def one_PSO_run(run_id, theta, tol, n_grid, x_tol, w_tol):
    """
    Run PSO once, measure runtime, store n_stop, and compute Phi_at_support diagnostics.

    Parameters:
    - run_id: an identifier for the run, run_id = 0, 1, ..., n_runs-1.
              Useful for labelling the run and generating unique seeds for each run.
    - theta: the parameter vector for the model (length 2 for MM, length 3 for Emax)
    - tol, n_grid, x_tol, w_tol: settings for the equivalence check

    Returns:
    dict
        A dictionary containing all the values of interest for this run, e.g.:
        runtime, n_stop, Phi_at_support and max Phi_at_support, etc.
    """

    # Ensure theta is a numpy array of floats for consistency
    theta = np.array(theta, dtype=float)

    # Choose correct gradient function based on theta
    gradient = get_gradient(theta)

    # Compute x_max based on theta for the equivalence check
    x_max = ldo.get_x_max(theta)

    # Create different random seed for this run for reproducibility
    seed = 100000 + run_id  

    # Start timer
    start = pc()
    # Run PSO optimization
    res_pso = ldo.PSO(theta, n_iter_max=n_iter_max, seed=seed) # Contains 'x_star', 'w_star', 'n_stop'
    # Stop timer
    stop = pc()

    # Compute runtime
    runtime = stop - start

    # Run equivalence check for PSO result
    eq_res = equivalence_check(     # Contains 'x_star_cleaned', 'w_star_cleaned', 
        theta,                      # 'Phi_at_support', 'max_abs_Phi_at_support', 'converged' etc.
        res_pso['x_star'], 
        res_pso['w_star'], 
        x_max=x_max, 
        gradient=gradient, 
        FIM=ldo.FIM, 
        n_grid=n_grid, 
        w_tol=w_tol, 
        x_tol=x_tol, 
        tol=tol
    )

    # Get absolute values at support points |Φ(xi, ξ*)|
    abs_Phi_at_support = np.abs(eq_res['Phi_at_support'])

    # Return all values of interest in a dictionary
    return {
        "run_id": run_id,   # Run number
        "algorithm": "PSO",     # Algorithm name
        "runtime": runtime,     # Time in seconds until stopping criterion is met
        "n_stop": int(res_pso["n_stop"]),   # Number of iterations until stopping criterion is met
        "abs_Phi_at_support": abs_Phi_at_support.tolist(),  # List of |Φ(xi, ξ*)| at the support points
        "max_abs_Phi_at_support": float(eq_res["max_abs_Phi_at_support"]),  # Max |Φ(xi, ξ*)| at the support points
        "n_support": int(len(abs_Phi_at_support)),    # Number of support points in the design
        "x_star_cleaned": eq_res["x_star_cleaned"].tolist(),    # List of support points after cleaning
        "w_star_cleaned": eq_res["w_star_cleaned"].tolist(),    # List of weights after cleaning
        "success": eq_res["converged"]  # Whether the design passed the optimality check (max |Φ(x, ξ*)| ≤ tol and |Φ(xi, ξ*)| ≤ tol at support points)
    }



" Worker function for GWO "

def one_GWO_run(run_id, theta, tol, n_grid, x_tol, w_tol):
    """ 
    Run GWO once, measure runtime, store n_stop, and compute Phi_at_support diagnostics.
    
    Parameters:

    - run_id: an identifier for the run, run_id = 0, 1, ..., n_runs-1.
              Useful for labelling the run and generating unique seeds for each run.         
    - theta: the parameter vector for the model (length 2 for MM, length 3 for Emax)
    - tol, n_grid, x_tol, w_tol: settings for the equivalence check

    Returns:
    dict
        A dictionary containing all the values of interest for this run, e.g.:
        runtime, n_stop, Phi_at_support and max Phi_at_support, etc.
    """

    # Ensure theta is a numpy array of floats for consistency
    theta = np.array(theta, dtype=float)

    # Choose correct gradient function based on theta
    gradient = get_gradient(theta)

    # Compute x_max based on theta for the equivalence check
    x_max = ldo.get_x_max(theta)

    # Create different random seed for this run for reproducibility
    seed = 200000 + run_id 
    
    # Start timer
    start = pc()
    # Run GWO optimization
    res_gwo = ldo.GWO(theta, n_iter_max=n_iter_max, seed=seed) # Contains 'x_star', 'w_star', 'n_stop'
    # Stop timer
    stop = pc()

    # Compute runtime
    runtime = stop - start

    # Run equivalence check for GWO result
    eq_res = equivalence_check(     # Contains 'x_star_cleaned', 'w_star_cleaned', 
        theta,                      # 'Phi_at_support', 'max_abs_Phi_at_support', 'converged' etc.
        res_gwo['x_star'],
        res_gwo['w_star'], 
        x_max=x_max, 
        gradient=gradient, 
        FIM=ldo.FIM, 
        n_grid=n_grid, 
        w_tol=w_tol, 
        x_tol=x_tol, 
        tol=tol
    )

    # Get absolute values at support points |Φ(xi, ξ*)|
    abs_Phi_at_support = np.abs(eq_res['Phi_at_support'])

    # Return all values of interest in a dictionary
    return {
        "run_id": run_id,   # Run number
        "algorithm": "GWO",     # Algorithm name
        "runtime": runtime,     # Time in seconds until stopping criterion is met
        "n_stop": int(res_gwo["n_stop"]),   # Number of iterations until stopping criterion is met
        "abs_Phi_at_support": abs_Phi_at_support.tolist(),  # List of |Φ(xi, ξ*)| at the support points
        "max_abs_Phi_at_support": float(eq_res["max_abs_Phi_at_support"]),  # Max |Φ(xi, ξ*)| at the support points
        "n_support": int(len(abs_Phi_at_support)),    # Number of support points in the design
        "x_star_cleaned": eq_res["x_star_cleaned"].tolist(),    # List of support points after cleaning
        "w_star_cleaned": eq_res["w_star_cleaned"].tolist(),    # List of weights after cleaning
        "success": eq_res["converged"]  # Whether the design passed the optimality check (max |Φ(x, ξ*)| ≤ tol and |Φ(xi, ξ*)| ≤ tol at support points)
    }



# ============================================================
# Parallel experiment
# ============================================================


"""
The next function does:
    - one_pso_run(...) n_runs=1000 times
    - one_gwo_run(...) n_runs=1000 times
    - both in parallel using processes
    - collects all returned dictionaries
"""


" Function to run all PSO and GWO runs in parallel and collect results "

def run_parallel_experiment(theta, n_runs=1000, max_workers=None, 
                            tol=5e-3, n_grid=2000, x_tol=1e-3, w_tol=1e-3):
    
    # Ensure theta is a numpy array of floats for consistency
    theta = np.array(theta, dtype=float)

    # Create empty list to store n_runs PSO result dictionaries
    pso_results = []
    # Create empty list to store n_runs GWO result dictionaries
    gwo_results = []

    print(f"\nStarting parallel experiment with n_runs = {n_runs}")

    " First batch: all PSO runs "

    # Create a pool (collection) of worker processes waiting to run all PSO runs in parallel
    with future.ProcessPoolExecutor(max_workers=max_workers) as ex:

        " Submit all PSO jobs "
        # Submit to the process pool to get a list of Future objects representing the pending results
        pso_futures = [
            ex.submit(one_PSO_run, run_id=i, theta=theta, tol=tol, n_grid=n_grid, x_tol=x_tol, w_tol=w_tol) 
            for i in range(n_runs)
        ] # List comprehension to submit n_runs PSO jobs with different run_id

        " Collect PSO results as they complete "
        # Use generator future.as_completed to iterate over the Future objects in the order they finish
        for j, f in enumerate(future.as_completed(pso_futures), start=1):
            # Store PSO result (dict) from the jth completed future f in the pso_results list
            pso_results.append(f.result())
            # Print progress every 50 completed runs
            if j % 50 == 0 or j == n_runs:
                print(f"  Completed {j}/{n_runs} PSO runs")

    " Second batch: all GWO runs "

    # Create a new pool of worker processes waiting to run all GWO runs in parallel
    with future.ProcessPoolExecutor(max_workers=max_workers) as ex:

        " Submit all GWO jobs "
        # Submit to the process pool to get a list of Future objects representing the pending results
        gwo_futures = [
            ex.submit(one_GWO_run, run_id=i, theta=theta, tol=tol, n_grid=n_grid, x_tol=x_tol, w_tol=w_tol) 
            for i in range(n_runs)
        ] # List comprehension to submit n_runs GWO jobs with different run_id

        " Collect GWO results as they complete "
        # Use generator future.as_completed to iterate over the Future objects in the order they finish
        for j, f in enumerate(future.as_completed(gwo_futures), start=1):
            # Store GWO result (dict) from the jth completed future f in the gwo_results list
            gwo_results.append(f.result())
            # Print progress every 50 completed runs
            if j % 50 == 0 or j == n_runs:
                print(f"  Completed {j}/{n_runs} GWO runs")

    # Sort results by run_id to ensure they are in the same order as the runs were submitted
    pso_results.sort(key=lambda x: x['run_id']) # Use 'run_id' value from each dict x in pso_results to sort the list
    gwo_results.sort(key=lambda x: x['run_id']) # Use 'run_id' value from each dict x in gwo_results to sort the list

    return {
        "theta": theta, # The parameter vector used for this experiment (same for all runs)
        "pso_results": pso_results, # List of dicts with PSO results for each run
        "gwo_results": gwo_results  # List of dicts with GWO results for each run
    }



# ============================================================
# Post-processing 
# ============================================================


"""
From run_parallel_experiment(...) we get raw results:
    - 1000 PSO dictionaries
    - 1000 GWO dictionaries

Each dictionary contains things like:
    - runtime
    - n_stop
    - Phi_at_support
    - max_abs_Phi_at_support

The final output should be the following summary statistics for those values:

                max, Q3, median, Q1, min, mean

So the next step is:
    1) Extract relevant quantities and store them in:
        - lists of runtimes
        - lists of n_stop
        - lists of max_abs_Phi_at_support
    2) Summarize each list with summarize(...) to get the summary statistics
    3) Optionally also summarize support-point-wise |Φ(x_i,xi*)| values

That is exactly what the next function does.
"""


" Function to build diagnostic summary from the raw results of the parallel experiment "

def build_diagnostic_summary(results):
    """
    Build summary statistics for runtime, n_stop, and max_abs_Phi_at_support for both PSO and GWO.

    Parameters:
    results (dict): The raw results dictionary returned by run_parallel_experiment(...), containing:
                    - theta
                    - pso_results: list of dicts with PSO results for each run
                    - gwo_results: list of dicts with GWO results for each run
    Returns:
    summary (dict): A dictionary containing the summary statistics for each quantity of interest, e.g.:
                    {
                        "PSO_runtime": {...},
                        "PSO_n_stop": {...},
                        "PSO_max_abs_Phi_at_support": {...},
                        "GWO_runtime": {...},
                        "GWO_n_stop": {...},
                        "GWO_max_abs_Phi_at_support": {...}
                    }
    """

    # Extract PSO and GWO results
    pso_results = results["pso_results"]
    gwo_results = results["gwo_results"]

    # Extract success indicators (True/False per run) for PSO and GWO
    pso_successes = [res['success'] for res in pso_results]
    gwo_successes = [res['success'] for res in gwo_results]

    # Full success rates in percent (max |Φ(x, ξ*)| ≤ tol and |Φ(xi, ξ*)| ≤ tol at support points)
    pso_success_rate = 100 * np.mean(pso_successes) # True behaves like 1, False like 0
    gwo_success_rate = 100 * np.mean(gwo_successes)

    # Support-only success rates in percent (|Φ(xi, ξ*)| ≤ tol at support points)
    pso_support_successes = [bool(res['max_abs_Phi_at_support'] <= tol) for res in pso_results]
    gwo_support_successes = [bool(res['max_abs_Phi_at_support'] <= tol) for res in gwo_results]
    pso_support_success_rate = 100 * np.mean(pso_support_successes)
    gwo_support_success_rate = 100 * np.mean(gwo_support_successes)

    # Extract runtimes, n_stop, and max_abs_Phi_at_support for PSO
    pso_runtimes = [res['runtime'] for res in pso_results]
    pso_n_stops = [res['n_stop'] for res in pso_results]
    pso_max_abs_Phi = [res['max_abs_Phi_at_support'] for res in pso_results]

    # Extract runtimes, n_stop, and max_abs_Phi_at_support for GWO
    gwo_runtimes = [res['runtime'] for res in gwo_results] 
    gwo_n_stops = [res['n_stop'] for res in gwo_results]
    gwo_max_abs_Phi = [res['max_abs_Phi_at_support'] for res in gwo_results]

    # Summarize each list with summarize(...) to get the summary statistics
    summary = {
        "PSO_runtime": summarize(pso_runtimes),
        "PSO_n_stop": summarize(pso_n_stops),
        "PSO_max_abs_Phi_at_support": summarize(pso_max_abs_Phi),
        "PSO_success_rate": float(pso_success_rate),
        "PSO_support_success_rate": float(pso_support_success_rate),

        "GWO_runtime": summarize(gwo_runtimes),
        "GWO_n_stop": summarize(gwo_n_stops),
        "GWO_max_abs_Phi_at_support": summarize(gwo_max_abs_Phi),
        "GWO_success_rate": float(gwo_success_rate),
        "GWO_support_success_rate": float(gwo_support_success_rate)
    }

    return summary



" Function to print the diagnostic summary in a nice format "

def print_full_summary(summary, tol=tol):
    """
    Print all main summaries.
    """
    print("\n==================== SUMMARY ====================")

    print_summary("PSO n_stop", summary["PSO_n_stop"])
    print_summary("PSO runtime", summary["PSO_runtime"])
    print_summary("PSO max(|Φ(x_i, ξ*)|)", summary["PSO_max_abs_Phi_at_support"])
    print(f"\nFor the full Equivalence thm. check (max |Φ(x, ξ*)| ≤ {tol} and |Φ(xi, ξ*)| ≤ {tol} at support points):")
    print(f"     PSO success rate = {summary['PSO_success_rate']:.2f}%")
    print(f"For the support-only check (|Φ(xi, ξ*)| ≤ {tol} at support points):")
    print(f"     PSO success rate = {summary['PSO_support_success_rate']:.2f}%\n")

    print_summary("GWO n_stop", summary["GWO_n_stop"])
    print_summary("GWO runtime", summary["GWO_runtime"])
    print_summary("GWO max(|Φ(x_i, ξ*)|)", summary["GWO_max_abs_Phi_at_support"])
    print(f"\nFor the full Equivalence thm. check (max |Φ(x, ξ*)| ≤ {tol} and |Φ(xi, ξ*)| ≤ {tol} at support points):")
    print(f"     GWO success rate = {summary['GWO_success_rate']:.2f}%")
    print(f"For the support-only check (|Φ(xi, ξ*)| ≤ {tol} at support points):")
    print(f"     GWO success rate = {summary['GWO_support_success_rate']:.2f}%\n")

    if "PSO_supportwise_abs_Phi" in summary:
        print("\nPSO support-point-wise |Φ(x_i, ξ*)| summaries")
        for key, stats in summary["PSO_supportwise_abs_Phi"].items():
            print_summary(f"  {key}", stats)

    if "GWO_supportwise_abs_Phi" in summary:
        print("\nGWO support-point-wise |Φ(x_i, ξ*)| summaries")
        for key, stats in summary["GWO_supportwise_abs_Phi"].items():
            print_summary(f"  {key}", stats)



# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    results = run_parallel_experiment(
        theta=theta,
        n_runs=n_runs,
        max_workers=max_workers,
        tol=tol,
        n_grid=n_grid,
        w_tol=w_tol,
        x_tol=x_tol
    )

    summary = build_diagnostic_summary(results)
    print_full_summary(summary)