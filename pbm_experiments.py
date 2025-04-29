"""
Run and post-process PyBaMM DFN simulations for a range of fast-charge/discharge 
protocols, extract cycling features, and save them to disk for downstream modeling.

Contents:
  • MODEL_OPTIONS, VAR_PTS, GROUPS  
      Configuration dicts for the DFN physics options, discretization, and
      experimental groups (charge/discharge rates, cycle counts, etc.).

  • get_parameter_values()  
      Build and tweak an OKane2022/Prada2013 ParameterValues object.

  • run_experiment(), run_group_experiment()  
      Helpers to assemble PyBaMM experiments (CCCV, RPT, ageing, derated) and
      solve them for a given group.

  • plot_results()  
      Quick scatter of discharge capacity vs. RPT number.

  • extract_features()  
      Parse PyBaMM SolutionList objects to compute Ah throughput, time under
      load, capacity drops, etc., returning a pandas.DataFrame. 

  • run_group_by_id() / main  
      Orchestrates a full workflow: run sims, extract features, plot, and
      pickle the resulting DataFrame to `simulated_PBM_data/`.

Usage:
    $ python pbm_experiments.py
    Enter the group ID ['G1','G2',…,'G18']: G3
"""


import pybamm
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import random
import os

# —————————————————————————————
# CONFIGURATION
# —————————————————————————————
# Model options for DFN and additional physics
MODEL_OPTIONS = {
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "partially reversible",
    "lithium plating porosity change": "true",  # alias for "SEI porosity change"
    "particle mechanics": ("swelling and cracking", "swelling only"),
    "SEI on cracks": "true",
    "loss of active material": "stress-driven",
    "calculate discharge energy": "true",  # for compatibility with older PyBaMM versions
}

# Discretization points for spatial and particle variables
VAR_PTS = {
    "x_n": 3,  # negative electrode
    "x_s": 3,  # separator
    "x_p": 3,  # positive electrode
    "r_n": 9,  # negative particle
    "r_p": 9,  # positive particle
}


def get_parameter_values():
    """
    Create and update parameter values using PyBaMM's OKane2022 parameter set.
    
    Returns:
        pybamm.ParameterValues: Updated parameter values.
    """
    
    parameter_values = pybamm.ParameterValues("OKane2022")
    parameter_values_ = pybamm.ParameterValues("Prada2013")   # To avoid the parameter set error! 

    # Voltage and OCP settings
    parameter_values.update({
        'Positive electrode OCP [V]': pybamm.input.parameters.lithium_ion.Prada2013.LFP_ocp_Afshar2017,
        'Lower voltage cut-off [V]': "2.5",
        'Upper voltage cut-off [V]': "3.6",
        'Open-circuit voltage at 0% SOC [V]': '2.5',
        'Open-circuit voltage at 100% SOC [V]': "3.6",
        'Nominal cell capacity [A.h]': "1.2",
        'Current function [A]': '1.2'
    })
    # Concentration parameters
    parameter_values.update({
        'Maximum concentration in positive electrode [mol.m-3]': "31170.0",
        'Maximum concentration in negative electrode [mol.m-3]': "45850.0",
        'Initial concentration in positive electrode [mol.m-3]': "29440.0",
        'Initial concentration in negative electrode [mol.m-3]': "20100.0",
    })
    # Geometry settings
    parameter_values.update({
        'Separator thickness [m]': '1.9e-05',
        'Positive electrode thickness [m]': '5.8e-05',
        'Negative electrode thickness [m]': '3.6e-05',
        'Electrode height [m]': '0.055',
        'Electrode width [m]': '0.850'
    })
    # Positive electrode properties
    parameter_values.update({
        'Positive electrode active material volume fraction': '0.54',
        'Positive electrode charge transfer coefficient': '0.5',
        'Positive electrode conductivity [S.m-1]': '0.34',
        'Positive electrode diffusivity [m2.s-1]': '6e-15',
        'Positive electrode porosity': '0.426',
        'Positive particle radius [m]': '5.22e-06'
    })
    # Additional parameters
    parameter_values.update({
        'Reference temperature [K]': '298',
        'Separator porosity': '0.45',
        'Negative electrode active material volume fraction': 0.75,
        'Negative electrode charge transfer coefficient': 0.5,
        'Negative electrode conductivity [S.m-1]': 215.0,
        'Negative electrode diffusivity [m2.s-1]': 5e-14,
        'Negative electrode exchange-current density [A.m-2]':
            pybamm.input.parameters.lithium_ion.Prada2013.graphite_LGM50_electrolyte_exchange_current_density_Chen2020,
        'Negative electrode porosity': 0.36,
        'Negative particle radius [m]': '5.86e-06',
        'SEI reaction exchange current density [A.m-2]': '1.5e-06',
        'SEI kinetic rate constant [m.s-1]': '2e-12',
        'Lithium plating kinetic rate constant [m.s-1]': '1e-08'
    })

    return parameter_values


# —————————————————————————————
# EXPERIMENTAL GROUPS
# —————————————————————————————
# One-step fast charging groups:
GROUPS = {
    "G1": {"charge_current": 6.6468, "discharge_current": 2.7984, "discharge_cutoff": 2.6804, "M1": 4, "M2": 16, "N": 120},    # M and N are chosen based on the transition point observed in experimental data (i.e., the RPT and the region at which battery capacity transition happens)
    "G2": {"charge_current": 6.8844, "discharge_current": 2.9916, "discharge_cutoff": 3.093, "M1": 6, "M2": 14, "N": 120},
    "G3": {"charge_current": 4.8972, "discharge_current": 4.3356, "discharge_cutoff": 2.8754, "M1": 9, "M2": 11, "N": 120},
    "G4": {"charge_current": 6.1452, "discharge_current": 4.158, "discharge_cutoff": 3.006, "M1": 9, "M2": 11, "N": 100},
    "G5": {"charge_current": 5.7216, "discharge_current": 3.6408, "discharge_cutoff": 3.0535, "M1": 9, "M2": 11, "N": 100},
    "G16": {"charge_current": 5.0988, "discharge_current": 2.5428, "discharge_cutoff": 2.828, "M1": 10, "M2": 10, "N": 80},
    "G17": {"charge_current": 5.8104, "discharge_current": 3.8016, "discharge_cutoff": 3.0210, "M1": 8, "M2": 12, "N": 80},
    "G18": {"charge_current": 5.4936, "discharge_current": 4.0896, "discharge_cutoff": 2.897, "M1": 7, "M2": 13, "N": 100},
    # Two-step fast charging groups:
    "G13": {"charge_current": 4.8876, "charge_cutoff": 3.5329, "discharge_current": 4.7664, "discharge_cutoff": 3.009, "M1": 13, "M2": 7, "N": 100},
    "G15": {"charge_current": 5.7984, "charge_cutoff": 3.5502, "discharge_current": 3.99,   "discharge_cutoff": 3.0537, "M1": 14, "M2": 6, "N": 100},
}

# —————————————————————————————
# SIMULATION FUNCTIONS
# —————————————————————————————
def run_experiment(sim_model, experiment, parameter_values, var_pts, starting_solution=None):
    """
    Run a PyBaMM simulation using the provided model, experiment, parameters, and discretization.
    
    Args:
        sim_model: The PyBaMM model.
        experiment: The PyBaMM experiment object.
        parameter_values: The parameter values for the simulation.
        var_pts: Spatial discretization information.
        starting_solution: (Optional) A starting solution.
        
    Returns:
        The simulation solution.
    """
    sim = pybamm.Simulation(sim_model, experiment=experiment,
                            parameter_values=parameter_values, var_pts=var_pts)
    return sim.solve(starting_solution=starting_solution, calc_esoh=False)


def run_group_experiment(group_id, model, parameter_values):
    """
    Run experiments (initial, ageing, and derated cycles) for a specified group.
    For groups G13 and G15 the CCCV experiment uses a two-step fast charging protocol.
    
    Args:
        group_id (str): Experimental group identifier.
        model: The PyBaMM model.
        parameter_values: The simulation parameters.
        
    Returns:
        Tuple containing:
            - list: CCCV simulation solutions.
            - list: Charge simulation solutions.
            - list: RPT simulation solutions.
            - list: RPT cycle numbers.
            - list: Discharge capacities (from RPT experiments).
            - int: Number of ageing cycles (M1).
    """
    try:
        params = GROUPS[group_id]
    except KeyError:
        raise ValueError(f"Group '{group_id}' not found. Available groups: {list(GROUPS.keys())}")

    N = params['N']
    M1 = params['M1']
    M2 = params['M2']

    # Define experiments common to all groups.
    derated_cccv_experiment = pybamm.Experiment(
        [("Rest for 1 hour",
          "Charge at 0.6 A until 3.6 V",
          "Hold at 3.6 V until 50 mA",
          "Discharge at 0.6 A until 2.5 V",
          "Rest for 1 hour")] * N
    )

    charge_experiment = pybamm.Experiment(
        [("Charge at 0.6 A until 3.6V",
          "Hold at 3.6 V until 50 mA")]
    )

    rpt_experiment = pybamm.Experiment([("Discharge at 0.4 A until 2.5 V",)])

    # Define the CCCV experiment depending on the group.
    # For groups G13 and G15, use a two-step fast charging protocol.
    if group_id in ["G13", "G15"]:
        # Two-step fast charging:
        cccv_experiment = pybamm.Experiment(
            [(
                f"Charge at {params['charge_current']} A until {params['charge_cutoff']} V",
                "Charge at 1.2 A until 3.6V",  # Second charging stage (1C)
                "Hold at 3.6 V until 50 mA",
                f"Discharge at {params['discharge_current']} A until {params['discharge_cutoff']} V",
                "Rest for 1 hour",
            )] * N
        )
    else:
        # One-step fast charging:
        cccv_experiment = pybamm.Experiment(
            [(
                f"Charge at {params['charge_current']} A until 3.6 V",
                "Hold at 3.6 V until 50 mA",
                f"Discharge at {params['discharge_current']} A until {params['discharge_cutoff']} V",
                "Rest for 1 hour",
             )] * N
        )

    print(f"Running simulations for group {group_id}...")

    # Run initial cycle block: CCCV, then charge, then RPT.
    cccv_sol = run_experiment(model, cccv_experiment, parameter_values, VAR_PTS)
    charge_sol = run_experiment(model, charge_experiment, parameter_values, VAR_PTS, starting_solution=cccv_sol)
    rpt_sol = run_experiment(model, rpt_experiment, parameter_values, VAR_PTS, starting_solution=charge_sol)

    cccv_sols, charge_sols, rpt_sols = [cccv_sol], [charge_sol], [rpt_sol]

    # Run ageing cycles
    for i in range(1, M1):
        cccv_sol = run_experiment(model, cccv_experiment, parameter_values, VAR_PTS, starting_solution=rpt_sol)
        charge_sol = run_experiment(model, charge_experiment, parameter_values, VAR_PTS, starting_solution=cccv_sol)
        rpt_sol = run_experiment(model, rpt_experiment, parameter_values, VAR_PTS, starting_solution=charge_sol)
        cccv_sols.append(cccv_sol)
        charge_sols.append(charge_sol)
        rpt_sols.append(rpt_sol)

    # Run derated cycles
    for i in range(M2):
        cccv_sol = run_experiment(model, derated_cccv_experiment, parameter_values, VAR_PTS, starting_solution=rpt_sol)
        charge_sol = run_experiment(model, charge_experiment, parameter_values, VAR_PTS, starting_solution=cccv_sol)
        rpt_sol = run_experiment(model, rpt_experiment, parameter_values, VAR_PTS, starting_solution=charge_sol)
        cccv_sols.append(cccv_sol)
        charge_sols.append(charge_sol)
        rpt_sols.append(rpt_sol)

    # Compute RPT cycle numbers and discharge capacities.
    cccv_cycles, rpt_cycles, rpt_capacities = [], [], []

    # Ageing cycles:
    for i in range(M1):
        for j in range(N):
            cycle_index = i * (N + 2) + j
            cccv_cycles.append(cycle_index + 1)
            start_cap = rpt_sol.cycles[cycle_index].steps[2]["Discharge capacity [A.h]"].entries[0]
            end_cap = rpt_sol.cycles[cycle_index].steps[2]["Discharge capacity [A.h]"].entries[-1]
            rpt_capacities.append(end_cap - start_cap)
        rpt_cycles.append((i + 1) * (N + 2) / N)
    # Derated cycles:
    for i in range(M2):
        for j in range(N):
            cycle_index = (M1 + i) * (N + 1) + j
            cccv_cycles.append(cycle_index + 1)
            start_cap = rpt_sols[M1 + i].cycles[j].steps[2]["Discharge capacity [A.h]"].entries[0]
            end_cap = rpt_sols[M1 + i].cycles[j].steps[2]["Discharge capacity [A.h]"].entries[-1]
            rpt_capacities.append(end_cap - start_cap)
        rpt_cycles.append((M1 + i + 1) * (N + 1) / N)

    print(f"Results for group {group_id} obtained.")
    return cccv_sols, charge_sols, rpt_sols, rpt_cycles, rpt_capacities, M1


def plot_results(rpt_cycles, rpt_capacities, group_id):
    """
    Plot discharge capacity vs. RPT cycle number for the selected group.
    
    Args:
        rpt_cycles (list): RPT cycle numbers.
        rpt_capacities (list): Corresponding discharge capacities.
        group_id (str): Experimental group identifier.
    """
    plt.figure(figsize=(10, 8), dpi=200)
    plt.scatter(rpt_cycles, rpt_capacities, s=150, label=f"Group {group_id}")
    plt.xlabel("RPT number [-]", fontsize=16)
    plt.ylabel("Discharge capacity [A.h]", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.title(f"Discharge Capacity vs. RPT Number (Group {group_id})", fontsize=18)
    plt.show()


# —————————————————————————————
# FEATURE EXTRACTION FUNCTIONS
# —————————————————————————————
def extract_features(group_id, cccv_sols, charge_sols, rpt_sols, rpt_capacities, M1):
    """
    Extract additional performance metrics from the simulation results and
    create a Pandas DataFrame.
    
    Metrics extracted include:
      - RPT discharge capacity
      - Charge throughput (Ah)
      - Time under load (in hours)
      - Total elapsed time from start (in hours)
      - Durations (in hours) below 3A, between 3A and 4A, and above 4A
      - Capacity drop between consecutive RPT cycles
    
    Args:
        group_id (str): Group identifier.
        cccv_sols (list): List of CCCV simulation solutions.
        charge_sols (list): List of charge simulation solutions.
        rpt_sols (list): List of RPT simulation solutions.
        rpt_capacities (list): List of discharge capacities from RPT experiments.
        M1 (int): Number of ageing cycles.
    
    Returns:
        pd.DataFrame: DataFrame containing all extracted features.
    """
    rpt_discharge_capacities = []
    charge_throughputs = []
    time_under_loads = []
    time_elapsed_from_start = []
    group_numbers = []
    capacity_drops = []
    duration_below_3A = []
    duration_between_3A_and_4A = []
    duration_above_4A = []

    total_time_elapsed = 0

    # For current-based durations
    last_below_3A = 0
    last_between_3A_and_4A = 0
    last_above_4A = 0

    # Loop over each RPT cycle
    for idx in range(len(rpt_sols)):
        discharge_capacity = rpt_capacities[idx]
        rpt_discharge_capacities.append(discharge_capacity)
        group_numbers.append(group_id)

        # Calculate time under load for each block
        cccv_time = cccv_sols[idx]["Time [s]"].entries[-1] - cccv_sols[idx]["Time [s]"].entries[0]
        charge_time = charge_sols[idx]["Time [s]"].entries[-1] - charge_sols[idx]["Time [s]"].entries[0]
        rpt_time = rpt_sols[idx]["Time [s]"].entries[-1] - rpt_sols[idx]["Time [s]"].entries[0]
        time_under_load = cccv_time + charge_time + rpt_time
        time_under_loads.append(time_under_load / 3600)  # seconds to hours

        total_time_elapsed += time_under_load
        time_elapsed_from_start.append(total_time_elapsed / 3600)

        # Charge throughput: for first M1 cycles use CCCV data, otherwise charge data.
        if idx < M1:
            current_vals = cccv_sols[idx]["Current [A]"].entries
            time_vals = cccv_sols[idx]["Time [s]"].entries
        else:
            current_vals = charge_sols[idx]["Current [A]"].entries
            time_vals = charge_sols[idx]["Time [s]"].entries
        charge_throughput = abs(sum(
            i * (t2 - t1) for i, t1, t2 in zip(current_vals, time_vals[:-1], time_vals[1:])
        ) / 3600)
        charge_throughputs.append(charge_throughput)

        # Duration at different current levels (using CCCV data)
        abs_currents = [abs(i) for i in cccv_sols[idx]["Current [A]"].entries]
        time_intervals = [t2 - t1 for t1, t2 in zip(cccv_sols[idx]["Time [s]"].entries[:-1],
                                                     cccv_sols[idx]["Time [s]"].entries[1:])]
        below_3A = sum(interval for i, interval in zip(abs_currents[:-1], time_intervals) if i < 3)
        between_3A_and_4A = sum(interval for i, interval in zip(abs_currents[:-1], time_intervals) if 3 <= i < 4)
        above_4A = sum(interval for i, interval in zip(abs_currents[:-1], time_intervals) if i >= 4)

        if below_3A != last_below_3A:
            last_below_3A = below_3A
        else:
            below_3A = 0

        if between_3A_and_4A != last_between_3A_and_4A:
            last_between_3A_and_4A = between_3A_and_4A
        else:
            between_3A_and_4A = 0

        if above_4A != last_above_4A:
            last_above_4A = above_4A
        else:
            above_4A = 0

        duration_below_3A.append(below_3A / 3600)
        duration_between_3A_and_4A.append(between_3A_and_4A / 3600)
        duration_above_4A.append(above_4A / 3600)

        if idx == 0:
            capacity_drops.append(0)
        else:
            capacity_drop = rpt_capacities[idx - 1] - discharge_capacity
            capacity_drops.append(capacity_drop)

    df = pd.DataFrame({
        'RPT Number': list(range(1, len(rpt_discharge_capacities) + 1)),
        'Capacity (Ah)': rpt_discharge_capacities,
        'Ampere-Hour Throughput (Ah)': charge_throughputs,
        'Time Under Load (h)': time_under_loads,
        'Total Time Elapsed (h)': time_elapsed_from_start,
        'Group': group_numbers,
        'Capacity Drop (Ah)': capacity_drops,
        'Time Below 3A (h)': duration_below_3A,
        'Time Between 3A and 4A (h)': duration_between_3A_and_4A,
        'Time Above 4A (h)': duration_above_4A
    })
    return df


# —————————————————————————————
# MAIN EXECUTION
# —————————————————————————————
def run_group_by_id(group_id):
    """
    Execute the simulation, feature extraction, and plotting for the specified group.
    
    Args:
        group_id (str): The experimental group identifier (e.g., "G1", "G2", "G13", "G15", etc.).
    """
    # Initialize model and parameters
    model = pybamm.lithium_ion.DFN(MODEL_OPTIONS)
    parameter_values = get_parameter_values()

    # Run experiments for the selected group
    cccv_sols, charge_sols, rpt_sols, rpt_cycles, rpt_capacities, M1 = run_group_experiment(
        group_id, model, parameter_values
    )

    # Plot the simulation result (discharge capacity vs. RPT number)
    plot_results(rpt_cycles, rpt_capacities, group_id)

    # Extract performance features and create a DataFrame
    df = extract_features(group_id, cccv_sols, charge_sols, rpt_sols, rpt_capacities, M1)


    # Save the DataFrame to a pickle file 
    output_folder = "simulated_PBM_data"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filename = os.path.join(output_folder, f"{group_id}_PBM_Simulated.pkl")
    df.to_pickle(output_filename)
    print(f"Feature data saved to {output_filename}")
    return df


if __name__ == '__main__':
    group_id = input(f"Enter the group ID {list(GROUPS.keys())}: ").strip()
    features_df = run_group_by_id(group_id)
    # Optionally print the first few rows of the dataframe.
    print(features_df.head())