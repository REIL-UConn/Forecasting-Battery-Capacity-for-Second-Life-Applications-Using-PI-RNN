"""
PBM Experiments with Feature Extraction and Multiple Charging Protocols (Single Group)
-------------------------------------------------------------------------------------

Runs PyBaMM DFN experiments (CCCV, charge, RPT, and derated cycles) for a single group.
For G13/G15 uses a two-step fast charge; others use one-step.
Extracts features into a DataFrame and saves as a pickle.

Dependencies:
    pybamm, numpy, pandas, matplotlib, pickle
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
import numpy as np
import pybamm
import matplotlib.pyplot as plt
import pandas as pd


# ==============================
# Logging
# ==============================
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("pbm")


# ==============================
# Configuration
# ==============================
MODEL_OPTIONS = {
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "partially reversible",
    "lithium plating porosity change": "true",
    "particle mechanics": ("swelling and cracking", "swelling only"),
    "SEI on cracks": "true",
    "loss of active material": "stress-driven",
    "calculate discharge energy": "true",
}

VAR_PTS = {
    "x_n": 3,
    "x_s": 3,
    "x_p": 3,
    "r_n": 9,
    "r_p": 9,
}

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



def get_parameter_values() -> pybamm.ParameterValues:
    pv = pybamm.ParameterValues("OKane2022")
    _ = pybamm.ParameterValues("Prada2013")  # keep if you needed it for earlier param errors

    pv.update(
        {
            "Positive electrode OCP [V]": pybamm.input.parameters.lithium_ion.Prada2013.LFP_ocp_Afshar2017,
            "Lower voltage cut-off [V]": "2.5",
            "Upper voltage cut-off [V]": "3.6",
            "Open-circuit voltage at 0% SOC [V]": "2.5",
            "Open-circuit voltage at 100% SOC [V]": "3.6",
            "Nominal cell capacity [A.h]": "1.2",
            "Current function [A]": "1.2",
        }
    )

    pv.update(
        {
            "Maximum concentration in positive electrode [mol.m-3]": "31170.0",
            "Maximum concentration in negative electrode [mol.m-3]": "45850.0",
            "Initial concentration in positive electrode [mol.m-3]": "29440.0",
            "Initial concentration in negative electrode [mol.m-3]": "20100.0",
        }
    )

    pv.update(
        {
            "Separator thickness [m]": "1.9e-05",
            "Positive electrode thickness [m]": "5.8e-05",
            "Negative electrode thickness [m]": "3.6e-05",
            "Electrode height [m]": "0.055",
            "Electrode width [m]": "0.850",
        }
    )

    pv.update(
        {
            "Positive electrode active material volume fraction": "0.54",
            "Positive electrode charge transfer coefficient": "0.5",
            "Positive electrode conductivity [S.m-1]": "0.34",
            "Positive electrode diffusivity [m2.s-1]": "6e-15",
            "Positive electrode porosity": "0.426",
            "Positive particle radius [m]": "5.22e-06",
        }
    )

    pv.update(
        {
            "Reference temperature [K]": "298",
            "Separator porosity": "0.45",
            "Negative electrode active material volume fraction": 0.75,
            "Negative electrode charge transfer coefficient": 0.5,
            "Negative electrode conductivity [S.m-1]": 215.0,
            "Negative electrode diffusivity [m2.s-1]": 5e-14,
            "Negative electrode exchange-current density [A.m-2]": pybamm.input.parameters.lithium_ion.Prada2013.graphite_LGM50_electrolyte_exchange_current_density_Chen2020,
            "Negative electrode porosity": 0.36,
            "Negative particle radius [m]": "5.86e-06",
            "SEI reaction exchange current density [A.m-2]": "1.5e-06",
            "SEI kinetic rate constant [m.s-1]": "2e-12",
            "Lithium plating kinetic rate constant [m.s-1]": "1e-08",
        }
    )
    return pv


def run_experiment(
    sim_model,
    experiment: pybamm.Experiment,
    parameter_values: pybamm.ParameterValues,
    var_pts: dict,
    starting_solution=None,
    tag: str = "",
):
    t0 = time.time()
    sim = pybamm.Simulation(sim_model, experiment=experiment, parameter_values=parameter_values, var_pts=var_pts, output_variables=["Time [s]", "Current [A]", "Discharge capacity [A.h]"],)
    sol = sim.solve(starting_solution=starting_solution, calc_esoh=False)
    logger.info("Solved %-10s | wall=%.1fs", tag, time.time() - t0)
    return sol


def _extract_capacity_from_cycle(cycle) -> float:
    """
    Robustly get discharge capacity delta from a cycle.
    Old pickles sometimes have capacity at cycle-level, other times inside a step.
    """
    if "Discharge capacity [A.h]" in cycle:
        q = cycle["Discharge capacity [A.h]"].entries
        return float(q[-1] - q[0])

    # fallback: find a step that contains discharge capacity
    for step in reversed(getattr(cycle, "steps", [])):
        if "Discharge capacity [A.h]" in step:
            q = step["Discharge capacity [A.h]"].entries
            return float(q[-1] - q[0])

    raise KeyError("Discharge capacity [A.h] not found in cycle or steps.")


def run_group_experiment(group_id: str, model, parameter_values):
    if group_id not in GROUPS:
        raise ValueError(f"Group '{group_id}' not found. Available: {list(GROUPS.keys())}")

    p = GROUPS[group_id]
    N, M1, M2 = p["N"], p["M1"], p["M2"]

    derated_cccv_experiment = pybamm.Experiment(
        [
            (
                "Rest for 1 hour",
                "Charge at 0.6 A until 3.6 V",
                "Hold at 3.6 V until 50 mA",
                "Discharge at 0.6 A until 2.5 V",
                "Rest for 1 hour",
            )
        ]
        * N
    )

    charge_experiment = pybamm.Experiment([("Charge at 0.6 A until 3.6V", "Hold at 3.6 V until 50 mA")])
    rpt_experiment = pybamm.Experiment([("Discharge at 0.4 A until 2.5 V",)])

    if group_id in ["G13", "G15"]:
        cccv_experiment = pybamm.Experiment(
            [
                (
                    f"Charge at {p['charge_current']} A until {p['charge_cutoff']} V",
                    "Charge at 1.2 A until 3.6V",
                    "Hold at 3.6 V until 50 mA",
                    f"Discharge at {p['discharge_current']} A until {p['discharge_cutoff']} V",
                    "Rest for 1 hour",
                )
            ]
            * N
        )
    else:
        cccv_experiment = pybamm.Experiment(
            [
                (
                    f"Charge at {p['charge_current']} A until 3.6 V",
                    "Hold at 3.6 V until 50 mA",
                    f"Discharge at {p['discharge_current']} A until {p['discharge_cutoff']} V",
                    "Rest for 1 hour",
                )
            ]
            * N
        )

    logger.info("Running group=%s | N=%s M1=%s M2=%s", group_id, N, M1, M2)

    # Initial block
    cccv_sol = run_experiment(model, cccv_experiment, parameter_values, VAR_PTS, tag="CCCV(0)")
    charge_sol = run_experiment(model, charge_experiment, parameter_values, VAR_PTS, starting_solution=cccv_sol, tag="Charge(0)")
    rpt_sol = run_experiment(model, rpt_experiment, parameter_values, VAR_PTS, starting_solution=charge_sol, tag="RPT(0)")

    cccv_sols, charge_sols, rpt_sols = [cccv_sol], [charge_sol], [rpt_sol]

    # Ageing blocks
    for i in range(1, M1):
        cccv_sol = run_experiment(model, cccv_experiment, parameter_values, VAR_PTS, starting_solution=rpt_sol, tag=f"CCCV({i})")
        charge_sol = run_experiment(model, charge_experiment, parameter_values, VAR_PTS, starting_solution=cccv_sol, tag=f"Charge({i})")
        rpt_sol = run_experiment(model, rpt_experiment, parameter_values, VAR_PTS, starting_solution=charge_sol, tag=f"RPT({i})")
        cccv_sols.append(cccv_sol)
        charge_sols.append(charge_sol)
        rpt_sols.append(rpt_sol)

    # Derated blocks
    for i in range(M2):
        idx = M1 + i
        cccv_sol = run_experiment(model, derated_cccv_experiment, parameter_values, VAR_PTS, starting_solution=rpt_sol, tag=f"DeratedCCCV({idx})")
        charge_sol = run_experiment(model, charge_experiment, parameter_values, VAR_PTS, starting_solution=cccv_sol, tag=f"Charge({idx})")
        rpt_sol = run_experiment(model, rpt_experiment, parameter_values, VAR_PTS, starting_solution=charge_sol, tag=f"RPT({idx})")
        cccv_sols.append(cccv_sol)
        charge_sols.append(charge_sol)
        rpt_sols.append(rpt_sol)

    # Compute RPT cycle numbers and discharge capacities 
    cccv_cycles = []
    cccv_capacities = []
    rpt_cycles = []
    rpt_capacities = []

    # this assumes `rpt_sol` is the final cumulative solution 
    for i in range(M1):
        for j in range(N):
            cccv_cycles.append(i * (N + 2) + j + 1)

            start_capacity = (
                rpt_sol.cycles[i * (N + 2) + j]
                .steps[2]["Discharge capacity [A.h]"]
                .entries[0]
            )
            end_capacity = (
                rpt_sol.cycles[i * (N + 2) + j]
                .steps[2]["Discharge capacity [A.h]"]
                .entries[-1]
            )
            cccv_capacities.append(end_capacity - start_capacity)

        rpt_cycles.append((i + 1) * (N + 2) / N)

        start_capacity = rpt_sol.cycles[(i + 1) * (N + 2) - 1]["Discharge capacity [A.h]"].entries[0]
        end_capacity = rpt_sol.cycles[(i + 1) * (N + 2) - 1]["Discharge capacity [A.h]"].entries[-1]
        rpt_capacities.append(end_capacity - start_capacity)

    # Derated cycles: use the stored per-block RPT solutions 
    for i in range(M2):
        for j in range(N):
            cccv_cycles.append((M1 + i) * (N + 1) + j + 1)

            start_capacity = (
                rpt_sols[M1 + i].cycles[j]
                .steps[2]["Discharge capacity [A.h]"]
                .entries[0]
            )
            end_capacity = (
                rpt_sols[M1 + i].cycles[j]
                .steps[2]["Discharge capacity [A.h]"]
                .entries[-1]
            )
            cccv_capacities.append(end_capacity - start_capacity)

        rpt_cycles.append((M1 + i + 1) * (N + 1) / N)

        start_capacity = rpt_sols[M1 + i].cycles[-1]["Discharge capacity [A.h]"].entries[0]
        end_capacity = rpt_sols[M1 + i].cycles[-1]["Discharge capacity [A.h]"].entries[-1]
        rpt_capacities.append(end_capacity - start_capacity)


    logger.info("Computed rpt points: %d", len(rpt_capacities))
    return cccv_sols, charge_sols, rpt_sols, rpt_cycles, rpt_capacities, M1


def plot_results(rpt_cycles, rpt_capacities, group_id: str):
    # if len(rpt_cycles) != len(rpt_capacities):
    #     logger.warning("Plot length mismatch: len(x)=%d len(y)=%d", len(rpt_cycles), len(rpt_capacities))

    rpt_x = np.arange(1, len(rpt_capacities) + 1)

    plt.figure(figsize=(10, 8), dpi=200)
    plt.scatter(rpt_x, rpt_capacities, s=150, label=f"Group {group_id}")
    plt.xlabel("RPT number [-]", fontsize=16)
    plt.ylabel("Discharge capacity [A.h]", fontsize=16)
    plt.xticks(rpt_x)  # optional if you want every integer tick
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.title(f"Discharge Capacity vs. RPT Number (Group {group_id})", fontsize=18)
    plt.show()



def extract_features(group_id, cccv_sols, charge_sols, rpt_sols, rpt_capacities, M1) -> pd.DataFrame:
    rpt_discharge_capacities = []
    charge_throughputs = []
    time_under_loads = []
    time_elapsed_from_start = []
    group_numbers = []
    capacity_drops = []
    duration_below_3A = []
    duration_between_3A_and_4A = []
    duration_above_4A = []

    total_time_elapsed = 0.0
    last_below_3A = 0.0
    last_between_3A_and_4A = 0.0
    last_above_4A = 0.0

    for idx in range(len(rpt_sols)):
        discharge_capacity = rpt_capacities[idx]
        rpt_discharge_capacities.append(discharge_capacity)
        group_numbers.append(group_id)

        cccv_time = cccv_sols[idx]["Time [s]"].entries[-1] - cccv_sols[idx]["Time [s]"].entries[0]
        charge_time = charge_sols[idx]["Time [s]"].entries[-1] - charge_sols[idx]["Time [s]"].entries[0]
        rpt_time = rpt_sols[idx]["Time [s]"].entries[-1] - rpt_sols[idx]["Time [s]"].entries[0]
        time_under_load = float(cccv_time + charge_time + rpt_time)
        time_under_loads.append(time_under_load / 3600.0)

        total_time_elapsed += time_under_load
        time_elapsed_from_start.append(total_time_elapsed / 3600.0)

        if idx < M1:
            current_vals = cccv_sols[idx]["Current [A]"].entries
            time_vals = cccv_sols[idx]["Time [s]"].entries
        else:
            current_vals = charge_sols[idx]["Current [A]"].entries
            time_vals = charge_sols[idx]["Time [s]"].entries

        charge_throughput = abs(sum(i * (t2 - t1) for i, t1, t2 in zip(current_vals, time_vals[:-1], time_vals[1:])) / 3600.0)
        charge_throughputs.append(float(charge_throughput))

        abs_currents = [abs(i) for i in cccv_sols[idx]["Current [A]"].entries]
        time_intervals = [t2 - t1 for t1, t2 in zip(cccv_sols[idx]["Time [s]"].entries[:-1], cccv_sols[idx]["Time [s]"].entries[1:])]

        below_3A = sum(dt for i, dt in zip(abs_currents[:-1], time_intervals) if i < 3)
        between_3A_and_4A = sum(dt for i, dt in zip(abs_currents[:-1], time_intervals) if 3 <= i < 4)
        above_4A = sum(dt for i, dt in zip(abs_currents[:-1], time_intervals) if i >= 4)

        if below_3A == last_below_3A:
            below_3A = 0
        else:
            last_below_3A = below_3A

        if between_3A_and_4A == last_between_3A_and_4A:
            between_3A_and_4A = 0
        else:
            last_between_3A_and_4A = between_3A_and_4A

        if above_4A == last_above_4A:
            above_4A = 0
        else:
            last_above_4A = above_4A

        duration_below_3A.append(float(below_3A / 3600.0))
        duration_between_3A_and_4A.append(float(between_3A_and_4A / 3600.0))
        duration_above_4A.append(float(above_4A / 3600.0))

        if idx == 0:
            capacity_drops.append(0.0)
        else:
            capacity_drops.append(float(rpt_capacities[idx - 1] - discharge_capacity))

    return pd.DataFrame(
        {
            "RPT Number": list(range(1, len(rpt_discharge_capacities) + 1)),
            "Capacity (Ah)": rpt_discharge_capacities,
            "Ampere-Hour Throughput (Ah)": charge_throughputs,
            "Time Under Load (h)": time_under_loads,
            "Total Time Elapsed (h)": time_elapsed_from_start,
            "Group": group_numbers,
            "Capacity Drop (Ah)": capacity_drops,
            "Time Below 3A (h)": duration_below_3A,
            "Time Between 3A and 4A (h)": duration_between_3A_and_4A,
            "Time Above 4A (h)": duration_above_4A,
        }
    )


def run_group_by_id(group_id: str) -> pd.DataFrame:
    model = pybamm.lithium_ion.DFN(MODEL_OPTIONS)
    parameter_values = get_parameter_values()

    cccv_sols, charge_sols, rpt_sols, rpt_cycles, rpt_capacities, M1 = run_group_experiment(group_id, model, parameter_values)

    plot_results(rpt_cycles, rpt_capacities, group_id)

    df = extract_features(group_id, cccv_sols, charge_sols, rpt_sols, rpt_capacities, M1)

    out_dir = Path("CaseStudy2_Saved_Data")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_filename = out_dir / f"par_reversible_plating_{group_id}_features.pkl"
    df.to_pickle(output_filename)
    logger.info("Saved features: %s", output_filename)

    return df


if __name__ == "__main__":
    group_id = input(f"Enter the group ID {list(GROUPS.keys())}: ").strip()
    df = run_group_by_id(group_id)
    print(df.head())
