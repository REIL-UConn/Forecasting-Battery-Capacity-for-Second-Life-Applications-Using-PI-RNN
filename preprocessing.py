"""
preprocessing.py

Preprocessing Pipeline + Capacity Fade Plot for Batch 1 or Batch 2
------------------------------------------------------------------

This script processes cycling‑subfolder data and merges it with RPT capacity data
for either Batch 1 or Batch 2, then produces a capacity‑fade plot.

Usage:
    python preprocess_and_plot.py
    → Enter batch number (1 or 2) when prompted.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# =============================================================================
# CHANNEL MAPPINGS FOR EACH BATCH
# =============================================================================

CHANNEL_MAPPINGS = {
    "1": {
        ('G3', 'C1'): '6-1', ('G3', 'C2'): '6-2', ('G3', 'C3'): '6-3',
        ('G5', 'C1'): '6-4', ('G5', 'C2'): '6-5', ('G5', 'C3'): '6-6',
        ('G6', 'C1'): '6-7', ('G6', 'C2'): '6-8', ('G6', 'C3'): '7-1',
        ('G7', 'C1'): '7-2', ('G7', 'C2'): '7-3', ('G7', 'C3'): '7-4',
        ('G8', 'C1'): '7-5', ('G8', 'C2'): '7-6', ('G8', 'C3'): '7-7',
        ('G12','C1'): '7-8', ('G12','C2'): '8-1', ('G12','C3'): '8-2',
        ('G13','C1'): '8-3', ('G13','C2'): '8-4', ('G13','C3'): '8-5',
        ('G15','C1'): '8-6', ('G15','C2'): '8-7', ('G15','C3'): '8-8',
    },
    "2": {
        ('G1', 'C1'): '3-1', ('G1', 'C2'): '3-2', ('G1', 'C3'): '3-3',
        ('G2', 'C1'): '3-4', ('G2', 'C2'): '3-5', ('G2', 'C3'): '3-6',
        ('G4', 'C1'): '3-7', ('G4', 'C2'): '3-8', ('G4', 'C3'): '4-1',
        ('G11','C1'): '4-2', ('G11','C2'): '4-3', ('G11','C3'): '4-4',
        ('G14','C1'): '4-5', ('G14','C2'): '4-6', ('G14','C3'): '4-7',
        ('G16','C1'): '4-8', ('G16','C2'): '5-1', ('G16','C3'): '5-2',
        ('G17','C1'): '5-3', ('G17','C2'): '5-4', ('G17','C3'): '5-5',
        ('G18','C1'): '5-6', ('G18','C2'): '5-7', ('G18','C3'): '5-8',
    }
}


# =============================================================================
# COMMON CONFIGURATION
# =============================================================================

EXCEL_FILE      = "SOH estimation aging test management_update_10-15-2024.xlsx"
OUTPUT_DIR      = "Processed_data"
OUTPUT_TEMPLATE = "Processed_data_Cycling&RPT_Batch{batch}_Capacity_Forecasting_merged_update_Jan2025.pkl"



# =============================================================================
# UTILITIES
# =============================================================================

def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def calculate_ampere_hour_throughput(df):
    """
    Given a DataFrame with columns 'Date(h:min:s.ms)' and 'Current(A)', compute:
      - ampere‑hour throughput (Ah)
      - total time elapsed under load (h)
      - durations below 3A, between 3A and 4A, and above 4A (h)
    """
    df['timestamp'] = pd.to_datetime(df['Date(h:min:s.ms)'])
    df = df.sort_values('timestamp')
    df['dt_h'] = df['timestamp'].diff().dt.total_seconds().fillna(0) / 3600
    df['charge_Ah'] = df['Current(A)'].abs() * df['dt_h']
    throughput = df['charge_Ah'].sum()

    abs_i = df['Current(A)'].abs()
    t = df['dt_h']
    t_below_3 = t[abs_i < 3].sum()
    t_3_4     = t[(abs_i >= 3) & (abs_i < 4)].sum()
    t_above_4 = t[abs_i >= 4].sum()

    total_time = t_below_3 + t_3_4 + t_above_4
    return throughput, total_time, t_below_3, t_3_4, t_above_4


def process_cycling(base_dir, mapping):
    """
    Traverse all "Cycling n" subfolders under base_dir, process each Excel file,
    and return a DataFrame of cycling metrics.
    """
    records = []
    for (group, cell), channel in mapping.items():
        for sub in os.listdir(base_dir):
            subpath = os.path.join(base_dir, sub)
            if os.path.isdir(subpath) and sub.lower().startswith("cycling"):
                try:
                    rpt_num = int(sub.split()[-1])
                except ValueError:
                    continue
                for fname in os.listdir(subpath):
                    if fname.startswith(f"250012-{channel}") and fname.endswith(".xlsx"):
                        df = pd.read_excel(os.path.join(subpath, fname),
                                           sheet_name=3, engine='openpyxl')
                        throughput, total_t, t1, t2, t3 = calculate_ampere_hour_throughput(df)
                        records.append({
                            'Group': group,
                            'Cell': cell,
                            'Channel': channel,
                            'RPT Number': rpt_num,
                            'Ampere-Hour Throughput (Ah)': throughput,
                            'Total Time Elapsed (h)': total_t,
                            'Time Below 3A (h)': t1,
                            'Time Between 3A and 4A (h)': t2,
                            'Time Above 4A (h)': t3
                        })
                        break

    df = pd.DataFrame(records)
    df = df.sort_values(['Group','Cell','RPT Number']).reset_index(drop=True)
    df['Total Absolute Time From Start (h)'] = (
        df.groupby(['Group','Cell'])['Total Time Elapsed (h)'].cumsum()
    )
    return df


def read_rpt_data(excel_file, sheet, groups):
    """
    Read RPT capacity data from Excel, filter by groups, and return a tidy DataFrame.
    """
    wb = pd.read_excel(excel_file, sheet_name=sheet, engine='openpyxl')
    rpt_cols = [c for c in wb.columns if 'RPT' in c]
    wb = wb[wb['Group'].isin(groups)].copy()

    rows = []
    for _, row in wb.iterrows():
        for col in rpt_cols:
            val = pd.to_numeric(row[col], errors='coerce')
            if not np.isnan(val):
                rpt_num = int(''.join(filter(str.isdigit, col)))
                rows.append({
                    'Group': row['Group'],
                    'Cell':  row['Cell'],
                    'Channel': row['Channel'],
                    'RPT Number': rpt_num,
                    'Capacity (Ah)': val
                })
    return pd.DataFrame(rows)


def merge_and_save(cycle_df, rpt_df, batch):
    """
    Merge cycling and RPT data, save to pickle, and return the file path.
    """
    merged = pd.merge(
        cycle_df,
        rpt_df[['Group','Cell','RPT Number','Capacity (Ah)']],
        on=['Group','Cell','RPT Number'],
        how='left'
    )
    ensure_dir(OUTPUT_DIR)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_TEMPLATE.format(batch=batch))
    merged.to_pickle(out_path)
    print(f"Saved merged data to {out_path}")
    return out_path


# =============================================================================
# PLOTTING
# =============================================================================

def plot_capacity_fade(batch, merged_path):
    """
    Plot capacity fade (mean ± std) vs. RPT Number for each group in the batch.
    """
    df = pd.read_pickle(merged_path)

    # Exclude G12 for batch 1
    exclude = ['G12'] if batch == "1" else []
    groups = sorted([g for g in df['Group'].unique() if g not in exclude],
                    key=lambda x: int(x[1:]))

    palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728',
               '#9467bd','#8c564b','#e377c2','#7f7f7f']
    marker = 'o'
    plt.rcParams['font.family'] = 'Times New Roman'

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    for i, grp in enumerate(groups):
        sub = df[df['Group']==grp]
        means = sub.groupby('RPT Number')['Capacity (Ah)'].mean()
        stds  = sub.groupby('RPT Number')['Capacity (Ah)'].std()
        x = np.arange(len(means)) + i*0.1

        ax.plot(x, means, color=palette[i%len(palette)],
                linewidth=2, marker=marker, markersize=4, label=grp)
        ax.fill_between(x, means-stds, means+stds,
                        color=palette[i%len(palette)], alpha=0.1)

    ax.set_xlabel('RPT Number', fontsize=24)
    ax.set_ylabel('Capacity (Ah)', fontsize=24)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid(True)
    ax.legend(fontsize=13, loc='upper right', ncol=2)
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    batch = input("Enter batch number (1 or 2): ").strip()
    if batch not in CHANNEL_MAPPINGS:
        print("Invalid batch; please choose '1' or '2'.")
        return

    base_dir   = f"Batch {batch}"
    sheet_name = f"Batch {batch}"
    mapping    = CHANNEL_MAPPINGS[batch]
    groups     = sorted({grp for grp, _ in mapping.keys()})

    print(f"Processing cycling data in '{base_dir}' for groups: {groups}")
    cycle_df = process_cycling(base_dir, mapping)

    print(f"Reading RPT capacities from sheet '{sheet_name}'")
    rpt_df = read_rpt_data(EXCEL_FILE, sheet_name, groups)

    print("Merging and saving...")
    merged_path = merge_and_save(cycle_df, rpt_df, batch)

    print("Plotting capacity fade...")
    plot_capacity_fade(batch, merged_path)


if __name__ == "__main__":
    main()
