#!/usr/bin/env python
"""
Plot parameter sweep results from multi_sweep.py

Usage:
    python plot_sweep_results.py [filename.csv]
    
If no filename is provided, defaults to: genericnp_sensitivity_1_NH4_removal_fraction.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_1d_sweep(csv_file="sensitivity_1.csv"):
    """Plot results from 1D parameter sweep (one parameter varied) as line plots"""
    df = pd.read_csv(csv_file)
    # Strip '#' from column names if present (some CSVs have # in header)
    df.columns = df.columns.str.lstrip("#").str.strip()

    # Identify sweep parameters (same logic as main)
    output_patterns = [
        "LCOW",
        "Capital",
        "Electricity",
        "S_PO4 Concentration",
        "S_NH4 Concentration",
        "NH4_removal",  # Exact match only (not "NH4_removal_fraction")
        "P_removal",  # Exact match only (not "P_removal_fraction")
        "Solve Time",
        "Upgrade",
    ]

    sweep_params = []
    for col in df.columns:
        if df[col].nunique() > 1 and col not in ["solve_successful", "sweep_index"]:
            is_output = False
            for pattern in output_patterns:
                if pattern == "NH4_removal" or pattern == "P_removal":
                    if col == pattern:
                        is_output = True
                        break
                else:
                    if pattern in col:
                        is_output = True
                        break

            if not is_output:
                sweep_params.append(col)

    param = sweep_params[0]

    # Get output columns - only LCOW (not Upgrade LCOW), Capital Cost, and Electricity Cost
    outputs = []
    for col in df.columns:
        if col == "LCOW":  # Exact match to exclude "Upgrade LCOW"
            outputs.append(col)
        elif "Capital" in col:
            outputs.append(col)
        elif "Electricity" in col:
            outputs.append(col)

    # Sort by parameter value for clean line plots
    df = df.sort_values(by=param)

    # Create line plots in 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    axes = axes.flatten()

    for idx, output in enumerate(outputs):
        axes[idx].plot(df[param], df[output], marker="o", linewidth=2, markersize=8)
        axes[idx].set_title(f"{output}", fontsize=14, fontweight="bold")
        axes[idx].set_xlabel(param, fontsize=12)
        axes[idx].set_ylabel(output, fontsize=12)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(csv_file.replace(".csv", "_lineplot.png"), dpi=300, bbox_inches="tight")
    print(f"Saved line plot to {csv_file.replace('.csv', '_lineplot.png')}")
    plt.show()


def plot_2d_sweep(csv_file="sensitivity_2.csv"):
    """Plot results from 2D parameter sweep (two parameters varied) as heatmaps"""
    df = pd.read_csv(csv_file)
    # Strip '#' from column names if present (some CSVs have # in header)
    df.columns = df.columns.str.lstrip("#").str.strip()

    # Identify sweep parameters (same logic as main)
    output_patterns = [
        "LCOW",
        "Capital",
        "Electricity",
        "S_PO4 Concentration",
        "S_NH4 Concentration",
        "NH4_removal",  # Exact match only (not "NH4_removal_fraction")
        "P_removal",  # Exact match only (not "P_removal_fraction")
        "Solve Time",
        "Upgrade",
    ]

    sweep_params = []
    for col in df.columns:
        if df[col].nunique() > 1 and col not in ["solve_successful", "sweep_index"]:
            is_output = False
            for pattern in output_patterns:
                if pattern == "NH4_removal" or pattern == "P_removal":
                    if col == pattern:
                        is_output = True
                        break
                else:
                    if pattern in col:
                        is_output = True
                        break

            if not is_output:
                sweep_params.append(col)

    param1, param2 = sweep_params

    # Get output columns - only LCOW (not Upgrade LCOW), Capital Cost, and Electricity Cost
    outputs = []
    for col in df.columns:
        if col == "LCOW":  # Exact match to exclude "Upgrade LCOW"
            outputs.append(col)
        elif "Capital" in col:
            outputs.append(col)
        elif "Electricity" in col:
            outputs.append(col)

    # Create heatmaps in 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    axes = axes.flatten()

    for idx, output in enumerate(outputs):
        # Create heatmap from pivoted data
        pivot_data = df.pivot(index=param2, columns=param1, values=output)
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3g",
            cmap="viridis",
            ax=axes[idx],
            cbar_kws={"label": output},
        )
        axes[idx].set_title(f"{output}", fontsize=14, fontweight="bold")
        axes[idx].set_xlabel(param1, fontsize=12)
        axes[idx].set_ylabel(param2, fontsize=12)

    plt.tight_layout()
    plt.savefig(csv_file.replace(".csv", "_heatmap.png"), dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to {csv_file.replace('.csv', '_heatmap.png')}")
    plt.show()


if __name__ == "__main__":
    # Get filename from command line argument or use default
    csv_file = "genericnp_sensitivity_1_NH4_removal_fraction.csv"
    # csv_file = "genericnp_sensitivity_2_NH4_removal_fraction_NH4_energy_intensity.csv"
    csv_file = "genericnp_sensitivity_3_NH4_removal_fraction_P_removal_fraction.csv"

    print(f"Plotting results from: {csv_file}")

    # Read CSV and identify sweep parameters
    df = pd.read_csv(csv_file)
    # Strip '#' from column names if present (some CSVs have # in header)
    df.columns = df.columns.str.lstrip("#").str.strip()

    # Identify sweep parameters
    # Exclude known output columns (exact matches or specific patterns)
    output_patterns = [
        "LCOW",
        "Capital",
        "Electricity",
        "S_PO4 Concentration",
        "S_NH4 Concentration",
        "NH4_removal",  # Exact match only (not "NH4_removal_fraction")
        "P_removal",  # Exact match only (not "P_removal_fraction")
        "Solve Time",
        "Upgrade",
    ]

    sweep_params = []
    for col in df.columns:
        if df[col].nunique() > 1 and col not in ["solve_successful", "sweep_index"]:
            # Check if column matches any output pattern
            is_output = False
            for pattern in output_patterns:
                if pattern == "NH4_removal" or pattern == "P_removal":
                    # For these, check exact match only
                    if col == pattern:
                        is_output = True
                        break
                else:
                    # For others, check substring match
                    if pattern in col:
                        is_output = True
                        break

            if not is_output:
                sweep_params.append(col)

    # Determine number of sweep parameters and plot accordingly
    if len(sweep_params) == 1:
        print(f"\n1D sweep detected. Creating line plots...")
        plot_1d_sweep(csv_file)
    elif len(sweep_params) == 2:
        print(f"\n2D sweep detected. Creating heatmaps...")
        plot_2d_sweep(csv_file)
    else:
        print(f"Unsupported number of sweep parameters: {len(sweep_params)}")
        print(f"Found sweep parameters: {sweep_params}")
