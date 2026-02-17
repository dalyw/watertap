import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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


def plot_ternary(csv_file="sensitivity_4.csv"):
    """Plot ternary diagram for 3D parameter sweep.

    Expects three sweep parameters:
    - NH4_removal_fraction (Recovery %)
    - NH4_energy_intensity (Energy Intensity [kWh/kg])
    - ammonia_recovery_value (Recovered Product Value [$/kg N])
    """
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.lstrip("#").str.strip()

    # Identify sweep parameters
    output_patterns = [
        "LCOW",
        "Capital",
        "Electricity",
        "S_PO4 Concentration",
        "S_NH4 Concentration",
        "NH4_removal",
        "P_removal",
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

    if len(sweep_params) != 3:
        print(
            f"Expected 3 sweep parameters for ternary plot, found {len(sweep_params)}: {sweep_params}"
        )
        return

    # Map parameters to ternary axes
    # Top vertex: Energy Intensity
    # Left vertex: Recovery % (NH4_removal_fraction * 100)
    # Right vertex: Recovered Product Value
    param_map = {}
    for param in sweep_params:
        if "energy_intensity" in param.lower():
            param_map["energy"] = param
        elif "removal" in param.lower() and "nh4" in param.lower():
            param_map["recovery"] = param
        elif "recovery_value" in param.lower() or "ammonia_recovery" in param.lower():
            param_map["value"] = param

    if len(param_map) != 3:
        print(f"Could not map parameters to ternary axes. Found: {param_map}")
        print(f"Available parameters: {sweep_params}")
        return

    # Normalize each variable to [0, 1] range for ternary coordinates
    df_norm = df.copy()
    df_norm["energy_norm"] = (
        df[param_map["energy"]] - df[param_map["energy"]].min()
    ) / (df[param_map["energy"]].max() - df[param_map["energy"]].min() + 1e-10)

    # Convert NH4 removal fraction to percentage for recovery axis
    recovery_pct = df[param_map["recovery"]] * 100
    df_norm["recovery_norm"] = (recovery_pct - recovery_pct.min()) / (
        recovery_pct.max() - recovery_pct.min() + 1e-10
    )

    df_norm["value_norm"] = (df[param_map["value"]] - df[param_map["value"]].min()) / (
        df[param_map["value"]].max() - df[param_map["value"]].min() + 1e-10
    )

    # Convert to barycentric coordinates (they should sum to 1)
    total = df_norm["energy_norm"] + df_norm["recovery_norm"] + df_norm["value_norm"]
    df_norm["energy_bary"] = df_norm["energy_norm"] / (total + 1e-10)
    df_norm["recovery_bary"] = df_norm["recovery_norm"] / (total + 1e-10)
    df_norm["value_bary"] = df_norm["value_norm"] / (total + 1e-10)

    # Convert barycentric to Cartesian for ternary plot
    # Top vertex (Energy): (0.5, sqrt(3)/2)
    # Left vertex (Recovery): (0, 0)
    # Right vertex (Value): (1, 0)
    # Barycentric: (energy_bary, recovery_bary, value_bary) where sum = 1
    sqrt3 = np.sqrt(3)
    x = (
        df_norm["energy_bary"] * 0.5
        + df_norm["recovery_bary"] * 0.0
        + df_norm["value_bary"] * 1.0
    )
    y = (
        df_norm["energy_bary"] * (sqrt3 / 2)
        + df_norm["recovery_bary"] * 0.0
        + df_norm["value_bary"] * 0.0
    )

    # Create ternary plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw ternary triangle
    ax.plot([0, 0.5, 1, 0], [0, sqrt3 / 2, 0, 0], "k-", linewidth=2)

    # Add vertex labels
    ax.text(
        0,
        -0.1,
        f"Recovery [%]\n({recovery_pct.min():.1f}-{recovery_pct.max():.1f})",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.5,
        sqrt3 / 2 + 0.1,
        f"Energy Intensity [kWh/kg]\n({df[param_map['energy']].min():.2f}-{df[param_map['energy']].max():.2f})",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        1,
        -0.1,
        f"Recovered Product Value [$/kg N]\n({df[param_map['value']].min():.2f}-{df[param_map['value']].max():.2f})",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    # Get output column (LCOW)
    output_col = None
    if "LCOW" in df.columns:
        output_col = "LCOW"
    else:
        # Find first output column
        for col in df.columns:
            if col == "LCOW" or "Capital" in col or "Electricity" in col:
                output_col = col
                break

    if output_col:
        # Color points by output value
        scatter = ax.scatter(
            x,
            y,
            c=df[output_col],
            cmap="viridis",
            s=100,
            alpha=0.7,
            edgecolors="k",
            linewidth=0.5,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(output_col, fontsize=12, fontweight="bold")
    else:
        ax.scatter(x, y, s=100, alpha=0.7, edgecolors="k", linewidth=0.5)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, sqrt3 / 2 + 0.2)

    plt.tight_layout()
    plt.savefig(csv_file.replace(".csv", "_ternary.png"), dpi=300, bbox_inches="tight")
    print(f"Saved ternary plot to {csv_file.replace('.csv', '_ternary.png')}")
    plt.show()


if __name__ == "__main__":
    # Get filename from command line argument or use default
    csv_file = "genericnp_sensitivity_1_NH4_removal_fraction.csv"
    # csv_file = "genericnp_sensitivity_2_NH4_removal_fraction_NH4_energy_intensity.csv"
    csv_file = "genericnp_sensitivity_3_NH4_removal_fraction_P_removal_fraction.csv"
    csv_file = "genericnp_sensitivity_4_NH4_removal_fraction_NH4_energy_intensity_ammonia_recovery_value.csv"
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
    elif len(sweep_params) == 3:
        print(f"\n3D sweep detected. Creating ternary diagram...")
        plot_ternary(csv_file)
    else:
        print(f"Unsupported number of sweep parameters: {len(sweep_params)}")
        print(f"Found sweep parameters: {sweep_params}")
