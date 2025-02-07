import pandas as pd
import glob
import numpy as np
from sklearn.metrics import r2_score
import json
from base_values import *

stream_tables = {}
constituents = [
    "S_A",
    "S_F",
    "S_I",
    "S_N2",
    "S_NH4",
    "S_NO3",
    "S_O2",
    "S_PO4",
    "S_K",
    "S_Mg",
    "S_IC",
    "X_AUT",
    "X_H",
    "X_I",
    "X_PAO",
    "X_PHA",
    "X_PP",
    "X_S",
]
for file in glob.glob(
    "initialization_training/BSM2_electroNH4_stream_table_N*.csv"
):  # import all files
    df = pd.read_csv(file)
    df.iloc[:, 0] = df.iloc[:, 0].str.replace("Mass Concentration ", "")
    df.rename(columns={df.columns[0]: "Constituent"}, inplace=True)

    for col in df.columns:
        if col not in ["Constituent", "Units"]:
            df[col] = df[col].replace("-", "0").astype(float)  # convert to float

    parts = file.replace(".csv", "").split("_")
    NH4_removal = float(next(p[1:] for p in parts if p.startswith("N")))
    P_removal = float(next(p[1:] for p in parts if p.startswith("P")))
    EI = float(next(p[1:] for p in parts if p.startswith("E")))
    df["NH4_removal"] = NH4_removal
    df["P_removal"] = P_removal
    df["EI"] = EI
    stream_tables[file.replace(".csv", "")] = df

data_list_R3 = []
data_list_translator = []
for (
    df
) in stream_tables.values():  # for each stream table, get the input parameter values
    params = {
        "NH4_removal": df["NH4_removal"].iloc[0],
        "P_removal": df["P_removal"].iloc[0],
        "EI": df["EI"].iloc[0],
    }

    row_R3 = params.copy()  # row for R3 value
    row_translator = params.copy()  # row for translator values
    for constituent in constituents:
        R3_value = df.loc[df["Constituent"] == constituent, "MX2__R3"].values[0]
        translator_value = df.loc[
            df["Constituent"] == constituent, "MX4__translator_asm2d_adm1"
        ].values[0]
        row_R3[constituent] = R3_value
        row_translator[constituent] = translator_value
    data_list_R3.append(row_R3)
    data_list_translator.append(row_translator)

df_R3 = pd.DataFrame(data_list_R3)
df_translator = pd.DataFrame(data_list_translator)

sensitivity_dfs = []
for file in glob.glob("sensitivity_*.csv"):
    df = pd.read_csv(file)
    # Drop rows where all values are NaN
    df = df.dropna(how="all")
    sensitivity_dfs.append(df)

# Combine all sensitivity dataframes
sensitivity_df = pd.concat(sensitivity_dfs, ignore_index=True)

# Create dataframes for sensitivity data
sensitivity_R3_data = {
    "NH4_removal": sensitivity_df["NH4_removal"],
    "P_removal": sensitivity_df["P_removal"],
    "EI": sensitivity_df["EI"],
}

sensitivity_translator_data = sensitivity_R3_data.copy()

# Add constituent columns for R3/MX2
for constituent in constituents:
    mx2_col = f"MX2_{constituent}"
    if mx2_col in sensitivity_df.columns:
        sensitivity_R3_data[constituent] = sensitivity_df[mx2_col]

# Add constituent columns for translator/MX4
for constituent in constituents:
    mx4_col = f"MX4_{constituent}"
    if mx4_col in sensitivity_df.columns:
        sensitivity_translator_data[constituent] = sensitivity_df[mx4_col]

# Convert to dataframes
sensitivity_R3_df = pd.DataFrame(sensitivity_R3_data)
sensitivity_translator_df = pd.DataFrame(sensitivity_translator_data)

# Drop rows with NaN values
sensitivity_R3_df = sensitivity_R3_df.dropna()
sensitivity_translator_df = sensitivity_translator_df.dropna()

# Append sensitivity data to existing dataframes
print(len(df_R3))
df_R3 = pd.concat([df_R3, sensitivity_R3_df], ignore_index=True)
print(len(df_R3))
print(len(df_translator))
df_translator = pd.concat([df_translator, sensitivity_translator_df], ignore_index=True)
print(len(df_translator))


def add_estimates(df, base_values):
    parameters = [
        "S_A",
        "S_F",
        "S_I",
        "S_N2",
        "S_NH4",
        "S_NO3",
        "S_O2",
        "S_PO4",
        "S_K",
        "S_Mg",
        "S_IC",
        "X_AUT",
        "X_H",
        "X_I",
        "X_PAO",
        "X_PHA",
        "X_PP",
        "X_S",
    ]
    equations_dict = {}

    for param in parameters:
        # Calculate deviations from base values
        scale = 1000 if param == "S_NO3" else 1
        y = (df[param].values * scale) - (base_values[param] * scale)

        # Create design matrices
        X_2var = df[["NH4_removal", "P_removal"]].values
        X_3var = df[["NH4_removal", "P_removal", "EI"]].values

        # Add quadratic terms
        X_quad_2var = np.column_stack([X_2var, X_2var[:, 0] ** 2, X_2var[:, 1] ** 2])
        X_quad_3var = np.column_stack(
            [X_3var, X_3var[:, 0] ** 2, X_3var[:, 1] ** 2, X_3var[:, 2] ** 2]
        )

        # Regularization
        reg_lambda = 1e-5
        models = {
            "2var_linear": (
                X_2var,
                X_2var.T @ X_2var + reg_lambda * np.eye(2),
                X_2var.T @ y,
            ),
            "2var_quad": (
                X_quad_2var,
                X_quad_2var.T @ X_quad_2var + reg_lambda * np.eye(X_quad_2var.shape[1]),
                X_quad_2var.T @ y,
            ),
            "3var_linear": (
                X_3var,
                X_3var.T @ X_3var + reg_lambda * np.eye(3),
                X_3var.T @ y,
            ),
            "3var_quad": (
                X_quad_3var,
                X_quad_3var.T @ X_quad_3var + reg_lambda * np.eye(X_quad_3var.shape[1]),
                X_quad_3var.T @ y,
            ),
        }

        try:
            # Fit all models and calculate R2 scores
            results = {}
            for model_name, (X, X_reg, y_reg) in models.items():
                z = np.linalg.solve(X_reg, y_reg)
                y_pred = X @ z
                if scale != 1:
                    y_pred = y_pred / scale
                y_pred += base_values[param]
                r2 = r2_score(df[param], y_pred)
                results[model_name] = (y_pred, z, r2)

            # Compare models with preference order, requiring significant improvement
            r2_2var_lin = results["2var_linear"][2]
            r2_2var_quad = results["2var_quad"][2]
            r2_3var_lin = results["3var_linear"][2]
            r2_3var_quad = results["3var_quad"][2]

            improvement_threshold = 0.02  # 2% improvement required
            min_r2_threshold = 0.8  # Minimum R2 score required

            best_r2 = max(r2_2var_lin, r2_2var_quad, r2_3var_lin, r2_3var_quad)

            if best_r2 >= min_r2_threshold:
                best_model = ("2var_linear", results["2var_linear"])
                if r2_2var_quad > (r2_2var_lin + improvement_threshold):
                    best_model = ("2var_quad", results["2var_quad"])
                if r2_3var_lin > (best_model[1][2] + improvement_threshold):
                    best_model = ("3var_linear", results["3var_linear"])
                if r2_3var_quad > (best_model[1][2] + improvement_threshold):
                    best_model = ("3var_quad", results["3var_quad"])

                # Drop rows with poor fits
                y_pred = best_model[1][0]
                residuals = np.abs(df[param] - y_pred)
                residual_threshold = np.std(residuals) * 2  # 2 standard deviations
                good_fit_mask = residuals <= residual_threshold

                if sum(good_fit_mask) > len(df) * 0.7:  # Keep at least 70% of data
                    df = df[good_fit_mask].copy()
                    # Refit with cleaned data
                    return add_estimates(df, base_values)

                df[f"{param}_est"] = np.maximum(0, y_pred)
                equations_dict[param] = (
                    best_model[0],
                    best_model[1][1],
                    best_model[1][2],
                )
                print(f"{param}: Using {best_model[0]} (R2={best_model[1][2]:.4f})")
            else:
                df[f"{param}_est"] = df[param]
                equations_dict[param] = ("original", None, None)
                print(f"{param}: Using original values (best R2={best_r2:.4f})")

        except np.linalg.LinAlgError:
            df[f"{param}_est"] = df[param]
            equations_dict[param] = ("original", None, None)
            print(f"Could not calculate fits for {param}")

    return df, equations_dict


# Add estimates and get equations
print("Fitting models for R3:")
df_R3_with_est, equations_R3 = add_estimates(df_R3.copy(), base_values_R3)
print("\nFitting models for translator:")
df_translator_with_est, equations_translator = add_estimates(
    df_translator.copy(), base_values_translator
)


# Convert equations to coefficient dictionaries
def get_coeffs_dict(equations, base_values):
    coeffs = {}
    for param, (model_type, coeffs_arr, r2) in equations.items():
        if model_type == "2var_linear":
            scale = 1 / 1000 if param == "S_NO3" else 1
            coeffs[param] = {
                "NH4_removal": coeffs_arr[0] * scale,
                "P_removal": coeffs_arr[1] * scale,
                "NH4_removal^2": 0,
                "P_removal^2": 0,
                "EI": 0,
                "EI^2": 0,
                "base": base_values[param],
            }
        elif model_type == "2var_quad":
            scale = 1 / 1000 if param == "S_NO3" else 1
            coeffs[param] = {
                "NH4_removal": coeffs_arr[0] * scale,
                "P_removal": coeffs_arr[1] * scale,
                "NH4_removal^2": coeffs_arr[2] * scale,
                "P_removal^2": coeffs_arr[3] * scale,
                "EI": 0,
                "EI^2": 0,
                "base": base_values[param],
            }
        elif model_type == "3var_linear":
            scale = 1 / 1000 if param == "S_NO3" else 1
            coeffs[param] = {
                "NH4_removal": coeffs_arr[0] * scale,
                "P_removal": coeffs_arr[1] * scale,
                "EI": coeffs_arr[2] * scale,
                "NH4_removal^2": 0,
                "P_removal^2": 0,
                "EI^2": 0,
                "base": base_values[param],
            }
        elif model_type == "3var_quad":
            scale = 1 / 1000 if param == "S_NO3" else 1
            coeffs[param] = {
                "NH4_removal": coeffs_arr[0] * scale,
                "P_removal": coeffs_arr[1] * scale,
                "EI": coeffs_arr[2] * scale,
                "NH4_removal^2": coeffs_arr[3] * scale,
                "P_removal^2": coeffs_arr[4] * scale,
                "EI^2": coeffs_arr[5] * scale,
                "base": base_values[param],
            }
        else:
            coeffs[param] = {
                k: 0
                for k in [
                    "NH4_removal",
                    "P_removal",
                    "EI",
                    "NH4_removal^2",
                    "P_removal^2",
                    "EI^2",
                ]
            }
    return coeffs


R3_coeffs = get_coeffs_dict(equations_R3, base_values_R3)
translator_coeffs = get_coeffs_dict(equations_translator, base_values_translator)

with open("R3_coeffs.json", "w") as f:
    json.dump(R3_coeffs, f)
with open("translator_coeffs.json", "w") as f:
    json.dump(translator_coeffs, f)

df_R3_with_est.to_csv("df_R3_with_est.csv", index=False)
df_translator_with_est.to_csv("df_translator_with_est.csv", index=False)
