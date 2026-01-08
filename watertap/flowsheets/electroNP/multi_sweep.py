#################################################################################
# WaterTAP Copyright (c) 2020-2026, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Laboratory of the Rockies, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################
import logging
import time

logging.getLogger("idaes.core.util.scaling").disabled = True
from parameter_sweep import LinearSample, ParameterSweep
import pyomo.environ as pyo
import watertap.flowsheets.electroNP.BSM2_genericNP_no_bioP as genericNP_flowsheet
from pyomo.environ import units as pyunits
from pyomo.opt import SolverResults


def build_and_initialize_model(**kwargs):
    """Construct flowsheet following standard BSM2 pattern: initialize flowsheet first, then add costing."""
    m = genericNP_flowsheet.build_flowsheet(has_genericNP=True, basis="mass")

    # Set default operating conditions
    genericNP_flowsheet.set_operating_conditions(m)

    # Deactivate pressure equality constraints before init
    for mx in m.fs.mixers:
        mx.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 3].deactivate()

    # Initialize flowsheet FIRST (standard BSM2 pattern)
    genericNP_flowsheet.initialize_system(m, has_genericNP=True)

    # Re-deactivate after init
    for mx in m.fs.mixers:
        mx.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 3].deactivate()

    # Add costing AFTER initialization (standard BSM2 pattern)
    genericNP_flowsheet.add_costing(m)
    m.fs.costing.initialize()

    # Solve once to establish a consistent starting point
    solver = genericNP_flowsheet.get_solver()
    solver.options["max_iter"] = 5000
    results = solver.solve(m, tee=False)
    pyo.assert_optimal_termination(results)

    # Create Params for manual outputs that need to be tracked (for multiprocessing)
    m.solve_time = pyo.Param(initialize=0.0, mutable=True)
    upgrade_lcow_value = genericNP_flowsheet.calculate_upgrade_lcow(m)
    m.upgrade_lcow = pyo.Param(initialize=upgrade_lcow_value, mutable=True)

    return m


def build_outputs(model, **kwargs):
    """Build output dictionary from model. Must always return valid Pyomo components.

    Note: upgrade_lcow and solve_time Params are created in build_and_initialize_model
    for multiprocessing compatibility.
    """
    outputs = {}

    # parameter_sweep needs Pyomo components for outputs
    outputs["LCOW"] = model.fs.costing.LCOW
    outputs["GenericNP Capital Cost"] = model.fs.genericNP.costing.capital_cost
    outputs["Electricity Cost"] = model.fs.costing.aggregate_flow_costs["electricity"]
    outputs["S_PO4 Concentration"] = model.fs.genericNP.treated.conc_mass_comp[
        0, "S_PO4"
    ]
    outputs["S_NH4 Concentration"] = model.fs.genericNP.treated.conc_mass_comp[
        0, "S_NH4"
    ]
    outputs["NH4_removal"] = model.fs.genericNP.removal_factors["S_NH4"]
    outputs["P_removal"] = model.fs.genericNP.removal_factors["S_PO4"]

    # Update upgrade_lcow value (Param was created in build_and_initialize_model)
    upgrade_lcow_value = genericNP_flowsheet.calculate_upgrade_lcow(model)
    model.upgrade_lcow.set_value(upgrade_lcow_value)
    outputs["Upgrade LCOW"] = model.upgrade_lcow

    # solve_time Param was created in build_and_initialize_model
    outputs["Solve Time (s)"] = model.solve_time

    return outputs


def build_sweep_params(model, case_num=1, nx=11, **kwargs):
    """Build sweep parameters that directly vary model variables."""
    sweep_params = {}

    # 1D: NH4 removal fraction sensitivity
    sweep_params["NH4_removal_fraction"] = LinearSample(
        model.fs.genericNP.removal_factors["S_NH4"], 0.1, 0.95, nx
    )
    if case_num == 2:
        sweep_params["NH4_energy_intensity"] = LinearSample(
            model.fs.genericNP.energy_electric_flow["S_NH4"], 0.04, 2.5, nx
        )
    elif case_num == 3:
        sweep_params["P_removal_fraction"] = LinearSample(
            model.fs.genericNP.removal_factors["S_PO4"], 0.1, 0.95, nx
        )

    return sweep_params


def optimize_function(m_inner):
    """Solve wrapper that handles failures gracefully and tracks solve time.

    Must be a module-level function (not nested) for multiprocessing pickle support.
    """
    try:
        # Create solver inside function for multiprocessing compatibility
        solver = genericNP_flowsheet.get_solver()
        solver.options["max_iter"] = 5000

        # Track solve time
        start_time = time.time()
        results = solver.solve(m_inner, tee=True)
        solve_time = time.time() - start_time

        # Store solve time in model as a Pyomo Param so it can be included in outputs
        if not hasattr(m_inner, "solve_time"):
            m_inner.solve_time = pyo.Param(initialize=solve_time, mutable=True)
        else:
            m_inner.solve_time.set_value(solve_time)

        # Check if solve was successful
        if not pyo.check_optimal_termination(results):
            print(f"Warning: Solver returned {results.solver.termination_condition}")
        return results
    except Exception as e:
        print(f"Solve failed with exception: {e}")
        # Store solve time even on failure
        solve_time = time.time() - start_time if "start_time" in locals() else 0.0
        if not hasattr(m_inner, "solve_time"):
            m_inner.solve_time = pyo.Param(initialize=solve_time, mutable=True)
        else:
            m_inner.solve_time.set_value(solve_time)
        # Return a failed result object
        results = SolverResults()
        results.solver.termination_condition = "error"
        return results


def reinitialize_function(m):
    """Reinitialize model before each sweep point.

    Must be a module-level function (not lambda) for multiprocessing pickle support.
    """
    genericNP_flowsheet.initialize_system(m, has_genericNP=True)


def run_analysis(case_num=1, nx=11, interpolate_nan_outputs=True):

    # Determine sweep parameter names from case_num for filename
    if case_num == 1:
        sweep_param_names = ["NH4_removal_fraction"]
    elif case_num == 2:
        sweep_param_names = ["NH4_removal_fraction", "NH4_energy_intensity"]
    elif case_num == 3:
        sweep_param_names = ["NH4_removal_fraction", "P_removal_fraction"]
    else:
        sweep_param_names = ["unknown"]

    # Generate filename with sweep parameter names as metadata
    param_suffix = "_".join(sweep_param_names)
    output_filename = f"genericnp_sensitivity_{case_num}_{param_suffix}.csv"

    ps = ParameterSweep(
        csv_results_file_name=output_filename,
        interpolate_nan_outputs=interpolate_nan_outputs,
        optimize_function=optimize_function,
        optimize_kwargs={},
        initialize_before_sweep=False,
        reinitialize_before_sweep=True,  # Reinitialize to avoid local minima
        reinitialize_function=reinitialize_function,
        number_of_subprocesses=8,
        parallel_back_end="concurrent.futures",
    )

    results_array, results_dict = ps.parameter_sweep(
        build_model=build_and_initialize_model,
        build_model_kwargs=dict(),
        build_sweep_params=build_sweep_params,
        build_sweep_params_kwargs=dict(case_num=case_num, nx=nx),
        build_outputs=build_outputs,
        num_samples=nx,
    )

    return results_array, results_dict, None


if __name__ == "__main__":
    case_num = 3
    nx = 8  # min 5 points for interpolation

    print(f"Running GenericNP sensitivity case {case_num} with nx={nx}")
    results_array, results_dict, _ = run_analysis(case_num=case_num, nx=nx)
    print(f"Sweep complete! Results saved")
