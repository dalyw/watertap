import pyomo.environ as pyo
import json, sys, time, warnings, logging, random, csv
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import groupby

warnings.filterwarnings(
    "ignore", category=UserWarning, module="idaes.core.util.scaling"
)


# Suppress INFO level logging from IDAES initialization
# and WARNING level messages from idaes.core.util.scaling
class IDAESInfoFilter(logging.Filter):
    """Filter to suppress INFO level messages from IDAES loggers and scaling warnings."""

    def filter(self, record):
        if record.name.startswith("idaes") and record.levelno == logging.INFO:
            return False
        if (
            record.name == "idaes.core.util.scaling"
            and record.levelno == logging.WARNING
        ):
            return False
        return True


if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

watertap_path = Path(__file__).parent.parent.parent / "watertap"
if str(watertap_path) not in sys.path:
    print("Adding watertap path")
    sys.path.insert(0, str(watertap_path))

from watertap.flowsheets.electroNP import BSM2_genericNP_no_bioP as bsm2_genericNP
from watertap.flowsheets.electroNP.BSM2_genericNP_no_bioP import calculate_upgrade_lcow
from watertap.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from watertap.flowsheets.electroNP.standard_inputs import (
    add_effluent_tn_constraint,
    get_effluent_tn_kg_m3,
    transition_influent,
)

# logging suppression
for logger_name in ["idaes", "idaes.init", "idaes.core.util.scaling"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    for handler in logger.handlers:
        handler.setLevel(logging.ERROR)
        handler.addFilter(IDAESInfoFilter())

# Also check root logger handlers again
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if not any(isinstance(f, IDAESInfoFilter) for f in handler.filters):
        handler.addFilter(IDAESInfoFilter())

_idaes_filter = IDAESInfoFilter()
for _name in ["idaes", "idaes.init", "idaes.core.util.scaling"]:
    _logger = logging.getLogger(_name)
    _logger.setLevel(logging.ERROR)
    _logger.addFilter(_idaes_filter)
# Also apply to root logger handlers to catch dynamically-created child loggers
for _handler in logging.getLogger().handlers:
    _handler.addFilter(_idaes_filter)

WWTP_UNITS = ["fs.R1", "fs.R2", "fs.R3", "fs.R4", "fs.R5", "fs.CL", "fs.CL2"]


def _suppress_idaes_logging():
    """Re-apply IDAES log filter to all current handlers (catches dynamically-added ones)."""
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, IDAESInfoFilter) for f in handler.filters):
            handler.addFilter(_idaes_filter)
    for name in ["idaes", "idaes.init", "idaes.core.util.scaling"]:
        lgr = logging.getLogger(name)
        lgr.setLevel(logging.ERROR)
        if not any(isinstance(f, IDAESInfoFilter) for f in lgr.filters):
            lgr.addFilter(_idaes_filter)


def _init_flowsheet(has_genericNP, **main_kwargs):
    """Call bsm2_genericNP.main() with try/except recovery. Returns (m, results)."""
    _suppress_idaes_logging()
    try:
        return bsm2_genericNP.main(
            has_genericNP=has_genericNP,
            apply_costing=True,
            plot_network_before_solve=False,
            **main_kwargs,
        )
    except Exception as e:
        print(f"  main() raised {type(e).__name__}, rebuilding")
        m = bsm2_genericNP.build_flowsheet(has_genericNP=has_genericNP, basis="mass")
        if has_genericNP:
            bsm2_genericNP.set_operating_conditions(
                m,
                p_removal=main_kwargs.get("p_removal", 0.95),
                nh4_removal=main_kwargs.get("nh4_removal", 0.285),
                energy_intensity=main_kwargs.get("energy_intensity", 0.044),
            )
        else:
            bsm2_genericNP.set_operating_conditions(m)
        bsm2_genericNP.initialize_system(m, has_genericNP=has_genericNP)
        for mx in m.fs.mixers:
            mx.pressure_equality_constraints[0.0, 2].deactivate()
        m.fs.MX3.pressure_equality_constraints[0.0, 2].deactivate()
        m.fs.MX3.pressure_equality_constraints[0.0, 3].deactivate()
        costing_kw = {
            k: main_kwargs[k]
            for k in ("phosphorus_recovery_value", "ammonia_recovery_value")
            if k in main_kwargs
        }
        bsm2_genericNP.add_costing(m, **costing_kw)
        m.fs.costing.initialize()
        solver = get_solver()
        solver.options.update({"max_iter": 3000, "max_cpu_time": 120.0})
        results = solver.solve(m, tee=False)
        if pyo.check_optimal_termination(results):
            print(f"  Recovery solve succeeded")
        else:
            print(f"  Recovery solve: {results.solver.termination_condition}")
        return m, results


def _recovery_solve(m, results):
    """If results non-optimal, attempt aggressive re-solve."""
    if pyo.check_optimal_termination(results):
        return results
    print(f"  Attempting recovery ({results.solver.termination_condition})")
    solver = get_solver()
    solver.options.update({"max_iter": 3000, "max_cpu_time": 120.0})
    try:
        results = solver.solve(m, tee=False)
        print(f"  Recovery: {results.solver.termination_condition}")
    except Exception as e:
        print(f"  Recovery error: {e}")
    return results


def build_model(
    has_genericNP=False,
    influent_flow_m3_day=20000,
    influent_cod_mg_L=600,
    electricity_cost=0.12,
    **gnp_kwargs,
):
    """Build BSM2 flowsheet. For genericNP, gradually transitions from safe init values.

    Parameters
    has_genericNP : bool
        Include genericNP sidestream unit
    influent_flow_m3_day : float
        Influent flow rate (m³/day)
    influent_cod_mg_L : float
        Influent COD concentration (mg/L)
    electricity_cost : float
        Electricity cost ($/kWh)
    **gnp_kwargs : dict
        For genericNP: p_removal, n_to_p_ratio, nh4_removal, energy_intensity,
        phosphorus_recovery_value, ammonia_recovery_value
    """
    solver = get_solver()
    solver.options["max_cpu_time"] = 60.0

    if has_genericNP:
        # Init with values matching tear guesses to avoid AD convergence issues
        init_p = 0.95
        init_nh4 = init_p * gnp_kwargs.get("n_to_p_ratio", 0.3)
        init_ei, init_rv = 0.044, 0.1

        m, results = _init_flowsheet(
            True,
            p_removal=init_p,
            nh4_removal=init_nh4,
            energy_intensity=init_ei,
            phosphorus_recovery_value=init_rv,
            ammonia_recovery_value=init_rv,
        )
        results = _recovery_solve(m, results)

        # Gradually transition influent flow/composition (more steps = smaller jumps)
        transition_influent(
            m,
            target_flow_m3_day=influent_flow_m3_day,
            target_cod_mg_L=influent_cod_mg_L,
            n_steps=8,
            solver=solver,
        )

        # Gradually transition genericNP parameters (large jumps cause infeasibility)
        targets = {
            "p_removal": gnp_kwargs.get("p_removal", 0.0),
            "nh4_removal": gnp_kwargs.get("nh4_removal", 0.9),
            "energy_intensity": gnp_kwargs.get("energy_intensity", 156),
            "phosphorus_recovery_value": gnp_kwargs.get(
                "phosphorus_recovery_value", 0.0
            ),
            "ammonia_recovery_value": gnp_kwargs.get("ammonia_recovery_value", 10.0),
        }
        inits = dict(
            p_removal=init_p,
            nh4_removal=init_nh4,
            energy_intensity=init_ei,
            phosphorus_recovery_value=init_rv,
            ammonia_recovery_value=init_rv,
        )
        energy_units = (
            pyo.units.kWh / pyo.units.kg
            if m.fs.genericNP.config.basis == "mass"
            else pyo.units.kWh / pyo.units.mol
        )

        for step in range(1, 5):
            frac = step / 4

            def lerp(i, t):
                return i + frac * (t - i)

            m.fs.genericNP.removal_factors["S_PO4"].set_value(
                lerp(inits["p_removal"], targets["p_removal"])
            )
            m.fs.genericNP.removal_factors["S_NH4"].set_value(
                lerp(inits["nh4_removal"], targets["nh4_removal"])
            )
            ei = lerp(inits["energy_intensity"], targets["energy_intensity"])
            for comp in ("S_PO4", "S_NH4"):
                if comp in m.fs.genericNP.energy_electric_flow:
                    m.fs.genericNP.energy_electric_flow[comp].set_value(
                        ei * energy_units
                    )
            if hasattr(m.fs.costing, "genericNP"):
                m.fs.costing.genericNP.phosphorus_recovery_value = lerp(
                    inits["phosphorus_recovery_value"],
                    targets["phosphorus_recovery_value"],
                )
                m.fs.costing.genericNP.ammonia_recovery_value = lerp(
                    inits["ammonia_recovery_value"], targets["ammonia_recovery_value"]
                )
            try:
                results = solver.solve(m, tee=False)
                converged = pyo.check_optimal_termination(results)
            except ValueError:
                converged = False
            if not converged and step == 4:
                tc = getattr(results.solver, "termination_condition", "error")
                print(f"  WARNING: Final transition step ({tc})")
    else:
        m, results = _init_flowsheet(False)
        results = _recovery_solve(m, results)
        transition_influent(
            m,
            target_flow_m3_day=influent_flow_m3_day,
            target_cod_mg_L=influent_cod_mg_L,
            n_steps=8,
        )

    # MLE simplification: fix R1/R2 as pass-through
    m.fs.R1.volume[0].fix(10.0)
    m.fs.R2.volume[0].fix(10.0)
    try:
        results = solver.solve(m, tee=False)
        if not pyo.check_optimal_termination(results):
            print(f"  WARNING: R1/R2 fix: {results.solver.termination_condition}")
    except ValueError:
        print(f"  WARNING: R1/R2 fix raised solver error")

    m.fs.costing.electricity_cost.fix(electricity_cost)
    return m


def unfix_aeration_reactors(m, reactor_names):
    """Unfix DO concentration in specified aerobic reactors."""
    if not reactor_names:
        return ""
    for r_name in reactor_names:
        reactor = getattr(m.fs, r_name)
        reactor.outlet.conc_mass_comp[:, "S_O2"].unfix()
        reactor.outlet.conc_mass_comp[:, "S_O2"].setlb(0.5e-3)  # 0.5 mg/L
        reactor.outlet.conc_mass_comp[:, "S_O2"].setub(8.2e-3)  # 8.2 mg/L (saturation)
    print(f"  Unfixed DO in {', '.join(reactor_names)}")
    return ", ".join(reactor_names)


def unfix_split_fractions(m, split_names=None):
    """Unfix split fractions for internal recycle rates."""
    if not split_names:
        return ""
    bounds = {"SP1": ("underflow", 0.3, 0.8), "SP2": ("recycle", 0.8, 0.99)}
    for name in split_names:
        splitter = getattr(m.fs, name, None)
        if splitter is None or name not in bounds:
            continue
        port, lb, ub = bounds[name]
        splitter.split_fraction[:, port].unfix()
        splitter.split_fraction[:, port].setlb(lb)
        splitter.split_fraction[:, port].setub(ub)
    print(f"  Unfixed splits in {', '.join(split_names)}")
    return ", ".join(split_names)


def setup_optimization(
    m,
    hrt_lb=0.0,
    hrt_ub=4.0,
    unfix_aeration=None,
    unfix_splits=None,
    tn_limit_mg_L=None,
    initial_volume=None,
):
    """Set up optimization for genericNP HRT/volume."""
    start_volume = (
        initial_volume
        if initial_volume is not None
        else pyo.value(m.fs.genericNP.volume[0])
    )

    if initial_volume is not None:
        m.fs.genericNP.volume[0].set_value(start_volume)
        if hasattr(m.fs.genericNP, "mixed_state"):
            flow_rate = pyo.value(m.fs.genericNP.mixed_state[0].flow_vol) * 3600
            if flow_rate > 0:
                m.fs.genericNP.hydraulic_retention_time[0].set_value(
                    start_volume / flow_rate
                )

    m.fs.genericNP.hydraulic_retention_time[0].unfix()
    m.fs.genericNP.hydraulic_retention_time[0].setlb(hrt_lb)
    m.fs.genericNP.hydraulic_retention_time[0].setub(hrt_ub)
    m.fs.genericNP.volume[0].unfix()
    m.fs.genericNP.volume[0].setlb(1e-6)
    m.fs.genericNP.volume[0].set_value(start_volume)

    if unfix_aeration:
        unfix_aeration_reactors(m, unfix_aeration)
    if unfix_splits:
        unfix_split_fractions(m, unfix_splits)
    if tn_limit_mg_L is not None:
        add_effluent_tn_constraint(m, tn_limit_mg_L)

    m.fs.objective = pyo.Objective(expr=m.fs.costing.LCOW, sense=pyo.minimize)


def setup_thermal_stripping(m, kLa=0.45):
    """Configure genericNP as thermal stripping unit.

    removal = 1 - exp(-kLa * HRT), capital $55.56/m³, energy 156 kWh/kgN.
    """
    m.fs.genericNP.kLa = pyo.Param(
        initialize=kLa,
        units=pyo.units.hr**-1,
        doc="Volumetric mass transfer coefficient for ammonia",
    )

    m.fs.genericNP.removal_factors["S_NH4"].unfix()

    @m.fs.genericNP.Constraint(doc="NH4 removal = 1 - exp(-kLa * HRT)")
    def thermal_stripping_removal(blk):
        return blk.removal_factors["S_NH4"] == 1 - pyo.exp(
            -blk.kLa * blk.hydraulic_retention_time[0]
        )

    m.fs.costing.genericNP.sizing_cost.fix(2000 / 36)
    m.fs.genericNP.energy_electric_flow["S_NH4"].fix(156)
    m.fs.genericNP.magnesium_chloride_dosage.fix(0)
    for comp in m.fs.genericNP.energy_electric_flow:
        if comp != "S_NH4":
            m.fs.genericNP.energy_electric_flow[comp].fix(0)
    for comp in ("S_PO4", "S_NO3", "S_NO2"):
        m.fs.genericNP.removal_factors[comp].fix(0)


def _cleanup_scenario(m, aeration_names=None, split_names=None):
    """Remove scenario-specific components and refix variables so the model can be reused."""
    for comp_name in ["objective", "eq_effluent_TN_limit"]:
        if hasattr(m.fs, comp_name):
            m.fs.del_component(getattr(m.fs, comp_name))
    for r in ["R3", "R4"]:
        getattr(m.fs, r).volume[0].fix(10.0)
    for r_name in aeration_names or []:
        reactor = getattr(m.fs, r_name, None)
        if reactor is not None:
            reactor.outlet.conc_mass_comp[:, "S_O2"].fix()
    split_ports = {"SP1": "underflow", "SP2": "recycle"}
    for name in split_names or []:
        splitter = getattr(m.fs, name, None)
        if splitter is not None and name in split_ports:
            splitter.split_fraction[:, split_ports[name]].fix()
    if hasattr(m.fs, "genericNP"):
        m.fs.genericNP.hydraulic_retention_time[0].fix()
        m.fs.genericNP.volume[0].fix()


#  Solver
def solve_optimization(
    m, solver, tee=True, max_iter=5000, max_cpu_time=120, unfix_aeration=False
):
    """Solve optimization with warm-start settings."""
    if max_iter is not None:
        solver.options["max_iter"] = max_iter
    if max_cpu_time is not None:
        solver.options["max_cpu_time"] = float(max_cpu_time)
    solver.options.update(
        {
            "tol": 1e-3,
            "acceptable_tol": 1e-2,
            "acceptable_constr_viol_tol": 1e-3,
            "warm_start_init_point": "yes",
            "warm_start_bound_push": 1e-6,
            "warm_start_mult_bound_push": 1e-6,
            "mu_init": 1e-4,
        }
    )
    if unfix_aeration and isinstance(unfix_aeration, list):
        solver.options["bound_relax_factor"] = 1e-6

    try:
        results = solver.solve(m, tee=tee)
    except ValueError as e:
        print(f"  Solver error: {e}")
        return None

    if not pyo.check_optimal_termination(results):
        print(f"  Solver termination: {results.solver.termination_condition}")
        print(f"  DOF: {degrees_of_freedom(m)}")
    return results


def solve_staged(m, split_names, aeration_names, solver, tee=False, label=""):
    """Solve optimization with staged DOF release: volumes -> +splits -> +aeration.

    Each stage warm-starts from the previous stage's solution, making it easier
    for IPOPT to navigate the space from a poor initial point.
    """
    prefix = f"  [{label}] " if label else "  "
    results = None
    for name, setup_fn, ua in [
        ("Stage 1: volumes", lambda: None, False),
        ("Stage 2: + splits", lambda: unfix_split_fractions(m, split_names), False),
        (
            "Stage 3: + aeration",
            lambda: unfix_aeration_reactors(m, aeration_names),
            aeration_names,
        ),
    ]:
        setup_fn()
        dof = degrees_of_freedom(m)
        results = solve_optimization(m, solver=solver, tee=tee, unfix_aeration=ua)
        ok = results is not None and pyo.check_optimal_termination(results)
        lcow = f" LCOW={pyo.value(m.fs.costing.LCOW):.4f}" if results else ""
        tc = (
            "" if ok else (results.solver.termination_condition if results else "error")
        )
        print(f"{prefix}{name} ({dof} DOF): {'OK' if ok else tc}{lcow}")
    return results


#  LCOW extraction


def _wwtp_baseline(m, exclude_units=None):
    """Compute baseline WWTP LCOW components (to subtract for upgrade-only costs)."""
    exclude = {
        f"fs.{u}" if not u.startswith("fs.") else u for u in (exclude_units or [])
    }
    capex = indirect = fixed = 0.0
    for u in WWTP_UNITS:
        if u in exclude or u not in m.fs.costing.LCOW_component_direct_capex:
            continue
        capex += pyo.value(m.fs.costing.LCOW_component_direct_capex[u])
        indirect += pyo.value(m.fs.costing.LCOW_component_indirect_capex[u])
        fixed += pyo.value(m.fs.costing.LCOW_component_fixed_opex[u])
    return {"direct_capex": capex, "indirect_capex": indirect, "fixed_opex": fixed}


def _recovery_value_lcow(m):
    """Recovery value as LCOW credit ($/m³)."""
    annual_flow = pyo.value(m.fs.FeedWater.properties[0].flow_vol) * 365.25 * 86400
    if annual_flow <= 0 or not hasattr(m.fs, "genericNP"):
        return 0.0
    total = 0.0
    for product in ["ammonia product", "phosphorus salt product"]:
        if product in m.fs.costing.aggregate_flow_costs:
            total += (
                abs(pyo.value(m.fs.costing.aggregate_flow_costs[product])) / annual_flow
            )
    return total


def extract_lcow_components(m, exclude_units=None):
    """Aggregate upgrade LCOW: direct_capex, indirect_capex, fixed_opex, variable_opex, recovery_value."""
    agg = {
        k: sum(pyo.value(v) for v in getattr(m.fs.costing, attr).values())
        for k, attr in [
            ("direct_capex", "LCOW_aggregate_direct_capex"),
            ("indirect_capex", "LCOW_aggregate_indirect_capex"),
            ("fixed_opex", "LCOW_aggregate_fixed_opex"),
        ]
    }
    baseline = _wwtp_baseline(m, exclude_units)
    result = {k: agg[k] - baseline[k] for k in agg}
    result["recovery_value"] = _recovery_value_lcow(m)
    upgrade_lcow = pyo.value(m.fs.costing.LCOW) - sum(baseline.values())
    result["variable_opex"] = (
        upgrade_lcow - sum(result[k] for k in agg) + result["recovery_value"]
    )
    return result


def extract_lcow_by_upgrade(m, upgrade_units=("R3", "R4", "genericNP")):
    """Per-unit LCOW breakdown: {unit}_capex, total_opex, recovery_value."""
    result = {}
    for unit in upgrade_units:
        u = f"fs.{unit}"
        capex = 0.0
        if u in m.fs.costing.LCOW_component_direct_capex:
            capex += pyo.value(m.fs.costing.LCOW_component_direct_capex[u])
        if u in m.fs.costing.LCOW_component_indirect_capex:
            capex += pyo.value(m.fs.costing.LCOW_component_indirect_capex[u])
        result[f"{unit}_capex"] = capex

    baseline = _wwtp_baseline(m, upgrade_units)
    result["recovery_value"] = _recovery_value_lcow(m)
    upgrade_lcow = pyo.value(m.fs.costing.LCOW) - sum(baseline.values())
    upgrade_capex = sum(result[f"{u}_capex"] for u in upgrade_units)
    agg_fixed = sum(
        pyo.value(v) for v in m.fs.costing.LCOW_aggregate_fixed_opex.values()
    )
    upgrade_fixed = agg_fixed - baseline["fixed_opex"]
    result["total_opex"] = upgrade_lcow - upgrade_capex + result["recovery_value"]
    return result


def extract_scenario_results(
    m, scenario, aeration_names, split_names, do_component="S_O2"
):
    """Extract standard results from a solved model."""
    do_conc = {}
    for r_name in aeration_names:
        reactor = getattr(m.fs, r_name, None)
        if reactor is not None:
            do_conc[r_name] = (
                pyo.value(reactor.outlet.conc_mass_comp[0, do_component]) * 1000
            )

    split_frac = {}
    for name in split_names:
        splitter = getattr(m.fs, name, None)
        if splitter is not None:
            port = "underflow" if name == "SP1" else "recycle"
            split_frac[name] = pyo.value(splitter.split_fraction[0, port])

    results_dict = {
        "lcow": calculate_upgrade_lcow(m),
        "lcow_components": extract_lcow_components(
            m, exclude_units=scenario.get("exclude_units")
        ),
        "electricity_kwh_day": pyo.value(m.fs.costing.aggregate_flow_electricity),
        "tn": get_effluent_tn_kg_m3(m, 0) * 1000,
        "do_concentrations": do_conc or None,
        "split_fractions": split_frac or None,
    }

    # Reactor volumes
    results_dict["reactor_volumes"] = {
        r: pyo.value(getattr(m.fs, r).volume[0])
        for r in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
        if hasattr(m.fs, r)
    }

    if hasattr(m.fs, "genericNP"):
        results_dict["genericNP_hrt"] = pyo.value(
            m.fs.genericNP.hydraulic_retention_time[0]
        )
        results_dict["genericNP_volume"] = pyo.value(m.fs.genericNP.volume[0])

    # Effluent N species
    treated = m.fs.Treated.properties[0]
    n_species = {}
    for comp in ["S_NH4", "S_NO3", "S_NO2"]:
        try:
            n_species[comp] = pyo.value(treated.conc_mass_comp[comp]) * 1000
        except (KeyError, AttributeError):
            pass
    results_dict["n_species"] = n_species

    # Unit CAPEX (inlined)
    if scenario.get("extract_capex"):
        results_dict["unit_capex"] = {
            name: float(pyo.value(getattr(m.fs, name).costing.capital_cost))
            for name in scenario["extract_capex"]
            if hasattr(m.fs, name)
        }

    return results_dict


def classify_selected_upgrades(
    results_dict, reactor_vol_threshold=10.0, hrt_threshold=0.15
):
    """Classify which upgrades the optimizer selected based on post-solve values."""
    selected = []
    vols = results_dict.get("reactor_volumes", {})
    r3, r4 = vols.get("R3", 0), vols.get("R4", 0)
    if r3 > reactor_vol_threshold and r4 > reactor_vol_threshold:
        selected.append("two_anoxic_zones")
    elif r3 > reactor_vol_threshold:
        selected.append("one_anoxic_zone")
    gnp_hrt = results_dict.get("genericNP_hrt", 0)
    gnp_vol = results_dict.get("genericNP_volume", 0)
    if gnp_hrt > hrt_threshold and gnp_vol > reactor_vol_threshold:
        selected.append("sidestream_removal")
    return selected or ["none (aeration optimization only)"]


def print_scenario_summary(name, results_dict, status_label="solved"):
    """Print a summary for one scenario."""
    print(f"  {name.replace('_', ' ').upper()} ({status_label})")
    print(f"  Upgrade LCOW:      {results_dict['lcow']:.4f} $/m³")
    print(f"  Effluent TN:       {results_dict['tn']:.2f} mg/L")
    print(f"  Electricity:       {results_dict['electricity_kwh_day']:.0f} kWh/day")
    if results_dict.get("solve_time_s") is not None:
        print(f"  Solve time:        {results_dict['solve_time_s']:.1f} s")
    vols = results_dict.get("reactor_volumes", {})
    for r in ["R3", "R4"]:
        if r in vols:
            print(f"  {r} volume:       {vols[r]:.1f} m³")
    if results_dict.get("genericNP_hrt") is not None:
        print(f"  GenericNP HRT:     {results_dict['genericNP_hrt']:.3f} hr")
        print(f"  GenericNP volume:  {results_dict['genericNP_volume']:.2f} m³")
    if results_dict.get("do_concentrations"):
        print(
            f"  DO (mg/L):         {', '.join(f'{r}: {v:.2f}' for r, v in results_dict['do_concentrations'].items())}"
        )
    if results_dict.get("split_fractions"):
        print(
            f"  Split fractions:   {', '.join(f'{s}: {v:.3f}' for s, v in results_dict['split_fractions'].items())}"
        )
    if results_dict.get("unit_capex"):
        for unit, val in results_dict["unit_capex"].items():
            print(f"  {unit} CAPEX:       ${val:,.0f}")


def print_comparison_table(all_results):
    """Print a summary comparison table across all scenarios."""
    print(f"\n{'='*80}\n  SCENARIO COMPARISON\n{'='*80}")
    print(
        f"{'Scenario':<30} {'LCOW':>8} {'TN':>8} {'R3':>8} {'R4':>8} {'GNP HRT':>8} {'Time(s)':>8}"
    )
    print("-" * 88)
    for name, res in all_results.items():
        if res is None:
            continue
        vols = res.get("reactor_volumes", {})
        lcow = f"{res['lcow']:.4f}" if res.get("lcow") is not None else "N/A"
        tn = f"{res['tn']:.2f}" if res.get("tn") is not None else "N/A"
        r3 = f"{vols.get('R3', 0):.0f}" if "R3" in vols else "-"
        r4 = f"{vols.get('R4', 0):.0f}" if "R4" in vols else "-"
        hrt = (
            f"{res['genericNP_hrt']:.3f}"
            if res.get("genericNP_hrt") is not None
            else "-"
        )
        tm = (
            f"{res['solve_time_s']:.1f}" if res.get("solve_time_s") is not None else "-"
        )
        print(f"{name:<30} {lcow:>8} {tn:>8} {r3:>8} {r4:>8} {hrt:>8} {tm:>8}")
    if "combined" in all_results and all_results["combined"] is not None:
        selected = classify_selected_upgrades(all_results["combined"])
        print(f"\nCombined optimization selected: {', '.join(selected)}")
    print()


#  Diagnostics


def analyze_ammonia_load_split(m):
    """Analyze ammonia load distribution between influent and digestate recycle."""
    print("\nAmmonia Load Analysis")
    streams = [
        ("Influent", m.fs.FeedWater.outlet),
        ("Digestate (dewater overflow)", m.fs.dewater.overflow),
        ("Treated Digestate (after genericNP)", m.fs.genericNP.treated),
        ("Sludge Recycle (SP2)", m.fs.SP2.recycle),
        ("Thickener Overflow (MX3.recycle2)", m.fs.MX3.recycle2),
    ]
    loads = {}
    for name, stream in streams:
        flow = pyo.value(stream.flow_vol[0]) * 86400
        conc = pyo.value(stream.conc_mass_comp[0, "S_NH4"])
        loads[name] = {"flow": flow, "conc": conc * 1000, "load": flow * conc}
        print(f"  {name}: {flow:.2f} m³/day with {conc * 1000:.2f} NH4 mg/L")
    recycle_load = sum(loads[k]["load"] for k in list(loads)[2:])
    total = loads["Influent"]["load"] + recycle_load
    if total > 0:
        print(f"\n  Total NH4 load to R3: {total:.2f} kg/day")
        print(
            f"    Influent: {loads['Influent']['load']/total*100:.1f}%, "
            f"Recycle: {recycle_load/total*100:.1f}%"
        )


#  Variable snapshots (warm-start save/restore)


def save_variable_snapshot(m):
    """Save current values of all active variables for warm-start restore."""
    return {
        id(v): (v, v.value)
        for v in m.component_data_objects(pyo.Var, active=True)
        if v.value is not None
    }


def restore_variable_snapshot(snapshot):
    """Restore variable values from a previously saved snapshot."""
    for _, (v, val) in snapshot.items():
        try:
            if val is not None and val < 0 and abs(val) < 1e-20:
                if v.lb is not None and v.lb >= 0:
                    val = 0.0
            v.set_value(val)
        except (ValueError, AttributeError):
            pass


def perturb_snapshot(snapshot, perturbation=0.05, seed=None):
    """Perturb snapshot values by ±perturbation fraction for multi-start."""
    if seed is not None:
        random.seed(seed)
    for _, (v, val) in snapshot.items():
        if val is None or val == 0:
            continue
        new_val = val * (1.0 + random.uniform(-perturbation, perturbation))
        if v.lb is not None and new_val < v.lb:
            new_val = v.lb
        if v.ub is not None and new_val > v.ub:
            new_val = v.ub
        try:
            v.set_value(new_val)
        except (ValueError, AttributeError):
            pass


def run_tn_limit_sweep(
    tn_limits, build_kwargs, setup_kwargs, solve_kwargs, scenario_meta, output_dir=None
):

    print(
        f"  TN LIMIT SWEEP: {max(tn_limits):.1f} -> {min(tn_limits):.1f} mg/L ({len(tn_limits)} points)"
    )

    # Build model once
    print("  Building model")
    m = build_model(has_genericNP=True, **build_kwargs)

    # Set up combined optimization (volumes only — splits/aeration added by staged solve)
    sk = setup_kwargs
    for r_name in ["R3", "R4"]:
        reactor = getattr(m.fs, r_name)
        reactor.volume.unfix()
        reactor.volume[0].setlb(sk["anoxic_vol_lb"])
        reactor.volume[0].setub(sk["anoxic_vol_ub"])
    setup_thermal_stripping(m, kLa=sk.get("kLa", 0.45))
    setup_optimization(
        m,
        sk["hrt_lb"],
        sk["hrt_ub"],
        unfix_aeration=None,
        unfix_splits=None,
        tn_limit_mg_L=None,
        initial_volume=sk.get("initial_gnp_volume", 4.0),
    )
    m.fs.costing.electricity_cost.fix(sk["electricity_cost"])

    solver = solve_kwargs.get("solver", get_solver())
    tee = solve_kwargs.get("tee", False)
    ua = solve_kwargs.get("unfix_aeration", None)
    do_component = sk.get("do_component", "S_O2")
    max_restarts = solve_kwargs.get("max_restarts", 3)

    # Staged pre-solve (no TN constraint)
    print("  Staged pre-solve (no TN limit)")
    solve_staged(
        m, sk["unfix_splits"], sk["unfix_aeration"], solver, tee, label="pre-solve"
    )
    print(f"  Pre-solve TN={get_effluent_tn_kg_m3(m, 0)*1000:.2f} mg/L")

    # Save initial state as fallback
    initial_snapshot = save_variable_snapshot(m)
    best_snapshot = initial_snapshot

    sweep_results = []
    sweep_snapshots = []
    for i, tn_limit in enumerate(tn_limits):
        print(f"\n TN limit: {tn_limit:.2f} mg/L ({i+1}/{len(tn_limits)})")
        t_start = time.perf_counter()

        if hasattr(m.fs, "eq_effluent_TN_limit"):
            m.fs.del_component(m.fs.eq_effluent_TN_limit)
        add_effluent_tn_constraint(m, tn_limit)

        results = solve_optimization(m, solver=solver, tee=tee, unfix_aeration=ua)
        solve_time = time.perf_counter() - t_start
        optimal = results is not None and pyo.check_optimal_termination(results)

        # Multi-start fallback
        if not optimal and best_snapshot is not None:
            for restart in range(max_restarts):
                print(
                    f"    Restart {restart+1}/{max_restarts}: restoring best snapshot"
                    + (" (perturbed)" if restart > 0 else "")
                )
                restore_variable_snapshot(best_snapshot)
                if restart > 0:
                    perturb_snapshot(
                        best_snapshot, perturbation=0.05 * restart, seed=restart * 42
                    )
                if hasattr(m.fs, "eq_effluent_TN_limit"):
                    m.fs.del_component(m.fs.eq_effluent_TN_limit)
                add_effluent_tn_constraint(m, tn_limit)
                t_restart = time.perf_counter()
                results = solve_optimization(
                    m, solver=solver, tee=tee, unfix_aeration=ua
                )
                solve_time = time.perf_counter() - t_start
                if results is not None and pyo.check_optimal_termination(results):
                    optimal = True
                    print(
                        f"    Restart {restart+1} succeeded ({time.perf_counter()-t_restart:.1f}s)"
                    )
                    break

        if results is None:
            print(f"  All attempts failed at TN={tn_limit:.2f}")
            if best_snapshot is not None:
                restore_variable_snapshot(best_snapshot)
            sweep_results.append(
                {
                    "tn_limit": tn_limit,
                    "status": "error",
                    "results": None,
                    "lcow_components": None,
                    "lcow_by_upgrade": None,
                }
            )
            sweep_snapshots.append(None)
            continue

        status = "optimal" if optimal else str(results.solver.termination_condition)
        results_dict = extract_scenario_results(
            m, scenario_meta, sk["unfix_aeration"], sk["unfix_splits"], do_component
        )
        point_snapshot = save_variable_snapshot(m) if optimal else None
        sweep_snapshots.append(point_snapshot)

        if optimal:
            best_snapshot = point_snapshot
        elif best_snapshot is not None:
            restore_variable_snapshot(best_snapshot)

        results_dict["solve_time_s"] = solve_time
        lcow_comp = extract_lcow_components(
            m, exclude_units=scenario_meta.get("exclude_units")
        )
        lcow_by_upgrade = extract_lcow_by_upgrade(
            m, upgrade_units=["R3", "R4", "genericNP"]
        )

        selected = classify_selected_upgrades(results_dict)
        print(
            f"  LCOW={results_dict['lcow']:.4f} $/m³, TN={results_dict['tn']:.2f} mg/L, "
            f"status={status}, upgrades={', '.join(selected)}, time={solve_time:.1f}s"
        )

        sweep_results.append(
            {
                "tn_limit": tn_limit,
                "status": status,
                "results": results_dict,
                "lcow_components": lcow_comp,
                "lcow_by_upgrade": lcow_by_upgrade,
            }
        )

    # Backward repair & monotonicity pass
    print("Backward")
    n_fixed = 0
    for i in range(len(sweep_results) - 2, -1, -1):
        sr = sweep_results[i]
        tn_limit = sr["tn_limit"]
        is_non_optimal = sr.get("status") != "optimal"

        # Find nearest tighter optimal snapshot
        warm_snapshot = warm_lcow = None
        for j in range(i + 1, len(sweep_results)):
            if sweep_snapshots[j] is not None:
                warm_snapshot = sweep_snapshots[j]
                warm_lcow = sweep_results[j]["results"]["lcow"]
                break
        if warm_snapshot is None:
            continue

        need_resolve = is_non_optimal
        if not need_resolve and sr.get("results") is not None:
            if sr["results"]["lcow"] > warm_lcow:
                need_resolve = True
        if not need_resolve:
            continue

        reason = (
            "non-optimal"
            if is_non_optimal
            else f"LCOW={sr['results']['lcow']:.4f} > {warm_lcow:.4f}"
        )
        print(f"    TN={tn_limit:.2f}: {reason}, re-solving from tighter neighbour")

        restore_variable_snapshot(warm_snapshot)
        if hasattr(m.fs, "eq_effluent_TN_limit"):
            m.fs.del_component(m.fs.eq_effluent_TN_limit)
        add_effluent_tn_constraint(m, tn_limit)

        t_start = time.perf_counter()
        results = solve_optimization(m, solver=solver, tee=tee, unfix_aeration=ua)
        solve_time = time.perf_counter() - t_start

        if results is None or not pyo.check_optimal_termination(results):
            print(f"    Re-solve failed, keeping original")
            continue

        new_rd = extract_scenario_results(
            m, scenario_meta, sk["unfix_aeration"], sk["unfix_splits"], do_component
        )
        new_lcow = new_rd["lcow"]
        old_lcow = sr["results"]["lcow"] if sr.get("results") else float("inf")

        if is_non_optimal or new_lcow < old_lcow:
            new_rd["solve_time_s"] = solve_time
            selected = classify_selected_upgrades(new_rd)
            was = "was non-optimal" if is_non_optimal else f"was {old_lcow:.4f}"
            print(
                f"    Improved: LCOW={new_lcow:.4f} ({was}), upgrades={', '.join(selected)}"
            )
            sweep_results[i] = {
                "tn_limit": tn_limit,
                "status": "optimal",
                "results": new_rd,
                "lcow_components": extract_lcow_components(
                    m, exclude_units=scenario_meta.get("exclude_units")
                ),
                "lcow_by_upgrade": extract_lcow_by_upgrade(
                    m, upgrade_units=["R3", "R4", "genericNP"]
                ),
            }
            sweep_snapshots[i] = save_variable_snapshot(m)
            n_fixed += 1
        else:
            print(
                f"    Re-solve not better ({new_lcow:.4f} vs {old_lcow:.4f}), keeping original"
            )

    print(f"  Backward pass complete: {n_fixed} points improved")

    if output_dir is not None:
        csv_file = output_dir / "tn_limit_sweep.csv"
        fieldnames = [
            "tn_limit",
            "status",
            "lcow",
            "tn_actual",
            "R3_capex",
            "R4_capex",
            "genericNP_capex",
            "total_opex",
            "recovery_value",
            "R3_vol",
            "R4_vol",
            "gnp_hrt",
            "gnp_vol",
            "solve_time_s",
        ]
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sr in sweep_results:
                row = {"tn_limit": sr["tn_limit"], "status": sr["status"]}
                if sr["results"] is not None and sr.get("lcow_by_upgrade") is not None:
                    r, u = sr["results"], sr["lcow_by_upgrade"]
                    row.update(
                        {
                            "lcow": r.get("lcow"),
                            "tn_actual": r.get("tn"),
                            "R3_capex": u.get("R3_capex"),
                            "R4_capex": u.get("R4_capex"),
                            "genericNP_capex": u.get("genericNP_capex"),
                            "total_opex": u.get("total_opex"),
                            "recovery_value": u.get("recovery_value"),
                            "R3_vol": r.get("reactor_volumes", {}).get("R3"),
                            "R4_vol": r.get("reactor_volumes", {}).get("R4"),
                            "gnp_hrt": r.get("genericNP_hrt"),
                            "gnp_vol": r.get("genericNP_volume"),
                            "solve_time_s": r.get("solve_time_s"),
                        }
                    )
                writer.writerow(row)
        print(f"\n  Sweep results saved to {csv_file}")

    return sweep_results


def plot_lcow_stacked(sweep_results, output_dir=None, baseline_tn=None):
    """Two-panel plot: LCOW components (top) and effluent N species (bottom)."""
    COLORS = {
        "R3 CAPEX": "#E0C7ED",
        "R4 CAPEX": "#B19CBA",
        "GenericNP CAPEX": "#FED5D8",
        "Total OPEX": "#F39C12",
        "Recovery Credit": "#FED5D8",
        "NH4": "#F39C12",
        "NOx": "#134920",
    }

    valid = [sr for sr in sweep_results if sr.get("lcow_by_upgrade") is not None]
    if len(valid) < 2:
        print("  Not enough valid points for LCOW plot")
        return

    tn_limits = np.array([sr["tn_limit"] for sr in valid])
    if baseline_tn is None:
        baseline_tn = tn_limits[0]
    tn_pct = (baseline_tn - tn_limits) / baseline_tn * 100

    labels = ["R3 CAPEX", "R4 CAPEX", "GenericNP CAPEX", "Total OPEX"]
    keys = ["R3_capex", "R4_capex", "genericNP_capex", "total_opex"]
    components = {
        l: np.array([sr["lcow_by_upgrade"][k] for sr in valid])
        for l, k in zip(labels, keys)
    }
    recovery = np.array([sr["lcow_by_upgrade"]["recovery_value"] for sr in valid])

    nh4 = np.array([sr["results"].get("n_species", {}).get("S_NH4", 0) for sr in valid])
    no3 = np.array([sr["results"].get("n_species", {}).get("S_NO3", 0) for sr in valid])
    no2 = np.array([sr["results"].get("n_species", {}).get("S_NO2", 0) for sr in valid])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    # LCOW
    bottom = np.zeros_like(tn_pct, dtype=float)
    for label in labels:
        ax1.fill_between(
            tn_pct,
            bottom,
            bottom + components[label],
            alpha=0.7,
            label=label,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += components[label]
    total_lcow = np.array([sr["results"]["lcow"] for sr in valid])
    ax1.plot(
        tn_pct,
        total_lcow,
        "k-o",
        linewidth=2.5,
        markersize=5,
        label="Total Upgrade LCOW",
        zorder=10,
    )
    if np.any(recovery > 0):
        ax1.fill_between(
            tn_pct,
            0,
            -recovery,
            alpha=0.5,
            label="Recovery Credit",
            color=COLORS["Recovery Credit"],
            edgecolor="white",
            linewidth=0.5,
            hatch="//",
        )
    ax1.set_ylabel("LCOW Contribution ($/m³)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Upgrade LCOW and Effluent N Species vs. TN Reduction\n"
        "(Combined Scenario: Anoxic Zones + Sidestream Recovery)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    # Bottom: effluent N species bars
    bw = max((tn_pct[-1] - tn_pct[0]) / len(valid) * 0.35, 0.5)
    ax2.bar(
        tn_pct - bw / 2,
        nh4,
        bw,
        label="NH$_4^+$",
        color=COLORS["NH4"],
        alpha=0.8,
        edgecolor="white",
    )
    ax2.bar(
        tn_pct + bw / 2,
        no3 + no2,
        bw,
        label="NO$_x$",
        color=COLORS["NOx"],
        alpha=0.8,
        edgecolor="white",
    )
    ax2.plot(
        tn_pct, tn_limits, "k--", linewidth=1.5, alpha=0.6, label="TN Limit", zorder=5
    )
    ax2.set_xlabel(
        f"TN Reduction from Baseline ({baseline_tn:.1f} mg/L) [%]",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_ylabel("Effluent Concentration (mg/L)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    if output_dir is not None:
        plot_file = output_dir / "lcow_vs_tn_reduction.png"
        plt.savefig(plot_file, dpi=200, bbox_inches="tight")
        print(f"  Plot saved to {plot_file}")
    plt.close()


if __name__ == "__main__":
    tn_limit_mg_L = 20.0
    hrt_lb, hrt_ub = 0.1, 5.0
    unfix_aeration = ["R5", "R6", "R7"]
    unfix_splits = ["SP1", "SP2"]
    influent_flow_m3_day = 12 * 3785.41  # 12 MGD
    influent_cod_mg_L = 600
    electricity_cost = 0.12
    tee = False
    anoxic_vol_lb, anoxic_vol_ub = 10.0, 5000.0
    initial_gnp_volume = 4.0
    solver = get_solver()

    def _setup_scenario(m, s):
        """Configure optimization scenario on model m."""
        for r in ["R3", "R4"]:
            if r in s.get("unfix_anoxic", []):
                getattr(m.fs, r).volume.unfix()
                getattr(m.fs, r).volume[0].setlb(anoxic_vol_lb)
                getattr(m.fs, r).volume[0].setub(anoxic_vol_ub)
            else:
                getattr(m.fs, r).volume[0].fix(10.0)

        if not s["optimize"]:
            solver.solve(m, tee=False)
            return

        if s.get("thermal_stripping"):
            # thermal_stripping components already added at group level
            setup_optimization(
                m,
                hrt_lb,
                hrt_ub,
                unfix_aeration=None,
                unfix_splits=None,
                tn_limit_mg_L=tn_limit_mg_L,
                initial_volume=initial_gnp_volume,
            )
            m.fs.costing.electricity_cost.fix(electricity_cost)
        else:
            add_effluent_tn_constraint(m, tn_limit_mg_L)
            m.fs.objective = pyo.Objective(expr=m.fs.costing.LCOW, sense=pyo.minimize)

    # just run baseline and combined for now
    scenarios = [
        # {'name': 'baseline', 'has_genericNP': False, 'optimize': False,
        #  'exclude_units': None, 'extract_capex': None},
        # {'name': 'one_anoxic_zone', 'has_genericNP': False, 'optimize': True,
        #  'unfix_anoxic': ['R3'], 'exclude_units': ['R3'], 'extract_capex': ['R3']},
        # {'name': 'two_anoxic_zones', 'has_genericNP': False, 'optimize': True,
        #  'unfix_anoxic': ['R3', 'R4'], 'exclude_units': ['R3', 'R4'], 'extract_capex': ['R3', 'R4']},
        # {'name': 'sidestream_only', 'has_genericNP': True, 'optimize': True,
        #  'thermal_stripping': True, 'exclude_units': None, 'extract_capex': ['genericNP']},
        # {'name': 'combined', 'has_genericNP': True, 'optimize': True,
        #  'unfix_anoxic': ['R3', 'R4'], 'thermal_stripping': True,
        #  'exclude_units': ['R3', 'R4'], 'extract_capex': ['R3', 'R4', 'genericNP']},
    ]

    # Group scenarios by has_genericNP to build model once per type
    all_results = {}
    for has_gnp, group in groupby(scenarios, key=lambda s: s["has_genericNP"]):
        group = list(group)
        print(
            f"  Building {'genericNP' if has_gnp else 'non-genericNP'} model "
            f"(reusing for {len(group)} scenario(s))"
        )
        t_build = time.perf_counter()
        m = build_model(
            has_genericNP=has_gnp,
            influent_flow_m3_day=influent_flow_m3_day,
            influent_cod_mg_L=influent_cod_mg_L,
            electricity_cost=electricity_cost,
        )
        build_time = time.perf_counter() - t_build
        print(f"  Model built in {build_time:.1f}s")
        clean_snapshot = save_variable_snapshot(m)

        # For genericNP scenarios with thermal_stripping, set it up once
        gnp_thermal_setup = False

        for s in group:
            name = s["name"]
            print(f"\nSCENARIO: {name.upper().replace('_', ' ')}")
            t_start = time.perf_counter()

            # Restore clean state from build (reuse model)
            restore_variable_snapshot(clean_snapshot)
            _cleanup_scenario(
                m, aeration_names=unfix_aeration, split_names=unfix_splits
            )

            # For genericNP thermal stripping: set up once, keep for all gnp scenarios
            if s.get("thermal_stripping") and not gnp_thermal_setup:
                setup_thermal_stripping(m, kLa=0.45)
                gnp_thermal_setup = True

            _setup_scenario(m, s)

            if s["optimize"]:
                print(f"  DOF (stage 1): {degrees_of_freedom(m)}")
                results = solve_staged(
                    m, unfix_splits, unfix_aeration, solver, tee, label=name
                )
                if results is None:
                    print(f"  Skipping {name} (solver error)")
                    all_results[name] = None
                    continue
                optimal = pyo.check_optimal_termination(results)
                status_label = (
                    "optimal"
                    if optimal
                    else f"non-optimal ({results.solver.termination_condition})"
                )
            else:
                status_label = "solved"

            solve_time_s = time.perf_counter() - t_start
            results_dict = extract_scenario_results(m, s, unfix_aeration, unfix_splits)
            results_dict["solve_time_s"] = solve_time_s
            print_scenario_summary(name, results_dict, status_label)

            if name == "combined":
                selected = classify_selected_upgrades(results_dict)
                print(f"  --> Selected upgrades: {', '.join(selected)}")

            all_results[name] = results_dict

    print_comparison_table(all_results)

    output_data_dir = Path(__file__).parent / "output_data"
    output_data_dir.mkdir(exist_ok=True)
    results_file = output_data_dir / "optimization_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                **all_results,
                "optimization_params": {
                    "tn_limit_mg_L": tn_limit_mg_L,
                    "hrt_lb": hrt_lb,
                    "hrt_ub": hrt_ub,
                    "anoxic_vol_lb": anoxic_vol_lb,
                    "anoxic_vol_ub": anoxic_vol_ub,
                    "thermal_stripping_kLa": 0.45,
                    "thermal_stripping_capital_cost_per_m3": 2000 / 36,
                    "thermal_stripping_energy_kwh_per_kgN": 156,
                    "electricity_cost": electricity_cost,
                },
            },
            f,
            indent=2,
        )
    print(f"Results saved to {results_file}")

    # TN Limit Sweep
    tn_sweep_limits = np.linspace(20.0, 15.0, 6).tolist()
    sweep_results = run_tn_limit_sweep(
        tn_limits=tn_sweep_limits,
        build_kwargs={
            "influent_flow_m3_day": influent_flow_m3_day,
            "influent_cod_mg_L": influent_cod_mg_L,
            "ammonia_recovery_value": 0.0,
            "phosphorus_recovery_value": 0.0,
        },
        setup_kwargs={
            "anoxic_vol_lb": anoxic_vol_lb,
            "anoxic_vol_ub": anoxic_vol_ub,
            "hrt_lb": hrt_lb,
            "hrt_ub": hrt_ub,
            "initial_gnp_volume": initial_gnp_volume,
            "unfix_aeration": unfix_aeration,
            "unfix_splits": unfix_splits,
            "electricity_cost": electricity_cost,
            "do_component": "S_O2",
            "kLa": 0.45,
        },
        solve_kwargs={"solver": solver, "tee": tee, "unfix_aeration": unfix_aeration},
        scenario_meta={
            "exclude_units": ["R3", "R4"],
            "extract_capex": ["R3", "R4", "genericNP"],
        },
        output_dir=output_data_dir,
    )
    baseline_tn = all_results.get("baseline", {}).get("tn")
    plot_lcow_stacked(
        sweep_results, output_dir=output_data_dir, baseline_tn=baseline_tn
    )
