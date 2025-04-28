#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################
from pyomo.environ import (
    ConcreteModel,
    Objective,
    assert_optimal_termination,
    TransformationFactory,
    units as pyunits,
    value,
)
from pyomo.network import Arc

from idaes.core import FlowsheetBlock, UnitModelCostingBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.scaling import calculate_scaling_factors
from idaes.core.util.initialization import propagate_state
from idaes.models.unit_models import Product, Feed

from watertap.core.util.initialization import check_dof
from watertap.property_models.multicomp_aq_sol_prop_pack import MCASParameterBlock
from watertap.unit_models.ion_exchange_0D import IonExchange0D, RegenerantChem
from watertap.costing import WaterTAPCosting
from watertap.core.solvers import get_solver
from idaes.core.util import DiagnosticsToolbox  # TEMPORARY

import math
import pandas as pd

try:
    from watertap.tools.plot_network import plot_network
except:
    print("Missing dependencies for plot_network")


solver = get_solver()


def main():
    # The IX model currently only supports one "target" ion (i.e., the component in the water source that can be removed by IX)
    # All other ions are inert. This demo does not contain inert ions, but an example can be found in the IX test file:
    # watertap/watertap/unit_models/tests/test_ion_exchange_0D.py
    target_ion = "NH4_+"  # , PO4_-"
    ions = [target_ion, "Na_+", "Cl_-"]

    # See ix_build for details on building the model for this demo.
    m, ion_config = ix_build(ions)

    # Plot network initialization
    stream_table = pd.DataFrame()
    # Create column mapping for stream names to node names
    column_mapping = {}

    # Define position adjustments for better layout
    position_adjustments = {
        "ion_exchange": [0, 0],
        "ion_exchange_inlet": [-1, 0],
        "ion_exchange_outlet": [1, 0],
        "ion_exchange_regen": [0, -1],
        "product": [2, 0],
        "regen": [0, -2],
    }

    # Plot the network and save to file
    try:
        plot_network(
            m,
            stream_table,
            path_to_save="flowsheets/ion_exchange/ion_exchange_flowsheet.png",
            figsize=(10, 8),
            column_mapping=column_mapping,
            position_adjustments=position_adjustments,
            node_order=["feed", "ion_exchange", "regen", "product"],
        )
    except:
        print("Error plotting network. This may be due to missing dependencies.")

    # plot_network(
    #         m,
    #         stream_table,
    #         path_to_save="flowsheets/ion_exchange/ion_exchange_flowsheet.png",
    #         figsize=(10, 8),
    #         column_mapping=column_mapping,
    #         position_adjustments=position_adjustments,
    #         node_order=['feed', 'ion_exchange', 'regen', 'product']
    #     )

    # See set_operating_conditions for details on operating conditions for this demo.
    set_operating_conditions(m, ion_config)
    # See initialize_system for details on initializing the models for this demo.
    initialize_system(m)
    # Check the degrees of freedom of the model to ensure it is zero.
    check_dof(m)
    # Solve the model. Store the results in a local variable.
    results = solver.solve(m)
    # Ensure the solve resulted in an optimal termination status.
    assert_optimal_termination(results)
    # Display the degrees of freedom, termination status, and performance metrics of the model.
    print(f"\nDOF = {degrees_of_freedom(m)}")
    print(f"Model solve {results.solver.termination_condition.swapcase()}")
    display_results(m)

    # See optimize_system for details on optimizing this model for a specific condition.
    optimize_system(m)
    ix = m.fs.ion_exchange

    # With our model optimized to new conditions in optimize_system,
    # we can get the new number_columns and bed_depth and fix them in our model.
    num_col = math.ceil(
        ix.number_columns()
    )  # To eliminate fractional number of columns
    bed_depth = ix.bed_depth()
    ix.bed_depth.fix(bed_depth)
    ix.number_columns.fix(num_col)
    check_dof(m)
    results = solver.solve(m)
    assert_optimal_termination(results)
    print(f"\nDOF = {degrees_of_freedom(m)}")
    print(f"Model solve {results.solver.termination_condition.swapcase()}")
    display_results(m)

    regen_props = m.fs.regen.properties[0]

    print("\nConcentrations and flows (mass):")
    for phase in regen_props.phase_list:
        for comp in regen_props.component_list:
            conc = value(regen_props.conc_mass_phase_comp[phase, comp])
            flow = value(regen_props.flow_mass_phase_comp[phase, comp])
            print(f"  {phase}, {comp}: {conc} kg/m³")
            print(f"  {phase}, {comp}: {flow} kg/s")
        vol_flow = value(regen_props.flow_vol_phase[phase])
        print(f"  {phase}: {vol_flow} m³/s")

    for ion in ions:
        print(
            f'Regen stream concentration for {ion}: {value(m.fs.regen.properties[0].conc_mass_phase_comp["Liq", ion])}'
        )
        print(
            f'Regen stream concentration for {ion}: {value(ix.regeneration_stream[0].conc_mass_phase_comp["Liq", ion])}'
        )

    return m


def ix_build(ions, target_ion=None, hazardous_waste=False, regenerant="NaCl"):

    if not target_ion:
        target_ion = ions[0]

    # Create the model and flowsheet
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # ion_config is the dictionary needed to configure the property package.
    # For this demo, we have properties related to the target ion (Ca_2+),
    # the dissolved regenerant ions sodium (Na_+) and chloride (Cl_-)  and water (H2O)
    ion_config = get_ion_config(ions, regenerant)
    regenerant_stoich = ion_config["regenerant_stoich_data"][regenerant]
    regenerant_mw = ion_config["regenerant_mw_data"][regenerant]

    # The water property package used for the ion exchange model is the multi-component aqueous solution (MCAS) property package
    m.fs.properties = MCASParameterBlock(**ion_config["prop_config"])

    # Add the flowsheet level costing package
    m.fs.costing = WaterTAPCosting()

    # Add feed and product blocks to the flowsheet
    # These are the unit models on the flowsheet that the source water "flows" from/to
    # The must use the same property package as the ion exchange model
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.product = Product(property_package=m.fs.properties)
    m.fs.regen = Product(property_package=m.fs.properties)

    # Configuration dictionary used to instantiate the ion exchange model:
    #   "property_package" indicates which property package to use for the ion exchange model
    #   "target_ion" indicates which ion in the property package will be the reactive ion for the ion exchange model
    #   "hazardous_waste" indicates if the regeneration and spent resin is considered hazardous. If so, it adds costs
    #   "regenerant" indicates the chemical used to regenerate the ion exchange process
    ix_config = {
        "property_package": m.fs.properties,
        "target_ion": target_ion,
        "hazardous_waste": hazardous_waste,
        "regenerant": regenerant,
        "regenerant_stoich_data": ion_config["regenerant_stoich_data"],
        "regenerant_mw_data": ion_config["regenerant_mw_data"],
    }

    # Add the ion exchange model to the flowsheet
    m.fs.ion_exchange = ix = IonExchange0D(**ix_config)

    # Touch properties so they are available for scaling, initialization, and reporting.
    ix.process_flow.properties_in[0].conc_mass_phase_comp[...]
    ix.process_flow.properties_out[0].conc_mass_phase_comp[...]
    ix.fresh_regenerant[0].conc_mass_phase_comp[...]
    ix.spent_regenerant[0].conc_mass_phase_comp[...]
    m.fs.feed.properties[0].flow_vol_phase[...]
    m.fs.feed.properties[0].conc_mass_phase_comp[...]
    m.fs.product.properties[0].conc_mass_phase_comp[...]

    # Add costing blocks to the flowsheet
    # Here, the ion exchange model has its own unit-level costing Block
    m.fs.ion_exchange.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing  # Indicating which flowsheet costing block to use to aggregate unit-level costs to the system-level costs
    )
    # Call cost_process() method to create system-wide global parameters and add aggregating constraints to costing model
    m.fs.costing.cost_process()
    # Designate the volumetric flow on the Product block to be the stream used as the annual water production
    m.fs.costing.add_annual_water_production(
        m.fs.product.properties[0].flow_vol_phase["Liq"]
    )
    # Add LCOW variable to costing block
    m.fs.costing.add_LCOW(m.fs.product.properties[0].flow_vol_phase["Liq"])
    # Add specific energy consumption variable to costing block
    m.fs.costing.add_specific_energy_consumption(
        m.fs.product.properties[0].flow_vol_phase["Liq"]
    )

    # Arcs are used to "connect" Ports on unit process models to Ports on other unit process models
    # For example, in this next line the outlet Port on the Feed model is connected to the inlet Port on the ion exchange model
    m.fs.feed_to_ix = Arc(source=m.fs.feed.outlet, destination=ix.inlet)
    m.fs.ix_to_product = Arc(source=ix.outlet, destination=m.fs.product.inlet)
    # m.fs.fresh_regen_to_ix = Arc(source=ix.fresh_regen, destination=ix.inlet)  # REMOVED: fresh_regen is now internal only
    m.fs.ix_to_spent_regen = Arc(source=ix.spent_regen, destination=m.fs.regen.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)

    # Scaling variables in the model
    # Here, the molar flow for water ("flow_mol_phase_comp[Liq, H2O]") on the Feed block is scaled by 1e-4.
    # This is because the molar flow rate of water in this demo is ~2777 mol/s
    # and scaling factors are chosen such that the value of the variable multiplied by the scaling factor is ~1
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e-4, index=("Liq", "H2O")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e-2, index=("Liq", target_ion)
    )

    for comp in regenerant_stoich:
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e-2, index=("Liq", comp)
        )

    # any additional ions present in the property package
    for comp in m.fs.properties.component_list:
        if comp not in regenerant_stoich and comp != target_ion and comp != "H2O":
            m.fs.properties.set_default_scaling(
                "flow_mol_phase_comp", 1e-2, index=("Liq", comp)
            )

    # Call calculate_scaling_factors to apply scaling factors for each variable that we haven't set scaling factors for above.
    calculate_scaling_factors(m)

    return m, ion_config


def set_operating_conditions(
    m, ion_config, flow_in=0.05, conc_mass_in=0.1, solver=None
):
    if solver is None:
        solver = get_solver()
    ix = m.fs.ion_exchange
    target_ion = ix.config.target_ion

    var_args = {
        ("flow_vol_phase", "Liq"): flow_in,  # m3/s
        ("conc_mass_phase_comp", ("Liq", target_ion)): conc_mass_in,  # kg/m3
        ("pressure", None): 101325,
        ("temperature", None): 298,
    }

    component_list = list(m.fs.properties.component_list)
    for comp in component_list:
        if comp != target_ion and comp != "H2O":
            var_args[("conc_mass_phase_comp", ("Liq", comp))] = 0.0

    m.fs.feed.properties.calculate_state(
        var_args=var_args,
        hold_state=True,
    )
    # Fix key decision variables in ion exchange model.
    ix.langmuir[target_ion].fix(0.7)
    ix.resin_max_capacity.fix(3)
    ix.service_flow_rate.fix(15)
    # Note: number_columns and bed_depth are typically not known a priori for this model.
    # They can be determined by first running the model without these variables fixed.
    ix.number_columns.fix(4)
    ix.bed_depth.fix(1.7)

    # Fix remaining variables.
    # Using the .fix() method on a Var fixes the variable to its initialized value.
    ix.resin_diam.fix()
    ix.resin_bulk_dens.fix()
    ix.bed_porosity.fix()
    ix.dimensionless_time.fix()

    # Set service to regeneration flow ratio and regen efficiency
    ix.service_to_regen_flow_ratio.set_value(3)
    ix.regen_efficiency.set_value(0.6)

    ions_removed = value(ix.target_ion_mass)  # mol

    # Initialize fresh regenerant stream state
    fresh_regen_state = ix.fresh_regenerant[0]

    # Set fresh regenerant concentrations based on dosed mass and flow rate
    regenerant_stoich = ix.config.regenerant_stoich_data[ix.config.regenerant]
    for comp in component_list:
        if comp == "H2O":
            fresh_regen_state.conc_mass_phase_comp["Liq", "H2O"].set_value(
                997.0
            )  # kg/m3
        elif comp in regenerant_stoich:
            # Use the model's regenerant concentration expression
            conc = value(ix.regen_concentration) * regenerant_stoich[comp]  # kg/m3
            fresh_regen_state.conc_mass_phase_comp["Liq", comp].set_value(conc)
        else:
            fresh_regen_state.conc_mass_phase_comp["Liq", comp].set_value(0.0)

    # Initialize spent regenerant stream state
    spent_regen_state = ix.spent_regenerant[0]

    # Get charge data from ion_config
    charge_data = ion_config["prop_config"]["charge"]

    # Calculate spent regenerant concentrations
    regen_tank_vol = value(ix.regen_tank_vol)  # Get the tank volume from the model
    for comp in component_list:
        if comp == target_ion:
            spent_regen_state.conc_mass_phase_comp["Liq", comp].set_value(
                ions_removed * m.fs.properties.mw_comp[comp].value / regen_tank_vol
            )
        elif comp in regenerant_stoich:
            print(
                f"Debug - Component {comp}: charge={charge_data[comp]}, target charge={charge_data[target_ion]}"
            )
            if (
                charge_data[comp] * charge_data[target_ion] > 0
            ):  # Same charge sign as target
                initial_mass = (
                    fresh_regen_state.conc_mass_phase_comp["Liq", comp].value
                    * regen_tank_vol
                )
                used_mass = (
                    ions_removed
                    * regenerant_stoich[comp]
                    * m.fs.properties.mw_comp[comp].value
                )
                spent_conc = max(0, (initial_mass - used_mass) / regen_tank_vol)
                print(
                    f"  initial_mass={initial_mass}, used_mass={used_mass}, spent_conc={spent_conc}"
                )
                spent_regen_state.conc_mass_phase_comp["Liq", comp].set_value(
                    spent_conc
                )
            else:
                print(
                    f"  Counterion (no change): spent_conc={fresh_regen_state.conc_mass_phase_comp['Liq', comp].value}"
                )
                spent_regen_state.conc_mass_phase_comp["Liq", comp].set_value(
                    fresh_regen_state.conc_mass_phase_comp["Liq", comp].value
                )
        else:
            spent_regen_state.conc_mass_phase_comp["Liq", comp].set_value(
                fresh_regen_state.conc_mass_phase_comp["Liq", comp].value
            )


def initialize_system(m):
    # First we initialize the Feed block using values set in set_operating_conditions
    m.fs.feed.initialize()

    # We then propagate the state of the Feed block to the ion exchange model...
    propagate_state(m.fs.feed_to_ix)

    ix = m.fs.ion_exchange

    # Initialize the ion exchange model
    m.fs.ion_exchange.initialize()

    # With the ion exchange model initialized, we have initial guesses for the Product and Regen blocks
    # and can propagate the state of the IX effluent and regeneration stream.
    propagate_state(m.fs.ix_to_product)
    propagate_state(m.fs.ix_to_spent_regen)

    # Finally, we initialize the product, regen and costing blocks.
    m.fs.product.initialize()
    m.fs.regen.initialize()
    m.fs.costing.initialize()


def optimize_system(m):
    # Example of optimizing number of IX columns based on desired effluent equivalent concentration

    # Adding an objective to model.
    # In this case, we want to optimze the model to minimize the LCOW.
    m.fs.obj = Objective(expr=m.fs.costing.LCOW)
    ix = m.fs.ion_exchange
    target_ion = m.fs.ion_exchange.config.target_ion

    # For this demo, we are optimizing the model to have an effluent concentration of 25 mg/L.
    # Our initial model resulted in an effluent concentration of 0.21 mg/L.
    # By increasing the effluent concentration, we will have a longer breakthrough time, which will lead to less regeneration solution used,
    # and (hopefully) a lower cost.
    ix.process_flow.properties_out[0].conc_mass_phase_comp["Liq", target_ion].fix(0.025)

    # With the new effluent conditions for our ion exchange model, this will have implications for our downstream models (the Product and Regen blocks)
    # Thus, we must re-propagate the new effluent state to these models...
    propagate_state(m.fs.ix_to_product)
    propagate_state(m.fs.ix_to_spent_regen)
    # ...and re-initialize them to our new conditions.
    m.fs.product.initialize()
    m.fs.regen.initialize()

    # To adjust solution to fixed-pattern to achieve desired effluent, must unfix dimensionless_time.
    ix.dimensionless_time.unfix()
    # Can optimize around different design variables, e.g., bed_depth, service_flow_rate (or combinations of these)
    # Here demonstrates optimization around column design
    ix.number_columns.unfix()
    ix.bed_depth.unfix()
    optimized_results = solver.solve(m)
    assert_optimal_termination(optimized_results)


def get_ion_config(ions, regenerant="NaCl"):

    if not isinstance(ions, (list, tuple)):
        ions = [ions]

    # Ion diffusivity, molecular weight, and charge
    diff_data = {
        "Na_+": 1.33e-9,
        "Ca_2+": 9.2e-10,
        "Cl_-": 2.03e-9,
        "Mg_2+": 0.706e-9,
        "SO4_2-": 1.06e-9,
        "NH4_+": 1.96e-9,
    }
    mw_data = {
        "Na_+": 23e-3,
        "Ca_2+": 40e-3,
        "Cl_-": 35e-3,
        "Mg_2+": 24e-3,
        "SO4_2-": 96e-3,
        "NH4_+": 18e-3,
    }
    charge_data = {
        "Na_+": 1,
        "Ca_2+": 2,
        "Cl_-": -1,
        "Mg_2+": 2,
        "SO4_2-": -2,
        "NH4_+": 1,
    }

    # Regenerant stoichiometry and molecular weight (as solid compound)
    regenerant_stoich_data = {
        "HCl": {
            "H_+": 1,
            "Cl_-": 1,
        },
        "NaOH": {
            "Na_+": 1,
            "OH_-": 1,
        },
        "H2SO4": {
            "H_+": 2,
            "SO4_2-": 1,
        },
        "NaCl": {
            "Na_+": 1,
            "Cl_-": 1,
        },
        "MeOH": {
            "CH3OH": 1,
        },
        "single_use": {},
    }

    regenerant_mw_data = {
        "HCl": 36.46,  # g/mol
        "NaOH": 40.00,  # g/mol
        "H2SO4": 98.08,  # g/mol
        "NaCl": 58.44,  # g/mol
        "MeOH": 32.04,  # g/mol
        "single_use": 0.0,  # g/mol
    }

    # prop_config is without regenerant data, to be passed to property package
    prop_config = {
        "solute_list": [],
        "diffusivity_data": {},
        "mw_data": {"H2O": 18e-3},
        "charge": {},
    }

    for ion in ions:
        prop_config["solute_list"].append(ion)
        prop_config["diffusivity_data"][("Liq", ion)] = diff_data[ion]
        prop_config["mw_data"][ion] = mw_data[ion]
        prop_config["charge"][ion] = charge_data[ion]

    # Add ions from regenerant dissolution to the model, if they are not already present
    regenerant_ions = list(regenerant_stoich_data[regenerant].keys())

    for ion in regenerant_ions:
        if ion not in ions and ion in diff_data:
            ions.append(ion)
            prop_config["solute_list"].append(ion)
            prop_config["diffusivity_data"][("Liq", ion)] = diff_data[ion]
            prop_config["mw_data"][ion] = mw_data[ion]
            prop_config["charge"][ion] = charge_data[ion]

    # Returns the prop_config for MCASParameterBlock as well as the regenerant stoich and mw data
    return {
        "prop_config": prop_config,
        "regenerant_stoich_data": regenerant_stoich_data,
        "regenerant_mw_data": regenerant_mw_data,
    }


def get_regenerant_stoichiometry(regenerant):
    regenerant_stoich_data = {
        "HCl": {"H_+": 1, "Cl_-": 1},
        "NaOH": {"Na_+": 1, "OH_-": 1},
        "H2SO4": {"H_+": 2, "SO4_2-": 1},
        "NaCl": {"Na_+": 1, "Cl_-": 1},
        "MeOH": {},  # No dissociation
        "single_use": {},  # No dissociation
    }

    return regenerant_stoich_data[regenerant]


def get_regenerant_mw(regenerant):
    regenerant_mw_data = {
        "HCl": 36.46e-3,
        "NaOH": 40.0e-3,
        "H2SO4": 98.08e-3,
        "NaCl": 58.44e-3,
        "MeOH": 32.04e-3,
        "single_use": 0,
    }

    return {regenerant: regenerant_mw_data[regenerant]}


def display_results(m):

    ix = m.fs.ion_exchange
    liq = "Liq"
    header = f'{"PARAM":<40s}{"VALUE":<40s}{"UNITS":<40s}\n'

    prop_in = ix.process_flow.properties_in[0]
    prop_out = ix.process_flow.properties_out[0]
    prop_regen = m.fs.regen.properties[0]

    recovery = prop_out.flow_vol_phase["Liq"]() / prop_in.flow_vol_phase["Liq"]()
    target_ion = ix.config.target_ion
    ion_set = ix.config.property_package.ion_set
    bv_to_regen = (ix.vel_bed() * ix.t_breakthru()) / ix.bed_depth()

    title = f'\n{"=======> SUMMARY <=======":^80}\n'
    print(title)
    print(header)
    print(f'{"LCOW":<40s}{f"{m.fs.costing.LCOW():<40.4f}"}{"$/m3":<40s}')
    print(
        f'{"TOTAL Capital Cost":<40s}{f"${ix.costing.capital_cost():<39,.2f}"}{"$":<40s}'
    )
    print(
        f'{"Specific Energy Consumption":<40s}{f"{m.fs.costing.specific_energy_consumption():<39,.5f}"}{"kWh/m3":<40s}'
    )
    print(
        f'{f"Annual Regenerant cost ({ix.config.regenerant})":<40s}{f"${m.fs.costing.aggregate_flow_costs[ix.config.regenerant]():<39,.2f}"}{"$/yr":<40s}'
    )
    print(f'{"BV Until Regen":<40s}{bv_to_regen:<40.3f}{"Bed Volumes":<40s}')
    print(
        f'{f"Breakthrough/Initial Conc. [{target_ion}]":<40s}{ix.c_norm[target_ion]():<40.3%}'
    )
    print(
        f'{"Vol. Flow In [m3/s]":<40s}{prop_in.flow_vol_phase[liq]():<40.5f}{"m3/s":<40s}'
    )
    print(
        f'{"Vol. Flow Out [m3/s]":<40s}{prop_out.flow_vol_phase[liq]():<40.5f}{"m3/s":<40s}'
    )
    print(f'{"Water Vol. Recovery":<40s}{recovery:<40.2%}{"%":<40s}')
    print(f'{"Breakthrough Time [hr]":<40s}{ix.t_breakthru() / 3600:<40.3f}{"hr":<40s}')
    print(f'{"Number Columns":<40s}{ix.number_columns():<40.2f}{"---":<40s}')
    print(f'{"Column Vol.":<40s}{ix.col_vol_per():<40.2f}{"m3":<40s}')
    print(f'{"Bed Depth":<40s}{ix.bed_depth():<40.2f}{"m":<40s}')
    for ion in ion_set:
        print(
            f'{f"Removal [{ion}]":<40s}{1 - prop_out.conc_mass_phase_comp[liq, ion]() / prop_in.conc_mass_phase_comp[liq, ion]():<40.4%}{"%":<40s}'
        )
        print(
            f'{f"Conc. In [{ion}, mg/L]":<40s}{pyunits.convert(prop_in.conc_mass_phase_comp[liq, ion], to_units=pyunits.mg/pyunits.L)():<40.3e}{"mg/L":<40s}'
        )
        print(
            f'{f"Conc. Out [{ion}, mg/L]":<40s}{pyunits.convert(prop_out.conc_mass_phase_comp[liq, ion], to_units=pyunits.mg/pyunits.L)():<40.3e}{"mg/L":<40s}'
        )
        print(
            f'{f"Regen Conc. [{ion}, mg/L]":<40s}{pyunits.convert(prop_regen.conc_mass_phase_comp[liq, ion], to_units=pyunits.mg/pyunits.L)():<40.3e}{"mg/L":<40s}'
        )
        print()


if __name__ == "__main__":
    m = main()
