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
from idaes.core.util.scaling import calculate_scaling_factors, set_scaling_factor
from idaes.core.util.initialization import propagate_state
from idaes.models.unit_models import Product, Feed
from idaes.core.util import DiagnosticsToolbox
from idaes.core import MaterialFlowBasis

from watertap.core.util.initialization import check_dof
from watertap.property_models.multicomp_aq_sol_prop_pack import MCASParameterBlock
from watertap.unit_models.ion_exchange_0D import IonExchange0D
from watertap.unit_models.genericNP_ZO import GenericNPZO
from watertap.costing import WaterTAPCosting
from watertap.core.solvers import get_solver

from idaes.core.util.tables import (
    create_stream_table_dataframe,
    stream_table_dataframe_to_string,
)

import math
from watertap.tools.plot_network import plot_network


solver = get_solver()


def main():
    # The IX model currently only supports one "target" ion (i.e., the component in the water source that can be removed by IX)
    # All other ions are inert. This demo does not contain inert ions, but an example can be found in the IX test file:
    # watertap/watertap/unit_models/tests/test_ion_exchange_0D.py
    target_ion = "NH4_+"  # UPDATED TARGET
    ions = ["NH4_+", "Na_+", "Cl_-"]  # ADDED NA+ AND CL- TO LIST TRACKED

    # See ix_build for details on building the model for this demo.
    m = ix_build(ions, target_ion)  # BUILD SPECIFYING NH4+
    # See set_operating_conditions for details on operating conditions for this demo.
    set_operating_conditions(m)
    # See initialize_system for details on initializing the models for this demo.
    initialize_system(m)
    # Check the degrees of freedom of the model to ensure it is zero.
    check_dof(m)

    # Check key variable values before solving
    print("\n=== Key Variable Values Before Solving ===")
    print(
        f"NH4+ concentration in feed: {value(m.fs.feed.properties[0].conc_mass_phase_comp['Liq', 'NH4_+'])} kg/m3"
    )
    print(
        f"NH4+ concentration after IX: {value(m.fs.ion_exchange.process_flow.properties_out[0].conc_mass_phase_comp['Liq', 'NH4_+'])} kg/m3"
    )
    print(
        f"NH4+ concentration in regen stream: {value(m.fs.ion_exchange.regeneration_stream[0].conc_mass_phase_comp['Liq', 'NH4_+'])} kg/m3"
    )
    print(
        f"NH4+ mass flow in: {value(m.fs.genericNP.properties_in[0].flow_mass_phase_comp['Liq', 'NH4_+'])} kg/s"
    )
    print(
        f"NH4+ mass flow in treated: {value(m.fs.genericNP.properties_treated[0].flow_mass_phase_comp['Liq', 'NH4_+'])} kg/s"
    )
    print(
        f"NH4+ mass flow in byproduct: {value(m.fs.genericNP.properties_byproduct[0].flow_mass_phase_comp['Liq', 'NH4_+'])} kg/s"
    )
    # print(f"Flow to electroNP: {value(m.fs.genericNP.inlet.flow_vol[0])} m3/s")

    # Solve the model. Store the results in a local variable.
    diagnostics = DiagnosticsToolbox(m)
    diagnostics.report_structural_issues()
    diagnostics.display_variables_at_or_outside_bounds()
    diagnostics.display_underconstrained_set()
    results = solver.solve(m)
    # Ensure the solve resulted in an optimal termination status.
    assert_optimal_termination(results)
    # Display the degrees of freedom, termination status, and performance metrics of the model.
    print(f"\nDOF = {degrees_of_freedom(m)}")
    print(f"Model solve {results.solver.termination_condition.swapcase()}")
    display_results(m)

    # # See optimize_system for details on optimizing this model for a specific condition.
    # optimize_system(m)
    # ix = m.fs.ion_exchange

    # # With our model optimized to new conditions in optimize_system,
    # # we can get the new number_columns and bed_depth and fix them in our model.
    # num_col = math.ceil(
    #     ix.number_columns()
    # )  # To eliminate fractional number of columns
    # bed_depth = ix.bed_depth()
    # ix.bed_depth.fix(bed_depth)
    # ix.number_columns.fix(num_col)
    # check_dof(m)
    # results = solver.solve(m)
    # assert_optimal_termination(results)
    # print(f"\nDOF = {degrees_of_freedom(m)}")
    # print(f"Model solve {results.solver.termination_condition.swapcase()}")
    # display_results(m)

    return m


import idaes.core.util.scaling as iscale


def ix_build(ions, target_ion=None, hazardous_waste=False, regenerant="NaCl"):

    if not target_ion:
        target_ion = ions[0]

    # Create the model and flowsheet
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # ion_config is the dictionary needed to configure the property package.
    # UPDATED - For this demo, we have properties related to the target ion (NH4_+), other ions (Na_+, Cl_-) and water (H2O)
    ion_props = get_ion_config(ions)

    # The water property package used for the ion exchange model is the multi-component aqueous solution (MCAS) property package
    # m.fs.properties =
    m.fs.properties = MCASParameterBlock(**ion_props)

    # Add the flowsheet level costing package
    m.fs.costing = WaterTAPCosting()

    # Add feed and product blocks to the flowsheet
    # These are the unit models on the flowsheet that the source water "flows" from/to
    # The must use the same property package as the ion exchange model
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.product = Product(property_package=m.fs.properties)
    # m.fs.regen = Product(property_package=m.fs.properties)

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
    }

    # Add the ion exchange model to the flowsheet
    m.fs.ion_exchange = ix = IonExchange0D(**ix_config)

    # DIFFERENT THAN IX - Add Generic Nutrient Recovery unit
    # Create a modified property parameter block with consistent units
    from pyomo.environ import units as pyunits

    # Configure GenericNPZO with appropriate unit handling
    m.fs.genericNP = GenericNPZO(
        property_package=m.fs.properties,
    )

    # Touch properties so they are available for scaling, initialization, and reporting.
    ix.process_flow.properties_in[0].conc_mass_phase_comp[...]
    ix.process_flow.properties_out[0].conc_mass_phase_comp[...]
    ix.process_flow.properties_in[0].flow_mol_phase_comp[...]
    ix.process_flow.properties_out[0].flow_mol_phase_comp[...]
    # ix.regeneration_stream[0].conc_mass_phase_comp[...]
    # ix.regeneration_stream[0].flow_mol_phase_comp[...]
    m.fs.feed.properties[0].flow_vol_phase[...]
    m.fs.feed.properties[0].flow_mol_phase_comp[...]  # ADDED
    m.fs.feed.properties[0].conc_mass_phase_comp[...]

    # Explicitly touch NH4_+ properties to ensure they're initialized
    ix.process_flow.properties_in[0].conc_mass_phase_comp["Liq", "NH4_+"]
    ix.process_flow.properties_out[0].conc_mass_phase_comp["Liq", "NH4_+"]
    ix.process_flow.properties_in[0].flow_mol_phase_comp["Liq", "NH4_+"]
    ix.process_flow.properties_out[0].flow_mol_phase_comp["Liq", "NH4_+"]
    ix.regeneration_stream[0].conc_mass_phase_comp["Liq", "NH4_+"]
    ix.regeneration_stream[0].flow_mol_phase_comp["Liq", "NH4_+"]
    m.fs.feed.properties[0].flow_mol_phase_comp["Liq", "NH4_+"]
    m.fs.feed.properties[0].conc_mass_phase_comp["Liq", "NH4_+"]

    # COPIED FROM ELECTRODIALYSIS_1STACK.PY
    # Fix state variables at the origin
    m.fs.feed.properties[0].pressure.fix(101325)  # feed pressure [Pa]
    # m.fs.product.properties[0].conc_mass_phase_comp[...]
    # m.fs.product.properties[0].flow_mol_phase_comp[...]

    # Add costing blocks to the flowsheet
    # Here, the ion exchange model has its own unit-level costing Block
    m.fs.ion_exchange.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing  # Indicating which flowsheet costing block to use to aggregate unit-level costs to the system-level costs
    )
    # DIFFERENT THAN IX - add costing block for the generic nutrient recovery unit
    m.fs.genericNP.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
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

    # DIFFERENT THAN IX - Connect regen directly to GenericNP
    m.fs.ix_regen_to_genericNP = Arc(source=ix.regen, destination=m.fs.genericNP.inlet)

    # Connect waste stream
    m.fs.waste = Product(property_package=m.fs.properties)
    m.fs.genericNP_to_waste = Arc(
        source=m.fs.genericNP.byproduct, destination=m.fs.waste.inlet
    )

    TransformationFactory("network.expand_arcs").apply_to(m)

    # Scaling variables in the model
    # Here, the molar flow for water ("flow_mol_phase_comp[Liq, H2O]") on the Feed block is scaled by 1e-4.
    # This is because the molar flow rate of water in this demo is ~2777 mol/s
    # and scaling factors are chosen such that the value of the variable multiplied by the scaling factor is ~1
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e-4, index=("Liq", "H2O")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 10, index=("Liq", target_ion)
    )
    # Call calculate_scaling_factors to apply scaling factors for each variable that we haven't set scaling factors for above.
    # iscale.set_scaling_factor(m.fs.genericNP.byproduct.flow_vol[0.0], 1e7)

    calculate_scaling_factors(m)

    return m


def set_operating_conditions(m, flow_in=0.05, conc_mass_in=0.0001, solver=None):
    if solver is None:
        solver = get_solver()
    ix = m.fs.ion_exchange
    target_ion = ix.config.target_ion

    # Initialize concentrations for all ions to zero
    conc_dict = {}
    for ion in m.fs.properties.component_list:
        if ion != "H2O":
            conc_dict[("conc_mass_phase_comp", ("Liq", ion))] = 0.0

    # Set the target ion concentration
    conc_dict[("conc_mass_phase_comp", ("Liq", target_ion))] = conc_mass_in  # kg/m3

    # Set the feed properties
    m.fs.feed.properties.calculate_state(
        var_args={
            ("flow_vol_phase", "Liq"): flow_in,  # m3/s
            **conc_dict,
            ("pressure", None): 101325,
            ("temperature", None): 298,
        },
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

    # DIFFERENT THAN IX - Set generic nutrient recovery parameters

    # Fix GenericNP unit vars
    # We need to manually convert from mass to molar basis for energy consumption
    # Get molecular weight of NH4+ from property package
    mw_NH4 = m.fs.properties.mw_comp["NH4_+"]
    # Convert 25 kWh/kg to kWh/mol, but strip units before passing to fix()
    energy_per_mol = 25 * mw_NH4
    energy_per_mol_value = value(energy_per_mol)  # Get the numeric value without units
    m.fs.genericNP.energy_electric_flow_mol["NH4_+"].fix(
        energy_per_mol_value
    )  # kWh/mol NH4+ removed
    # Convert MgCl2 dosage from kg/kg to mol/mol
    # Get molecular weight of MgCl2 (not directly in property package, so we calculate)
    mw_MgCl2 = 95.211e-3  # kg/mol
    # Convert 1.5 kg MgCl2 per kg nutrient to mol MgCl2 per mol nutrient
    dosage_mol_per_mol = 1.5 * (mw_NH4 / mw_MgCl2)
    m.fs.genericNP.magnesium_chloride_dosage_mol.fix(
        dosage_mol_per_mol
    )  # mol MgCl2 per mol nutrient removed

    # Set removal rates for target ion in GenericNP
    m.fs.genericNP.NH4_removal_mol.fix(0.5)  # 90% removal of ammonium
    m.fs.genericNP.NH4_removal_mass.fix(0.55)  # 90% removal of ammonium


def initialize_system(m):
    # First we initialize the Feed block using values set in set_operating_conditions
    m.fs.feed.initialize()

    # We then propagate the state of the Feed block to the ion exchange model...
    propagate_state(m.fs.feed_to_ix)
    # ... and then initialize the ion exchange model.
    m.fs.ion_exchange.initialize()
    # With the ion exchange model initialized, we have initial guesses for the Product and Regen blocks
    # and can propagate the state of the IX effluent and regeneration stream.
    propagate_state(m.fs.ix_to_product)

    # DIFFERENT THAN IX - propagate directly to GenericNP
    propagate_state(m.fs.ix_regen_to_genericNP)

    # Print regeneration stream properties before GenericNP
    print("\n=== Regeneration Stream Properties Before GenericNP ===")
    for ion in ["NH4_+", "Na_+", "Cl_-"]:
        print(
            f"Regen stream conc for {ion}: {value(m.fs.ion_exchange.regeneration_stream[0].conc_mass_phase_comp['Liq', ion])} kg/m3"
        )
        print(
            f"Regen stream flow for {ion}: {value(m.fs.ion_exchange.regeneration_stream[0].flow_mass_phase_comp['Liq', ion])} kg/s"
        )
    print(
        f"Regen stream volumetric flow: {value(m.fs.ion_exchange.regeneration_stream[0].flow_vol_phase['Liq'])} m3/s"
    )
    print("=======================================\n")

    # # Set the value for Na_+ and Cl_- to 8 in the regeneration stream if needed
    # m.fs.ion_exchange.regeneration_stream[0].conc_mass_phase_comp["Liq", "Na_+"].set_value(8)
    # m.fs.ion_exchange.regeneration_stream[0].conc_mass_phase_comp["Liq", "Cl_-"].set_value(8)

    # Initialize genericNP
    print(
        f"Degrees of freedom on genericNP before initialization: {degrees_of_freedom(m.fs.genericNP)}"
    )
    m.fs.genericNP.initialize()

    # Print genericNP inlet properties to verify flow is preserved
    print("\n=== GenericNP Inlet Properties ===")
    for ion in ["NH4_+", "Na_+", "Cl_-"]:
        print(
            f"GenericNP inlet conc for {ion}: {value(m.fs.genericNP.properties_in[0].conc_mass_phase_comp['Liq', ion])} kg/m3"
        )
        print(
            f"GenericNP inlet flow for {ion}: {value(m.fs.genericNP.properties_in[0].flow_mass_phase_comp['Liq', ion])} kg/s"
        )
    print(
        f"GenericNP inlet volumetric flow: {value(m.fs.genericNP.properties_in[0].flow_vol_phase['Liq'])} m3/s"
    )
    print("=======================================\n")

    propagate_state(m.fs.genericNP_to_waste)
    m.fs.product.initialize()
    m.fs.waste.initialize()
    m.fs.costing.initialize()


def get_ion_config(ions):
    diff_data = {
        "Na_+": 1.33e-9,
        "Ca_2+": 9.2e-10,
        "Cl_-": 2.03e-9,
        "Mg_2+": 0.706e-9,
        "SO4_2-": 1.06e-9,
        "NH4_+": 1.96e-9,  # ADDED, value from Claude3.5, need ref
    }
    mw_data = {
        "Na_+": 23e-3,
        "Ca_2+": 40e-3,
        "Cl_-": 35e-3,
        "Mg_2+": 24e-3,
        "SO4_2-": 96e-3,
        "NH4_+": 18e-3,  # ADDED, value from Claude3.5, need ref
    }
    charge_data = {
        "Na_+": 1,
        "Ca_2+": 2,
        "Cl_-": -1,
        "Mg_2+": 2,
        "SO4_2-": -2,
        "NH4_+": 1,
    }  # ADDED NH4_+

    ion_config = {
        "solute_list": ions,
        "diffusivity_data": {},
        "mw_data": {"H2O": 18e-3},
        "charge": {},
        "elec_mobility_data": {},
    }

    for ion in ions:
        ion_config["diffusivity_data"][("Liq", ion)] = diff_data[ion]
        ion_config["mw_data"][ion] = mw_data[ion]
        ion_config["charge"][ion] = charge_data[ion]
    return ion_config


def display_results(m):

    ix = m.fs.ion_exchange
    liq = "Liq"
    header = f'{"PARAM":<40s}{"VALUE":<40s}{"UNITS":<40s}\n'

    prop_in = ix.process_flow.properties_in[0]
    prop_out = ix.process_flow.properties_out[0]

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


if __name__ == "__main__":
    m = main()

    # Create a stream table with only the necessary streams
    stream_table = create_stream_table_dataframe(
        {
            "Feed": m.fs.feed.properties,
            "IX Outlet": m.fs.ion_exchange.process_flow.properties_out,
            "IX Regen": m.fs.ion_exchange.regeneration_stream,
            "GenericNP Treated": m.fs.genericNP.properties_treated,
            "GenericNP Byproduct": m.fs.genericNP.properties_byproduct,
            "Product": m.fs.product.properties,
            "Waste": m.fs.waste.properties,
        },
        time_point=0,
    )

    # Print all regen components and their properties
    regen_stream = m.fs.ion_exchange.regeneration_stream[0]
    print("\nComponent properties:")
    for phase in regen_stream.phase_list:
        print(f"\nPhase: {phase}")

        # Print concentration for each component
        for comp in regen_stream.params.component_list:
            try:
                conc = value(regen_stream.conc_mass_phase_comp[phase, comp])
                flow = value(regen_stream.flow_mass_phase_comp[phase, comp])
                print(f"  {comp}:")
                print(f"    Concentration: {conc} kg/m³")
                print(f"    Mass flow: {flow} kg/s")

                # Convert to mg/L for easier reading of low concentrations
                conc_mg_L = pyunits.convert(
                    regen_stream.conc_mass_phase_comp[phase, comp],
                    to_units=pyunits.mg / pyunits.L,
                )()
                print(f"    Concentration: {conc_mg_L} mg/L")
            except (KeyError, AttributeError) as e:
                print(f"  {comp}: Error accessing data - {str(e)}")

    # Print overall stream properties
    try:
        print("\nOverall stream properties:")
        print(f"  Total volumetric flow: {value(regen_stream.flow_vol)} m³/s")
        print(f"  Total mass flow: {value(regen_stream.flow_mass)} kg/s")
        print(f"  Temperature: {value(regen_stream.temperature)} K")
        print(f"  Pressure: {value(regen_stream.pressure)} Pa")
    except AttributeError as e:
        print(f"Error accessing overall properties: {str(e)}")

    print("=======================================\n")
    constituents_to_print = ["NH4_+", "Na_+", "Cl_-"]

    # Print stream table to debug
    print("\n=== Stream Table ===")
    print(f"Columns: {stream_table.columns}")
    print(f"Index: {stream_table.index}")
    print("Sample values:")
    for constituent in constituents_to_print:
        for column in stream_table.columns:
            # Find the row that contains the constituent
            for idx in stream_table.index:
                if constituent in idx:
                    print(
                        f"Found {constituent} in row: {idx}, column: {column}, value: {stream_table.loc[idx, column]}"
                    )

    # Create the network plot with the stream table
    column_mapping = {
        "Feed": "feed",
        "IX Outlet": "ion_exchange.process_flow.properties_out",  # Specify exact path
        "IX Regen": "ion_exchange.regeneration_stream",  # Specify exact path for regeneration stream
        "GenericNP Treated": "genericNP.properties_treated",
        "GenericNP Byproduct": "genericNP.properties_byproduct",
        "Product": "product",
        "Waste": "waste",
    }

    # Add performance metrics table to the bottom left corner
    # Get values from the model
    ix = m.fs.ion_exchange
    liq = "Liq"
    prop_in = ix.process_flow.properties_in[0]
    prop_out = ix.process_flow.properties_out[0]
    recovery = prop_out.flow_vol_phase["Liq"]() / prop_in.flow_vol_phase["Liq"]()
    target_ion = ix.config.target_ion
    ion_set = ix.config.property_package.ion_set
    bv_to_regen = (ix.vel_bed() * ix.t_breakthru()) / ix.bed_depth()

    metrics_text = (
        f"PARAM                                   VALUE                                   UNITS                                   \n\n"
        f"LCOW                                    {m.fs.costing.LCOW():<40.4f}$/m3                                    \n"
        f"TOTAL Capital Cost                      ${ix.costing.capital_cost():<39,.2f}$                                       \n"
        f"Specific Energy Consumption             {m.fs.costing.specific_energy_consumption():<39,.5f}kWh/m3                                  \n"
        f"Annual Regenerant cost ({ix.config.regenerant})           ${m.fs.costing.aggregate_flow_costs[ix.config.regenerant]():<39,.2f}$/yr                                    \n"
        f"BV Until Regen                          {bv_to_regen:<40.3f}Bed Volumes                             \n"
        f"Breakthrough/Initial Conc. [{target_ion}]      {ix.c_norm[target_ion]():<40.3%}\n"
        f"Vol. Flow In [m3/s]                     {prop_in.flow_vol_phase[liq]():<40.5f}m3/s                                    \n"
        f"Vol. Flow Out [m3/s]                    {prop_out.flow_vol_phase[liq]():<40.5f}m3/s                                    \n"
        f"Water Vol. Recovery                     {recovery:<40.2%}%                                       \n"
        f"Breakthrough Time [hr]                  {ix.t_breakthru() / 3600:<40.3f}hr                                      \n"
        f"Number Columns                          {ix.number_columns():<40.2f}---                                     \n"
        f"Column Vol.                             {ix.col_vol_per():<40.2f}m3                                      \n"
        f"Bed Depth                               {ix.bed_depth():<40.2f}m                                       \n"
    )

    # Use the standard plot_network function with metrics_text as table_values
    plot_network(
        m,
        stream_table,
        path_to_save="./ix_electroNP_ammonia_test.png",
        constituents_to_print=constituents_to_print,
        column_mapping=column_mapping,
        table_values=metrics_text,
        figsize=(12, 8),
    )
