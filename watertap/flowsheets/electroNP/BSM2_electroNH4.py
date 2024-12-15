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
"""
Flowsheet example full Water Resource Recovery Facility
(WRRF; a.k.a., wastewater treatment plant) with ASM2d and ADM1 with P extension.

The flowsheet follows the same formulation as benchmark simulation model no.2 (BSM2)
but comprises different specifications for default values than BSM2.
"""

# Some more information about this module
__author__ = "Chenyu Wang, Adam Atia, Alejandro Garciadiego, Marcus Holly"

import pyomo.environ as pyo
from pyomo.network import Arc, SequentialDecomposition

from idaes.core import (
    FlowsheetBlock,
    UnitModelCostingBlock,
)
from idaes.models.unit_models import (
    CSTR,
    Feed,
    Separator,
    Product,
    Mixer,
    PressureChanger,
)
from idaes.models.unit_models.separator import SplittingType
from watertap.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale
from idaes.core.util.tables import (
    create_stream_table_dataframe,
    stream_table_dataframe_to_string,
)
from watertap.unit_models.cstr_injection import CSTR_Injection
from watertap.unit_models.clarifier import Clarifier
from watertap.property_models.unit_specific.anaerobic_digestion.modified_adm1_properties import (
    ModifiedADM1ParameterBlock,
)
from watertap.property_models.unit_specific.anaerobic_digestion.adm1_properties_vapor import (
    ADM1_vaporParameterBlock,
)
from watertap.property_models.unit_specific.anaerobic_digestion.modified_adm1_reactions import (
    ModifiedADM1ReactionParameterBlock,
)
from watertap.property_models.unit_specific.activated_sludge.modified_asm2d_properties import (
    ModifiedASM2dParameterBlock,
)
from watertap.property_models.unit_specific.activated_sludge.modified_asm2d_reactions import (
    ModifiedASM2dReactionParameterBlock,
)
from watertap.unit_models.translators.translator_adm1_asm2d import (
    Translator_ADM1_ASM2D,
)
from idaes.models.unit_models.mixer import MomentumMixingType
from watertap.unit_models.translators.translator_asm2d_adm1 import (
    Translator_ASM2d_ADM1,
)
from watertap.unit_models.anaerobic_digester import AD
from watertap.unit_models.dewatering import (
    DewateringUnit,
    ActivatedSludgeModelType as dewater_type,
)
from watertap.unit_models.thickener import (
    Thickener,
    ActivatedSludgeModelType as thickener_type,
)
from watertap.core.util.initialization import check_solve
from watertap.unit_models.electroNP_ZO import ElectroNPZO

from watertap.costing import WaterTAPCosting
from watertap.costing.unit_models.clarifier import (
    cost_circular_clarifier,
    cost_primary_clarifier,
)

from plot_network import plot_network
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from idaes.core.util.model_diagnostics import DiagnosticsToolbox

# Set up logger
_log = idaeslog.getLogger(__name__)


def main(has_electroNP=False):
    m = build_flowsheet(has_electroNP=has_electroNP)
    add_costing(m)
    set_operating_conditions(m)

    for mx in m.fs.mixers:
        mx.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 3].deactivate()
    print(f"DOF before initialization: {degrees_of_freedom(m)}")

    db = DiagnosticsToolbox(m)
    db.report_structural_issues()
    db.display_variables_with_extreme_jacobians()
    db.display_variables_with_extreme_scaling_factors()

    # initialize_system(m, has_electroNP=has_electroNP)
    # db.report_numerical_issues()
    #for mx in m.fs.mixers:
        #mx.pressure_equality_constraints[0.0, 2].deactivate()
    #m.fs.MX3.pressure_equality_constraints[0.0, 2].deactivate()
    #m.fs.MX3.pressure_equality_constraints[0.0, 3].deactivate()
    #print(f"DOF after initialization: {degrees_of_freedom(m)}")
    
    # results = solve(m)

    pyo.assert_optimal_termination(results)
    check_solve(
        results,
        checkpoint="re-solve with controls in place",
        logger=_log,
        fail_flag=True,
    )
    return m, results

def build_flowsheet(has_electroNP=False):
    m = pyo.ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.has_electroNP = has_electroNP

    # Properties
    m.fs.props_ASM2D = ModifiedASM2dParameterBlock()
    m.fs.rxn_props_ASM2D = ModifiedASM2dReactionParameterBlock(
        property_package=m.fs.props_ASM2D
    )
    m.fs.props_ADM1 = ModifiedADM1ParameterBlock()
    m.fs.props_vap_ADM1 = ADM1_vaporParameterBlock()
    m.fs.rxn_props_ADM1 = ModifiedADM1ReactionParameterBlock(
        property_package=m.fs.props_ADM1
    )

    # Feed water stream
    m.fs.FeedWater = Feed(property_package=m.fs.props_ASM2D)

    # ====================================================================
    # Primary Clarifier
    m.fs.CL = Clarifier(
        property_package=m.fs.props_ASM2D,
        outlet_list=["underflow", "effluent"],
        split_basis=SplittingType.componentFlow,
    )

    # ======================================================================
    # Activated Sludge Process
    # Mixer for feed water and recycled sludge
    m.fs.MX1 = Mixer(
        property_package=m.fs.props_ASM2D,
        inlet_list=["feed_water", "recycle"],
        momentum_mixing_type=MomentumMixingType.equality,
    )
    # First reactor (anaerobic) - standard CSTR
    m.fs.R1 = CSTR(
        property_package=m.fs.props_ASM2D, reaction_package=m.fs.rxn_props_ASM2D
    )
    # Second reactor (anaerobic) - standard CSTR
    m.fs.R2 = CSTR(
        property_package=m.fs.props_ASM2D, reaction_package=m.fs.rxn_props_ASM2D
    )
    # Third reactor (anoxic) - standard CSTR
    m.fs.R3 = CSTR(
        property_package=m.fs.props_ASM2D, reaction_package=m.fs.rxn_props_ASM2D
    )
    # Fourth reactor (anoxic) - standard CSTR
    m.fs.R4 = CSTR(
        property_package=m.fs.props_ASM2D, reaction_package=m.fs.rxn_props_ASM2D
    )
    # Fifth reactor (aerobic) - CSTR with injection
    m.fs.R5 = CSTR_Injection(
        property_package=m.fs.props_ASM2D, reaction_package=m.fs.rxn_props_ASM2D
    )
    # Sixth reactor (aerobic) - CSTR with injection
    m.fs.R6 = CSTR_Injection(
        property_package=m.fs.props_ASM2D, reaction_package=m.fs.rxn_props_ASM2D
    )
    # Seventh reactor (aerobic) - CSTR with injection
    m.fs.R7 = CSTR_Injection(
        property_package=m.fs.props_ASM2D, reaction_package=m.fs.rxn_props_ASM2D
    )
    m.fs.SP1 = Separator(
        property_package=m.fs.props_ASM2D, outlet_list=["underflow", "overflow"]
    )
    # Secondary Clarifier
    # TODO: Replace with more detailed model when available
    m.fs.CL2 = Clarifier(
        property_package=m.fs.props_ASM2D,
        outlet_list=["underflow", "effluent"],
        split_basis=SplittingType.componentFlow,
    )
    # Mixing sludge recycle and R5 underflow
    m.fs.MX2 = Mixer(
        property_package=m.fs.props_ASM2D,
        inlet_list=["reactor", "clarifier"],
        momentum_mixing_type=MomentumMixingType.equality,
    )
    # Sludge separator
    m.fs.SP2 = Separator(
        property_package=m.fs.props_ASM2D, outlet_list=["waste", "recycle"]
    )
    # Recycle pressure changer - use a simple isothermal unit for now
    m.fs.P1 = PressureChanger(property_package=m.fs.props_ASM2D)

    # ======================================================================
    # Thickener
    m.fs.thickener = Thickener(
        property_package=m.fs.props_ASM2D,
        activated_sludge_model=thickener_type.modified_ASM2D,
    )
    # Mixing feed and recycle streams from thickener and dewatering unit
    m.fs.MX3 = Mixer(
        property_package=m.fs.props_ASM2D,
        inlet_list=["feed_water", "recycle1", "recycle2"],
        momentum_mixing_type=MomentumMixingType.equality,
    )
    # Mixing sludge from thickener and primary clarifier
    m.fs.MX4 = Mixer(
        property_package=m.fs.props_ASM2D,
        inlet_list=["thickener", "clarifier"],
        momentum_mixing_type=MomentumMixingType.equality,
    )

    # ======================================================================
    # Anaerobic digester section
    # ASM2d-ADM1 translator
    m.fs.translator_asm2d_adm1 = Translator_ASM2d_ADM1(
        inlet_property_package=m.fs.props_ASM2D,
        outlet_property_package=m.fs.props_ADM1,
        inlet_reaction_package=m.fs.rxn_props_ASM2D,
        outlet_reaction_package=m.fs.rxn_props_ADM1,
        has_phase_equilibrium=False,
        outlet_state_defined=True,
        bio_P=False,
    )

    # Anaerobic digester
    m.fs.AD = AD(
        liquid_property_package=m.fs.props_ADM1,
        vapor_property_package=m.fs.props_vap_ADM1,
        reaction_package=m.fs.rxn_props_ADM1,
        has_heat_transfer=True,
        has_pressure_change=False,
    )

    # ADM1-ASM2d translator
    m.fs.translator_adm1_asm2d = Translator_ADM1_ASM2D(
        inlet_property_package=m.fs.props_ADM1,
        outlet_property_package=m.fs.props_ASM2D,
        # inlet_reaction_package=m.fs.rxn_props_ADM1,
        # outlet_reaction_package=m.fs.rxn_props_ASM2D,
        has_phase_equilibrium=False,
        outlet_state_defined=True,
    )

    # ======================================================================
    # Dewatering Unit
    m.fs.dewater = DewateringUnit(
        property_package=m.fs.props_ASM2D,
        activated_sludge_model=dewater_type.modified_ASM2D,
    )

    # ======================================================================
    # ElectroN-P
    if has_electroNP is True:
        m.fs.electroNP = ElectroNPZO(property_package=m.fs.props_ASM2D) # could also add component set, as a dict. or make list dependent on property package

    # ======================================================================
    # Product Blocks
    m.fs.Treated = Product(property_package=m.fs.props_ASM2D)
    m.fs.Sludge = Product(property_package=m.fs.props_ASM2D)
    # Mixers
    m.fs.mixers = (m.fs.MX1, m.fs.MX2, m.fs.MX4)

    # ======================================================================
    # Link units related to ASM section
    m.fs.stream2 = Arc(source=m.fs.MX1.outlet, destination=m.fs.R1.inlet)
    m.fs.stream3 = Arc(source=m.fs.R1.outlet, destination=m.fs.R2.inlet)
    m.fs.stream4 = Arc(source=m.fs.R2.outlet, destination=m.fs.MX2.reactor)
    m.fs.stream5 = Arc(source=m.fs.MX2.outlet, destination=m.fs.R3.inlet)
    m.fs.stream6 = Arc(source=m.fs.R3.outlet, destination=m.fs.R4.inlet)
    m.fs.stream7 = Arc(source=m.fs.R4.outlet, destination=m.fs.R5.inlet)
    m.fs.stream8 = Arc(source=m.fs.R5.outlet, destination=m.fs.R6.inlet)
    m.fs.stream9 = Arc(source=m.fs.R6.outlet, destination=m.fs.R7.inlet)
    m.fs.stream10 = Arc(source=m.fs.R7.outlet, destination=m.fs.SP1.inlet)
    m.fs.stream11 = Arc(source=m.fs.SP1.overflow, destination=m.fs.CL2.inlet)
    m.fs.stream12 = Arc(source=m.fs.SP1.underflow, destination=m.fs.MX2.clarifier)
    m.fs.stream13 = Arc(source=m.fs.CL2.effluent, destination=m.fs.Treated.inlet)
    m.fs.stream14 = Arc(source=m.fs.CL2.underflow, destination=m.fs.SP2.inlet)
    m.fs.stream15 = Arc(source=m.fs.SP2.recycle, destination=m.fs.P1.inlet)
    m.fs.stream16 = Arc(source=m.fs.P1.outlet, destination=m.fs.MX1.recycle)
    # m.fs.stream17 = Arc(source=m.fs.SP2.waste, destination=m.fs.Sludge.inlet)

    # Link units related to AD section
    m.fs.stream_AD_translator = Arc(
        source=m.fs.AD.liquid_outlet, destination=m.fs.translator_adm1_asm2d.inlet
    )
    m.fs.stream_SP_thickener = Arc(
        source=m.fs.SP2.waste, destination=m.fs.thickener.inlet
    )
    m.fs.stream3adm = Arc(
        source=m.fs.thickener.underflow, destination=m.fs.MX4.thickener
    )
    m.fs.stream7adm = Arc(source=m.fs.thickener.overflow, destination=m.fs.MX3.recycle2)
    m.fs.stream9adm = Arc(source=m.fs.CL.underflow, destination=m.fs.MX4.clarifier)
    m.fs.stream_translator_dewater = Arc(
        source=m.fs.translator_adm1_asm2d.outlet, destination=m.fs.dewater.inlet
    )
    m.fs.stream1a = Arc(source=m.fs.FeedWater.outlet, destination=m.fs.MX3.feed_water)
    m.fs.stream1b = Arc(source=m.fs.MX3.outlet, destination=m.fs.CL.inlet)
    m.fs.stream1c = Arc(source=m.fs.CL.effluent, destination=m.fs.MX1.feed_water)
    m.fs.stream_dewater_sludge = Arc(
        source=m.fs.dewater.underflow, destination=m.fs.Sludge.inlet
    )
    if has_electroNP is True:
        m.fs.stream_dewater_electroNP = Arc(
            source=m.fs.dewater.overflow, destination=m.fs.electroNP.inlet
        )
        m.fs.stream_electroNP_mixer = Arc(
            source=m.fs.electroNP.treated, destination=m.fs.MX3.recycle1
        )
    else:
        m.fs.stream_dewater_mixer = Arc(
            source=m.fs.dewater.overflow, destination=m.fs.MX3.recycle1
        )
    m.fs.stream10adm = Arc(
        source=m.fs.MX4.outlet, destination=m.fs.translator_asm2d_adm1.inlet
    )
    m.fs.stream_translator_AD = Arc(
        source=m.fs.translator_asm2d_adm1.outlet, destination=m.fs.AD.inlet
    )

    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    # Oxygen concentration in reactors 3 and 4 is governed by mass transfer
    # Add additional parameter and constraints
    m.fs.R5.KLa = pyo.Var(
        initialize=240,
        units=pyo.units.hour**-1,
        doc="Lumped mass transfer coefficient for oxygen",
    )
    m.fs.R6.KLa = pyo.Var(
        initialize=240,
        units=pyo.units.hour**-1,
        doc="Lumped mass transfer coefficient for oxygen",
    )
    m.fs.R7.KLa = pyo.Var(
        initialize=84,
        units=pyo.units.hour**-1,
        doc="Lumped mass transfer coefficient for oxygen",
    )
    m.fs.S_O_eq = pyo.Param(
        default=8e-3,
        units=pyo.units.kg / pyo.units.m**3,
        mutable=True,
        doc="Dissolved oxygen concentration at equilibrium",
    )

    m.fs.aerobic_reactors = (m.fs.R5, m.fs.R6, m.fs.R7)
    for R in m.fs.aerobic_reactors:
        iscale.set_scaling_factor(R.KLa, 1e-2)
        iscale.set_scaling_factor(R.hydraulic_retention_time[0], 1e-3)

    @m.fs.R5.Constraint(m.fs.time, doc="Mass transfer constraint for R3")
    def mass_transfer_R5(self, t):
        return pyo.units.convert(
            m.fs.R5.injection[t, "Liq", "S_O2"], to_units=pyo.units.kg / pyo.units.hour
        ) == (
            m.fs.R5.KLa
            * m.fs.R5.volume[t]
            * (m.fs.S_O_eq - m.fs.R5.outlet.conc_mass_comp[t, "S_O2"])
        )

    @m.fs.R6.Constraint(m.fs.time, doc="Mass transfer constraint for R4")
    def mass_transfer_R6(self, t):
        return pyo.units.convert(
            m.fs.R6.injection[t, "Liq", "S_O2"], to_units=pyo.units.kg / pyo.units.hour
        ) == (
            m.fs.R6.KLa
            * m.fs.R6.volume[t]
            * (m.fs.S_O_eq - m.fs.R6.outlet.conc_mass_comp[t, "S_O2"])
        )

    @m.fs.R7.Constraint(m.fs.time, doc="Mass transfer constraint for R4")
    def mass_transfer_R7(self, t):
        return pyo.units.convert(
            m.fs.R7.injection[t, "Liq", "S_O2"], to_units=pyo.units.kg / pyo.units.hour
        ) == (
            m.fs.R7.KLa
            * m.fs.R7.volume[t]
            * (m.fs.S_O_eq - m.fs.R7.outlet.conc_mass_comp[t, "S_O2"])
        )

    return m


def set_operating_conditions(m):
    # Feed Water Conditions
    print(f"DOF before feed: {degrees_of_freedom(m)}")
    m.fs.FeedWater.flow_vol.fix(20935.15 * pyo.units.m**3 / pyo.units.day)
    m.fs.FeedWater.temperature.fix(308.15 * pyo.units.K)
    m.fs.FeedWater.pressure.fix(1 * pyo.units.atm)
    m.fs.FeedWater.conc_mass_comp[0, "S_O2"].fix(1e-6 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_F"].fix(1e-6 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_A"].fix(70 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_NH4"].fix(26.6 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_NO3"].fix(1e-6 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_PO4"].fix(1e-6 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_I"].fix(57.45 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_N2"].fix(25.19 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "X_I"].fix(84 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "X_S"].fix(94.1 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "X_H"].fix(370 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "X_PAO"].fix(
        51.5262 * pyo.units.g / pyo.units.m**3
    )
    m.fs.FeedWater.conc_mass_comp[0, "X_PP"].fix(1e-6 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "X_PHA"].fix(1e-6 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "X_AUT"].fix(1e-6 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_IC"].fix(5.652 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_K"].fix(374.6925 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_Mg"].fix(20 * pyo.units.g / pyo.units.m**3)

    # Primary Clarifier
    # TODO: Update primary clarifier once more detailed model available
    m.fs.CL.split_fraction[0, "effluent", "H2O"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_A"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_F"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_I"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_N2"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_NH4"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_NO3"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_O2"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_PO4"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_IC"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_K"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_Mg"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "X_AUT"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "X_H"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "X_I"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "X_PAO"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "X_PHA"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "X_PP"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "X_S"].fix(0.5192)

    # Reactor sizing
    m.fs.R1.volume.fix(1000 * pyo.units.m**3)
    m.fs.R2.volume.fix(1000 * pyo.units.m**3)
    m.fs.R3.volume.fix(1500 * pyo.units.m**3)
    m.fs.R4.volume.fix(1500 * pyo.units.m**3)
    m.fs.R5.volume.fix(3000 * pyo.units.m**3)
    m.fs.R6.volume.fix(3000 * pyo.units.m**3)
    m.fs.R7.volume.fix(3000 * pyo.units.m**3)

    # Injection rates to Reactions 5, 6 and 7
    for j in m.fs.props_ASM2D.component_list:
        if j != "S_O2":
            # All components except S_O have no injection
            m.fs.R5.injection[:, :, j].fix(0)
            m.fs.R6.injection[:, :, j].fix(0)
            m.fs.R7.injection[:, :, j].fix(0)
    # Then set injections rates for O2
    m.fs.R5.outlet.conc_mass_comp[:, "S_O2"].fix(1.91e-3)
    m.fs.R6.outlet.conc_mass_comp[:, "S_O2"].fix(2.60e-3)
    m.fs.R7.outlet.conc_mass_comp[:, "S_O2"].fix(3.20e-3)

    # Set fraction of outflow from reactor 5 that goes to recycle
    m.fs.SP1.split_fraction[:, "underflow"].fix(0.60)

    # Secondary Clarifier
    # TODO: Update once more detailed model available
    m.fs.CL2.split_fraction[0, "effluent", "H2O"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_A"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_F"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_I"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_N2"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_NH4"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_NO3"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_O2"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_PO4"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_IC"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_K"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_Mg"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "X_AUT"].fix(0.00187)
    m.fs.CL2.split_fraction[0, "effluent", "X_H"].fix(0.00187)
    m.fs.CL2.split_fraction[0, "effluent", "X_I"].fix(0.00187)
    m.fs.CL2.split_fraction[0, "effluent", "X_PAO"].fix(0.00187)
    m.fs.CL2.split_fraction[0, "effluent", "X_PHA"].fix(0.00187)
    m.fs.CL2.split_fraction[0, "effluent", "X_PP"].fix(0.00187)
    m.fs.CL2.split_fraction[0, "effluent", "X_S"].fix(0.00187)

    m.fs.CL2.surface_area.fix(1500 * pyo.units.m**2)

    # Sludge purge separator
    m.fs.SP2.split_fraction[:, "recycle"].fix(0.985)

    # Outlet pressure from recycle pump
    m.fs.P1.outlet.pressure.fix(101325)

    # AD
    m.fs.AD.volume_liquid.fix(3400)
    m.fs.AD.volume_vapor.fix(300)
    m.fs.AD.liquid_outlet.temperature.fix(308.15)

    # Dewatering Unit - fix either HRT or volume.
    m.fs.dewater.hydraulic_retention_time.fix(1800 * pyo.units.s)

    # Thickener unit
    m.fs.thickener.hydraulic_retention_time.fix(86400 * pyo.units.s)
    m.fs.thickener.diameter.fix(10 * pyo.units.m)

    # ElectroNP
    if m.fs.has_electroNP is True:
        m.fs.electroNP.energy_electric_flow_mass.fix(
            0.4 * pyo.units.kWh / pyo.units.kg
        )
        m.fs.electroNP.magnesium_chloride_dosage.fix(0.388)
        P_removal = 0.1
        NH4_removal = 0.5
        m.fs.electroNP.P_removal = P_removal
        m.fs.electroNP.NH4_removal = NH4_removal
        m.fs.electroNP.N_removal = 0.1
        m.fs.electroNP.frac_mass_H2O_treated[0].fix(0.99)

    def scale_variables(m):
        for var in m.fs.component_data_objects(pyo.Var, descend_into=True):
            if "flow_vol" in var.name:
                if "thickener" in var.parent_component().name or "AD" in var.parent_component().name:
                    print(var.name)
                    print(var.value)
                    iscale.set_scaling_factor(var, 0)
                else:
                    iscale.set_scaling_factor(var, 1e-4)
            if "temperature" in var.name:
                iscale.set_scaling_factor(var, 1e-2)
            if "pressure" in var.name and not "pressure_sat" in var.name:
                iscale.set_scaling_factor(var, 1e-4)
            if "conc_mass_comp" in var.name:
                iscale.set_scaling_factor(var, 1e1)
            # if "work" in var.name:
            #     iscale.set_scaling_factor(var, 1e6)
            # if "deltaP" in var.name:
            #     iscale.set_scaling_factor(var, 1e5)
            # if "heat" in var.name:
            #     iscale.set_scaling_factor(var, 1e9)
            # if "reaction_rate" in var.name:
            #     iscale.set_scaling_factor(var, 1e-4)
            # if "rate_reaction_extent" in var.name:
            #     iscale.set_scaling_factor(var, 1e-4)
            if "rate_reaction_generation" in var.name:
                iscale.set_scaling_factor(var, 1e-5)
                
    for unit in ("R1", "R2", "R3", "R4", "R5", "R6", "R7"):
        block = getattr(m.fs, unit)
        iscale.set_scaling_factor(
            block.control_volume.reactions[0.0].rate_expression, 1e3
        )
        iscale.set_scaling_factor(block.cstr_performance_eqn, 1e3)
        iscale.set_scaling_factor(
            block.control_volume.rate_reaction_stoichiometry_constraint, 1e3
        )
        iscale.set_scaling_factor(block.control_volume.material_balances, 1e3)

    # Apply scaling
    scale_variables(m)
    iscale.calculate_scaling_factors(m)


def initialize_system(m, has_electroNP=False):
    # Initialize flowsheet
    # Apply sequential decomposition - 1 iteration should suffice
    seq = SequentialDecomposition()
    seq.options.tear_method = "Direct"
    seq.options.iterLim = 1
    seq.options.tear_set = [m.fs.stream5, m.fs.stream10adm]

    G = seq.create_graph(m)
    # Uncomment this code to see tear set and initialization order
    order = seq.calculation_order(G)
    print("Initialization Order")
    for o in order:
        print(o[0].name)

    if has_electroNP:
        # P_removal = 0.65 - 0.95
        tear_guesses = {
            "flow_vol": {0: 1.2366},
            "conc_mass_comp": {
                (0, "S_A"): 0.0006,
                (0, "S_F"): 0.0004,
                (0, "S_I"): 0.057,
                (0, "S_N2"): 0.04,
                (0, "S_NH4"): 0.006,
                (0, "S_NO3"): 0.002,
                (0, "S_O2"): 0.0019,
                (0, "S_PO4"): 0.09,
                (0, "S_K"): 0.37,
                (0, "S_Mg"): 0.020,
                (0, "S_IC"): 0.13,
                (0, "X_AUT"): 0.085,
                (0, "X_H"): 3.5,
                (0, "X_I"): 3.1,
                (0, "X_PAO"): 3.4,
                (0, "X_PHA"): 0.087,
                (0, "X_PP"): 1.1,
                (0, "X_S"): 0.057,
            },
            "temperature": {0: 308.15},
            "pressure": {0: 101325},
        }

        tear_guesses2 = {
            "flow_vol": {0: 0.003},
            "conc_mass_comp": {
                (0, "S_A"): 0.1,
                (0, "S_F"): 0.15,
                (0, "S_I"): 0.057,
                (0, "S_N2"): 0.034,
                (0, "S_NH4"): 0.025,
                (0, "S_NO3"): 0.0015,
                (0, "S_O2"): 0.0013,
                (0, "S_PO4"): 0.1,
                (0, "S_K"): 0.38,
                (0, "S_Mg"): 0.024,
                (0, "S_IC"): 0.074,
                (0, "X_AUT"): 0.21,
                (0, "X_H"): 23,
                (0, "X_I"): 11,
                (0, "X_PAO"): 10.5,
                (0, "X_PHA"): 0.006,
                (0, "X_PP"): 2.7,
                (0, "X_S"): 3.9,
            },
            "temperature": {0: 308.15},
            "pressure": {0: 101325},
        }

    else:
        tear_guesses = {
            "flow_vol": {0: 1.2368},
            "conc_mass_comp": {
                (0, "S_A"): 0.0006,
                (0, "S_F"): 0.0004,
                (0, "S_I"): 0.057,
                (0, "S_N2"): 0.047,
                (0, "S_NH4"): 0.0075,
                (0, "S_NO3"): 0.003,
                (0, "S_O2"): 0.0019,
                (0, "S_PO4"): 0.73,
                (0, "S_K"): 0.37,
                (0, "S_Mg"): 0.020,
                (0, "S_IC"): 0.13,
                (0, "X_AUT"): 0.11,
                (0, "X_H"): 3.5,
                (0, "X_I"): 3.2,
                (0, "X_PAO"): 3.2,
                (0, "X_PHA"): 0.084,
                (0, "X_PP"): 1.07,
                (0, "X_S"): 0.057,
            },
            "temperature": {0: 308.15},
            "pressure": {0: 101325},
        }

        tear_guesses2 = {
            "flow_vol": {0: 0.003},
            "conc_mass_comp": {
                (0, "S_A"): 0.097,
                (0, "S_F"): 0.15,
                (0, "S_I"): 0.057,
                (0, "S_N2"): 0.036,
                (0, "S_NH4"): 0.03,
                (0, "S_NO3"): 0.002,
                (0, "S_O2"): 0.0013,
                (0, "S_PO4"): 0.74,
                (0, "S_K"): 0.38,
                (0, "S_Mg"): 0.024,
                (0, "S_IC"): 0.075,
                (0, "X_AUT"): 0.28,
                (0, "X_H"): 23.4,
                (0, "X_I"): 11.4,
                (0, "X_PAO"): 10.1,
                (0, "X_PHA"): 0.0044,
                (0, "X_PP"): 2.7,
                (0, "X_S"): 3.9,
            },
            "temperature": {0: 308.15},
            "pressure": {0: 101325},
        }

    # Pass the tear_guess to the SD tool
    seq.set_guesses_for(m.fs.R3.inlet, tear_guesses)
    seq.set_guesses_for(m.fs.translator_asm2d_adm1.inlet, tear_guesses2)

    def function(unit):
        unit.initialize(outlvl=idaeslog.INFO, solver="ipopt-watertap")

    seq.run(m, function)

def add_costing(m):
    """Add costing block"""
    m.fs.costing = WaterTAPCosting()

    m.fs.costing.base_currency = pyo.units.USD_2020

    # Costing Blocks
    m.fs.R1.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R2.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R3.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R4.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R5.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.CL.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=cost_primary_clarifier,
    )

    m.fs.CL2.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=cost_circular_clarifier,
    )

    # m.fs.RADM.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    # m.fs.DU.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    # m.fs.TU.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)

    # TODO: Leaving out mixer costs; consider including later

    # process costing and add system level metrics

    if hasattr(m.fs, 'electroNP'):
        m.fs.electroNP.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
        m.fs.costing.electroNP.phosphorus_recovery_value = 0.1 # TODO: add NH4 removal value
        m.fs.costing.electroNP.ammonia_recovery_value = 0.1

    m.fs.costing.cost_process()
    m.fs.costing.add_electricity_intensity(m.fs.FeedWater.properties[0].flow_vol)
    m.fs.costing.add_annual_water_production(m.fs.Treated.properties[0].flow_vol)
    m.fs.costing.add_LCOW(m.fs.FeedWater.properties[0].flow_vol)
    m.fs.costing.add_specific_energy_consumption(m.fs.FeedWater.properties[0].flow_vol)

    # m.fs.objective = pyo.Objective(expr=m.fs.costing.LCOW)
    iscale.set_scaling_factor(m.fs.costing.LCOW, 1e3)
    iscale.set_scaling_factor(m.fs.costing.total_capital_cost, 1e-7)
    iscale.set_scaling_factor(m.fs.costing.total_capital_cost, 1e-5)

def solve(m, solver=None):
    if solver is None:
        solver = get_solver()
    results = solver.solve(m, tee=True)
    check_solve(results, checkpoint="closing recycle", logger=_log, fail_flag=True)
    pyo.assert_optimal_termination(results)
    # add_costing(m) # add costing after solve
    return results


if __name__ == "__main__":
    # This method builds and runs a steady state activated sludge flowsheet.
    m, results = main(has_electroNP=True)
    if m.fs.has_electroNP is False:
        stream_table = create_stream_table_dataframe(
            {
                "Feed": m.fs.FeedWater.outlet,
                # "R3 inlet": m.fs.R3.inlet,
                # "ASM-ADM translator inlet": m.fs.translator_asm2d_adm1.inlet,
                "R1": m.fs.R1.outlet,
                "R2": m.fs.R2.outlet,
                "R3": m.fs.R3.outlet,
                "R4": m.fs.R4.outlet,
                "R5": m.fs.R5.outlet,
                "R6": m.fs.R6.outlet,
                "R7": m.fs.R7.outlet,
                "thickener outlet": m.fs.thickener.underflow,
                "ADM-ASM translator outlet": m.fs.translator_adm1_asm2d.outlet,
                "dewater outlet": m.fs.dewater.overflow,
                "Treated water": m.fs.Treated.inlet,
                "Sludge": m.fs.Sludge.inlet,
            },
            time_point=0,
        )
    else:
        stream_table = create_stream_table_dataframe(
            {
                "Feed": m.fs.FeedWater.outlet,
                "R3 inlet": m.fs.R3.inlet,
                "ASM-ADM translator inlet": m.fs.translator_asm2d_adm1.inlet,
                "R1": m.fs.R1.outlet,
                "R2": m.fs.R2.outlet,
                "R3": m.fs.R3.outlet,
                "R4": m.fs.R4.outlet,
                "R5": m.fs.R5.outlet,
                "R6": m.fs.R6.outlet,
                "R7": m.fs.R7.outlet,
                "thickener outlet": m.fs.thickener.underflow,
                "ADM-ASM translator outlet": m.fs.translator_adm1_asm2d.outlet,
                "dewater outlet": m.fs.dewater.overflow,
                "electroNP treated": m.fs.electroNP.treated,
                "electroNP byproduct": m.fs.electroNP.byproduct,
                "Treated water": m.fs.Treated.inlet,
                "Sludge": m.fs.Sludge.inlet,
            },
            time_point=0,
            
        )
    # print(stream_table_dataframe_to_string(stream_table))

#     plot_network(m, stream_table, path_to_save="BSM2_electroNP_flowsheet.png")
from parameter_sweep import (
    LinearSample,
    parameter_sweep,
)

def build_model(**kwargs):
    # return main(has_electroNP=has_electroNP)[0]
    m = build_flowsheet(has_electroNP=True)
    set_operating_conditions(m)
    for mx in m.fs.mixers:
        mx.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 3].deactivate()
    print(f"DOF before initialization: {degrees_of_freedom(m)}")

    initialize_system(m, has_electroNP=True)
    for mx in m.fs.mixers:
        mx.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 3].deactivate()
    print(f"DOF after initialization: {degrees_of_freedom(m)}")

    add_costing(m) # add costing after solve

    return m

def build_sweep_params(model, nx=1, **kwargs):
    sweep_params = {}
    sweep_params["N removal"] = LinearSample(
        model.fs.electroNP.N_removal, 0.1, 0.2, nx
    )
    sweep_params["N removal intensity"] = LinearSample(
        model.fs.electroNP.energy_electric_flow_mass, 0.4, 0.6, nx
    )
    return sweep_params

def build_outputs(model, **kwargs):
    outputs = {}
    outputs["Electricity Intensity"] = model.fs.costing.electricity_intensity
    outputs["Treated Water Flow"] = model.fs.Treated.flow_vol[0]
    outputs["Effluent NH4 Concentration"] = model.fs.Treated.conc_mass_comp[0, "S_NH4"]
    return outputs

def reinitialize_function(model):
    initialize_system(model, has_electroNP=True)

def run_analysis(case_num=1, interpolate_nan_outputs=True, output_filename=None):
    if output_filename is None:
        output_filename = f"sensitivity_{case_num}"

    global_results = parameter_sweep(
        build_model,
        build_sweep_params,
        build_outputs,
        csv_results_file_name=f"{output_filename}.csv",
        h5_results_file_name=f"{output_filename}.h5",
        optimize_function=solve,
        # reinitialize_function=reinitialize_function,
        interpolate_nan_outputs=interpolate_nan_outputs,
    )

    return global_results

run_parameter_sweep = False
if run_parameter_sweep:
    results = run_analysis()

    # create dataframe of results
    df_results = pd.DataFrame()
    # df_results["Feed Flow (m3/d)"] = results[1]["sweep_params"]["feed_flow"]["value"]
    df_results["N removal"] = results[1]["sweep_params"]["N removal"]["value"]
    df_results["N removal intensity"] = results[1]["sweep_params"]["N removal intensity"]["value"]
    df_results["Electricity Intensity"] = results[1]["outputs"]["Electricity Intensity"]["value"]

    pivot_df = df_results.pivot(index="N removal", columns="N removal intensity", values="Electricity Intensity")
    pivot_df = pivot_df.round(2)
    # round index and column names to 2 decimal
    pivot_df.index = pivot_df.index.round(2)
    pivot_df.columns = pivot_df.columns.round(2)

    heatmap = sns.heatmap(pivot_df, annot=True, cmap='YlOrBr', cbar_kws={'label': 'Plant-Wide Electricity Intensity (kWh/m3)'})
    plt.xlabel("N removal (%) from Anaerobic Digestate", fontsize=12)
    plt.ylabel("Energy intensity of N removal (kWh/kg N)", fontsize=12)
    plt.savefig("sensitivity_heatmap.png", dpi=300)
    plt.show()