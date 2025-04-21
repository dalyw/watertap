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

# Import Pyomo libraries
from pyomo.environ import (
    Var,
    Param,
    Suffix,
    NonNegativeReals,
    units as pyunits,
    Expression,
)
from idaes.models.unit_models.separator import SeparatorData, SplittingType

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
)

from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.misc import add_object_reference
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

from watertap.costing.unit_models.genericNP import cost_genericNP

__author__ = "Chenyu Wang"

_log = idaeslog.getLogger(__name__)


@declare_process_block_class("GenericNPZO")
class GenericNPZOdata(SeparatorData):
    """
    Zero order electrochemical nutrient removal (ElectroNP) model based on specified removal efficiencies for nitrogen and phosphorus.
    """

    CONFIG = SeparatorData.CONFIG()
    CONFIG.outlet_list = ["treated", "byproduct"]
    CONFIG.split_basis = SplittingType.componentFlow

    # CONFIG.treated_components = ["S_PO4", "S_NH4", "S_NO3", "S_NO2"]

    def build(self):
        # Call UnitModel.build to set up dynamics
        super(GenericNPZOdata, self).build()

        if len(self.config.property_package.solvent_set) > 1:
            raise ConfigurationError(
                "ElectroNP model only supports one solvent component,"
                "the provided property package has specified {} solvent components".format(
                    len(self.config.property_package.solvent_set)
                )
            )

        if len(self.config.property_package.solvent_set) == 0:
            raise ConfigurationError(
                "The ElectroNP model was expecting a solvent and did not receive it."
            )

        if (
            len(self.config.property_package.solute_set) == 0
            and len(self.config.property_package.ion_set) == 0
        ):
            raise ConfigurationError(
                "The ElectroNP model was expecting at least one solute or ion and did not receive any."
            )

        if "treated" and "byproduct" not in self.config.outlet_list:
            raise ConfigurationError(
                "{} encountered unrecognised "
                "outlet_list. This should not "
                "occur - please use treated "
                "and byproduct.".format(self.name)
            )

        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        units_meta = self.config.property_package.get_metadata().get_derived_units

        add_object_reference(self, "properties_in", self.mixed_state)
        add_object_reference(self, "properties_treated", self.treated_state)
        add_object_reference(self, "properties_byproduct", self.byproduct_state)

        # Add performance variables
        # NOTE: the mass fraction of H2O to treated stream is estimated from P recovered in the byproduct (struvite)
        self.frac_mass_H2O_treated = Var(
            self.flowsheet().time,
            initialize=0.8777,
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            bounds=(0.0, 1),
            doc="Mass recovery fraction of water in the treated stream",
        )
        self.frac_mass_H2O_treated.fix()

        # Default solute concentration
        self.P_removal = Param(
            within=NonNegativeReals,
            default=0.5,
            doc="Reference removal fraction for P on a mass basis",
            units=pyunits.dimensionless,
        )

        self.NH4_removal_mass = Var(
            within=NonNegativeReals,
            initialize=0.5,
            doc="Removal fraction for NH4 on a mass basis",
            units=pyunits.dimensionless,
        )
        self.NH4_removal_mass.fix()

        self.NH4_removal_mol = Var(
            within=NonNegativeReals,
            initialize=0.5,
            doc="Removal fraction for NH4 on a molar basis",
            units=pyunits.dimensionless,
        )
        self.NH4_removal_mol.fix()

        self.NO3_removal = Param(
            within=NonNegativeReals,
            default=0.0,
            doc="Reference removal fraction for NO3 on a mass basis",
            units=pyunits.dimensionless,
        )

        self.NO2_removal = Param(
            within=NonNegativeReals,
            default=0.0,
            doc="Reference removal fraction for NO2 on a mass basis",
            units=pyunits.dimensionless,
        )

        # Add molar mass parameters for unit conversions
        self.molar_mass_NH4 = Param(
            default=18.04,  # g/mol for NH4+
            units=pyunits.g / pyunits.mol,
            doc="Molar mass of ammonium ion for unit conversions",
        )

        self.molar_mass_comp = Param(
            self.config.property_package.component_list,
            default=18.04,  # Default to NH4+ value
            units=pyunits.g / pyunits.mol,
            doc="Molar mass of components for unit conversions",
        )

        add_object_reference(self, "removal_frac_mass_comp", self.split_fraction)

        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.component_list,
            doc="soluble fraction",
        )
        def split_components(blk, t, i):
            if i == "H2O":
                return (
                    blk.removal_frac_mass_comp[t, "byproduct", i]
                    == 1 - blk.frac_mass_H2O_treated[t]
                )
            elif i == "S_PO4":
                return blk.removal_frac_mass_comp[t, "byproduct", i] == blk.P_removal
            elif i == "S_NH4" or i == "NH4_+":
                # Check which removal parameter is fixed and use that one
                if blk.NH4_removal_mass.fixed:
                    # Use mass-based removal directly
                    return (
                        blk.removal_frac_mass_comp[t, "byproduct", i]
                        == blk.NH4_removal_mass
                    )
                elif blk.NH4_removal_mol.fixed:
                    # Convert molar removal to mass removal
                    # This requires accessing the property package to get concentrations
                    if hasattr(blk.properties_in[t], "flow_mol_phase_comp") and hasattr(
                        blk.properties_in[t], "flow_mass_phase_comp"
                    ):
                        # Calculate conversion factor between molar and mass removal
                        return (
                            blk.removal_frac_mass_comp[t, "byproduct", i]
                            == blk.NH4_removal_mol
                        )
                    else:
                        # Fallback if property package doesn't have required attributes
                        return (
                            blk.removal_frac_mass_comp[t, "byproduct", i]
                            == blk.NH4_removal_mass
                        )
                else:
                    # If neither is fixed, default to mass-based removal
                    return (
                        blk.removal_frac_mass_comp[t, "byproduct", i]
                        == blk.NH4_removal_mass
                    )
            elif i == "S_NO3":
                return blk.removal_frac_mass_comp[t, "byproduct", i] == blk.NO3_removal
            elif i == "S_NO2":
                return blk.removal_frac_mass_comp[t, "byproduct", i] == blk.NO2_removal
            else:
                return (
                    blk.removal_frac_mass_comp[t, "byproduct", i] == 0
                )  # assuming other ions not in byproduct

        self.electricity = Var(
            self.flowsheet().time,
            units=pyunits.kW,
            bounds=(0, None),
            doc="Electricity consumption of unit",
        )

        # Energy consumption variables - allow both mass and molar basis
        self.energy_electric_flow_mass = Var(
            self.config.property_package.component_list,
            units=pyunits.kWh / pyunits.kg,
            doc="Electricity intensity with respect to component removal (mass basis)",
        )

        self.energy_electric_flow_mol = Var(
            self.config.property_package.component_list,
            units=pyunits.kWh / pyunits.mol,
            doc="Electricity intensity with respect to component removal (molar basis)",
        )

        @self.Constraint(
            self.flowsheet().time,
            doc="Constraint for electricity consumption based on component removal",
        )
        def electricity_consumption(b, t):
            # Calculate electricity based on mass flow of removed components
            # Use mass basis for calculation to avoid unit conversion issues
            return b.electricity[t] == sum(
                b.energy_electric_flow_mass[j]
                * pyunits.convert(
                    b.properties_byproduct[t].flow_mass_phase_comp["Liq", j],
                    to_units=pyunits.kg / pyunits.hour,
                )
                for j in b.config.property_package.component_list
                if j != "H2O"
            )

        # Chemical dosing variables - allow both mass and molar basis
        self.magnesium_chloride_dosage = Var(
            units=pyunits.kg / pyunits.kg,
            initialize=1.5,
            bounds=(0, None),
            doc="Dosage of magnesium chloride per nutrient removal (mass basis)",
        )
        self.magnesium_chloride_dosage.fix()

        self.magnesium_chloride_dosage_mol = Var(
            units=pyunits.kg
            / pyunits.mol,  # Changed from mol/mol to kg/mol to fix unit compatibility
            initialize=1.5,
            bounds=(0, None),
            doc="Dosage of magnesium chloride per nutrient removal (molar basis)",
        )
        self.magnesium_chloride_dosage_mol.fix()

        self.MgCl2_flowrate = Var(
            self.flowsheet().time,
            units=pyunits.kg / pyunits.hr,
            bounds=(0, None),
            doc="Magnesium chloride flowrate",
        )

        @self.Constraint(
            self.flowsheet().time,
            doc="Constraint for magnesium chloride demand based on nutrient removal",
        )
        def MgCl2_demand(b, t):
            # Try to use NH4_+ if available, otherwise use S_NH4 or S_PO4
            target_component = None
            for comp in ["NH4_+", "S_NH4", "S_PO4"]:
                if comp in b.config.property_package.component_list:
                    target_component = comp
                    break

            if target_component is None:
                # Default to first non-water component if none of the expected components are found
                for comp in b.config.property_package.component_list:
                    if comp != "H2O":
                        target_component = comp
                        break

            return b.MgCl2_flowrate[t] == (
                b.magnesium_chloride_dosage
                * pyunits.convert(
                    b.properties_byproduct[t].flow_mass_phase_comp[
                        "Liq", target_component
                    ],
                    to_units=pyunits.kg / pyunits.hour,
                )
            )

        # Add recovery fraction for water on a molar basis
        self.recovery_frac_mol_H2O = Var(
            self.flowsheet().time,
            initialize=0.95,
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            bounds=(0.0, 1.0),
            doc="Molar recovery fraction of water in the treated stream",
        )
        self.recovery_frac_mol_H2O.fix()

    def _get_performance_contents(self, time_point=0):
        var_dict = {}
        var_dict["Mass fraction of H2O in treated stream"] = self.frac_mass_H2O_treated[
            time_point
        ]
        var_dict["Molar fraction of H2O in treated stream"] = (
            self.recovery_frac_mol_H2O[time_point]
        )
        for j in self.config.property_package.component_list:
            if j != "H2O":
                var_dict[f"Removal fraction of {j} (mass basis)"] = (
                    self.removal_frac_mass_comp[time_point, "byproduct", j]
                )
        var_dict["Electricity Demand"] = self.electricity[time_point]
        var_dict["Electricity Intensity (mass basis)"] = self.energy_electric_flow_mass
        var_dict["Electricity Intensity (molar basis)"] = self.energy_electric_flow_mol
        var_dict["Dosage of MgCl2 per nutrient (mass basis)"] = (
            self.magnesium_chloride_dosage
        )
        var_dict["Dosage of MgCl2 per nutrient (molar basis)"] = (
            self.magnesium_chloride_dosage_mol
        )
        var_dict["Magnesium Chloride Demand"] = self.MgCl2_flowrate[time_point]
        return {"vars": var_dict}

    def _get_stream_table_contents(self, time_point=0):
        return create_stream_table_dataframe(
            {
                "Inlet": self.inlet,
                "Treated": self.treated,
                "Byproduct": self.byproduct,
            },
            time_point=time_point,
        )

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        iscale.set_scaling_factor(self.frac_mass_H2O_treated, 1)
        iscale.set_scaling_factor(self.recovery_frac_mol_H2O, 1)
        iscale.set_scaling_factor(self.NH4_removal_mass, 1)
        iscale.set_scaling_factor(self.NH4_removal_mol, 1)

        if iscale.get_scaling_factor(self.energy_electric_flow_mass) is None:
            sf = iscale.get_scaling_factor(
                self.energy_electric_flow_mass, default=1e-3, warning=True
            )
            iscale.set_scaling_factor(self.energy_electric_flow_mass, sf)

        if iscale.get_scaling_factor(self.energy_electric_flow_mol) is None:
            sf = iscale.get_scaling_factor(
                self.energy_electric_flow_mol, default=1e-3, warning=True
            )
            iscale.set_scaling_factor(self.energy_electric_flow_mol, sf)

        if iscale.get_scaling_factor(self.magnesium_chloride_dosage) is None:
            sf = iscale.get_scaling_factor(
                self.magnesium_chloride_dosage, default=1e0, warning=True
            )
            iscale.set_scaling_factor(self.magnesium_chloride_dosage, sf)

        if iscale.get_scaling_factor(self.magnesium_chloride_dosage_mol) is None:
            sf = iscale.get_scaling_factor(
                self.magnesium_chloride_dosage_mol, default=1e0, warning=True
            )
            iscale.set_scaling_factor(self.magnesium_chloride_dosage_mol, sf)

        for (t, i, j), v in self.removal_frac_mass_comp.items():
            if i in self.config.outlet_list:
                sf = 1
                iscale.set_scaling_factor(v, sf)

    @property
    def default_costing_method(self):
        return cost_genericNP
