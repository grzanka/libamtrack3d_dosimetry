from dataclasses import dataclass, field
from functools import lru_cache
import itertools

import numpy as np
import pandas as pd
import pyamtrack.libAT as libam


@dataclass
class SimulationSetup:
    '''
    TODO add list of available materials
    '''
    particle_name: str = "1H"
    dose_gy: float = 0.3
    material_name: str = "Aluminum Oxide"
    er_model_name: str = "ER_Edmund"
    rdd_model_name: str = "RDD_Geiss"
    gamma_response_model_name: str = "GR_GeneralTarget"
    stopping_power_source_name: str = "PSTAR"
    slab_thickness_um: float = 100.
    saturation_cross_section_factor: float = 1.4
    r: float = 44.
    smax: float = 27.6
    d01: float = 2.9
    c1: float = 1.
    m1: float = 1.
    d02: float = 4.66
    c2: float = 2.
    m2: float = 1.

    def __hash__(self):
        '''
        Calculate hash function from values of all attributes which are not properties
        This is needed for @lru_cache function to speedup calculations
        '''
        return hash((getattr(self, attr_name) for attr_name in vars(self).keys()))

    @property
    @lru_cache
    def particle_code(self) -> int:
        return libam.AT_particle_no_from_particle_name_single(self.particle_name)

    @property
    @lru_cache
    def material_code(self) -> int:
        return libam.AT_material_number_from_name(self.material_name)

    @property
    @lru_cache
    def rdd_model_code(self) -> int:
        return libam.RDDModels[self.rdd_model_name].value

    @property
    @lru_cache
    def er_model_code(self) -> int:
        return libam.AT_ERModels[self.er_model_name].value

    @property
    @lru_cache
    def gamma_response_model_code(self) -> int:
        return libam.AT_GammaResponseModels[self.gamma_response_model_name].value

    @property
    @lru_cache
    def stopping_power_source_code(self) -> int:
        return libam.stoppingPowerSource_no[self.stopping_power_source_name].value


def fluence_cm2(E_MeV_u: float, sim_setup: SimulationSetup) -> float:
    result_fluence_cm2 = libam.AT_fluence_cm2_from_dose_Gy_single(
        p_E_MeV_u=E_MeV_u,
        p_particle_no=sim_setup.particle_code,
        p_D_Gy=sim_setup.dose_gy,
        p_material_no=sim_setup.material_code,
        p_stopping_power_source_no=sim_setup.stopping_power_source_code)
    return result_fluence_cm2


def eloss_keV(E_MeV_u: float, sim_setup: SimulationSetup) -> float:
    result_eloss_keV = libam.AT_mean_energy_loss_keV(p_E_MeV_u=E_MeV_u,
                                                     p_particle_no=sim_setup.particle_code,
                                                     p_material_no=sim_setup.material_code,
                                                     p_slab_thickness_um=sim_setup.slab_thickness_um)
    return result_eloss_keV


def eloss_MeV(E_MeV_u: float, sim_setup: SimulationSetup) -> float:
    result_eloss_MeV = 1e-3 * eloss_keV(E_MeV_u, sim_setup)  # keV to MeV
    return result_eloss_MeV


def run_igk(E_MeV_u: float, a0_m: float, sim_setup: SimulationSetup):

    k1 = sim_setup.smax * (sim_setup.r / 100)
    k2 = sim_setup.smax * np.abs(1. - sim_setup.r / 100)
    gamma_parameter_peak_B = [
        k1, sim_setup.d01, sim_setup.c1, sim_setup.m1, k2, sim_setup.d02, sim_setup.c2, sim_setup.m2, 0
    ]

    relative_efficiency, S_HCP, S_gamma, sI_cm2, gamma_dose_Gy, P_I, P_g = [0], [0], [0], [0], [0], [0], [0]

    libam.AT_run_IGK_method(p_E_MeV_u=[E_MeV_u],
                            p_particle_no=[sim_setup.particle_code],
                            p_fluence_cm2_or_dose_Gy=[-sim_setup.dose_gy],
                            p_material_no=sim_setup.material_code,
                            p_stopping_power_source_no=sim_setup.stopping_power_source_code,
                            p_rdd_model=sim_setup.rdd_model_code,
                            p_rdd_parameters=[a0_m, 0, 0],
                            p_er_model=sim_setup.er_model_code,
                            p_gamma_model=sim_setup.gamma_response_model_code,
                            p_gamma_parameters=gamma_parameter_peak_B,
                            p_saturation_cross_section_factor=sim_setup.saturation_cross_section_factor,
                            p_write_output=False,
                            p_relative_efficiency=relative_efficiency,
                            p_S_HCP=S_HCP,
                            p_S_gamma=S_gamma,
                            p_sI_cm2=sI_cm2,
                            p_gamma_dose_Gy=gamma_dose_Gy,
                            p_P_I=P_I,
                            p_P_g=P_g)
    return relative_efficiency, S_HCP, S_gamma, sI_cm2, gamma_dose_Gy, P_I, P_g


def get_hpc(E_MeV_u: float, a0_m: float, sim_setup: SimulationSetup) -> float:
    current_E_MeV_u = E_MeV_u
    hcp : float = 0.
    for _ in range(10):
        current_E_MeV_u -= eloss_MeV(current_E_MeV_u, sim_setup)
        if current_E_MeV_u <= 0:
            break
        _, S_HCP, _, _, _, _, _ = run_igk(current_E_MeV_u, a0_m, sim_setup)    
        hcp += S_HCP[0] / 10.
    return hcp

def let_keV_um(E_MeV_u: float, sim_setup: SimulationSetup) -> float:
    pass

def create_df(sim_setup: SimulationSetup) -> pd.DataFrame:

    # iterating through dictionary is equivalent to R expand.grid
    data_dict = {
        'E_MeV_u': np.linspace(start=1.5, stop=60., num=400),
        'a0_m': 1e-9 * np.array([95., 50., 150.]),  # nm to m,
    }
    df: pd.DataFrame = pd.DataFrame.from_records(data=itertools.product(*data_dict.values()), columns=data_dict.keys())

    return df
