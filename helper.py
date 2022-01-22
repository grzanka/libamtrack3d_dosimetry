import itertools

import numpy as np
import pandas as pd
import pyamtrack.libAT as libam

from settings import SimulationSetup


def fluence_cm2(E_MeV_u: float, sim_setup: SimulationSetup) -> float:
    result_fluence_cm2 = libam.AT_fluence_cm2_from_dose_Gy_single(
        p_E_MeV_u=E_MeV_u,
        p_particle_no=sim_setup.beam.particle_code,
        p_D_Gy=sim_setup.beam.dose_gy,
        p_material_no=sim_setup.material.code,
        p_stopping_power_source_no=sim_setup.stopping_power_source_code)
    return result_fluence_cm2


def let_keV_um(E_MeV_u: float, sim_setup: SimulationSetup) -> float:
    stopping_power_keV_um = [0]
    libam.AT_Stopping_Power(p_stopping_power_source=sim_setup.stopping_power_source_name,
                            p_E_MeV_u=[E_MeV_u],
                            p_particle_no=[sim_setup.beam.particle_code],
                            p_material_no=sim_setup.material.code,
                            p_stopping_power_keV_um=stopping_power_keV_um)
    return stopping_power_keV_um[0]


def eloss_keV(E_MeV_u: float, sim_setup: SimulationSetup) -> float:
    result_eloss_keV = libam.AT_mean_energy_loss_keV(p_E_MeV_u=E_MeV_u,
                                                     p_particle_no=sim_setup.beam.particle_code,
                                                     p_material_no=sim_setup.material.code,
                                                     p_slab_thickness_um=sim_setup.material.slab_thickness_um)
    return result_eloss_keV


def eloss_MeV(E_MeV_u: float, sim_setup: SimulationSetup) -> float:
    result_eloss_MeV = 1e-3 * eloss_keV(E_MeV_u, sim_setup)  # keV to MeV
    return result_eloss_MeV


def run_igk(E_MeV_u: float, a0_nm: float, sim_setup: SimulationSetup):


    relative_efficiency, S_HCP, S_gamma, sI_cm2, gamma_dose_Gy, P_I, P_g = [0], [0], [0], [0], [0], [0], [0]

    a0_m = 1e-9 * a0_nm # nm -> m

    libam.AT_run_IGK_method(p_E_MeV_u=[E_MeV_u],
                            p_particle_no=[sim_setup.beam.particle_code],
                            p_fluence_cm2_or_dose_Gy=[-sim_setup.beam.dose_gy],
                            p_material_no=sim_setup.material.code,
                            p_stopping_power_source_no=sim_setup.stopping_power_source_code,
                            p_rdd_model=sim_setup.tst_model.rdd_model_code,
                            p_rdd_parameters=[a0_m, 0, 0],
                            p_er_model=sim_setup.tst_model.er_model_code,
                            p_gamma_model=sim_setup.gamma_response_model.code,
                            p_gamma_parameters=sim_setup.gamma_response_model.parameters_vector,
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


def get_hpc(E_MeV_u: float, a0_nm: float, sim_setup: SimulationSetup) -> float:
    current_E_MeV_u = E_MeV_u
    hcp: float = 0.
    for _ in range(10):
        current_E_MeV_u -= eloss_MeV(current_E_MeV_u, sim_setup)
        if current_E_MeV_u <= 0:
            break
        _, S_HCP, _, _, _, _, _ = run_igk(current_E_MeV_u, a0_nm, sim_setup)
        hcp += S_HCP[0] / 10.
    return hcp


def create_df(sim_setup: SimulationSetup) -> pd.DataFrame:

    # iterating through dictionary is equivalent to R expand.grid
    data_dict = {
        'E_MeV_u': np.linspace(start=sim_setup.start_E_MeV_u, stop=sim_setup.stop_E_MeV_u, num=sim_setup.num_E_MeV_u),
        'a0_nm': sim_setup.tst_model.a0_nm
    }
    df: pd.DataFrame = pd.DataFrame.from_records(data=itertools.product(*data_dict.values()), columns=data_dict.keys())

    return df
