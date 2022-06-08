import itertools

import numpy as np
import numpy.typing as npt
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

    a0_m = 1e-9 * a0_nm  # nm -> m

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


def rdd_dose_Gy(r_m: npt.NDArray, sim_setup: SimulationSetup) -> npt.NDArray:
    result_dose_Gy = np.full_like(r_m, np.nan)
    if sim_setup.beam.start_E_MeV_u != sim_setup.beam.stop_E_MeV_u:
        print("Only single energy value should be provided in sim_setup.beam (start_E_MeV_u == stop_E_MeV_u)")
        return result_dose_Gy
    if sim_setup.beam.num_E_MeV_u != 1:
        print("Only single energy value should be provided in sim_setup.beam (num_E_MeV_u == 1)")
        return result_dose_Gy
    if len(sim_setup.tst_model.a0_nm) != 1:
        print("Only single a0 value should be provided in sim_setup.tst_model")
        return result_dose_Gy
    a0_m = sim_setup.tst_model.a0_nm[0] * 1e-9

    # set default parameters based on https://github.com/libamtrack/library/blob/master/include/AT_RDD.h#L102
    rdd_parameters = [0., 0., 0.]
    if sim_setup.tst_model.rdd_model_code == libam.RDDModels.RDD_KatzPoint.value:
        rdd_parameters = [1e-10, 1e-10, 0.]
        if not sim_setup.tst_model.er_model_code in {
                libam.AT_ERModels.ER_ButtsKatz.value, libam.AT_ERModels.ER_Edmund.value,
                libam.AT_ERModels.ER_Waligorski.value
        }:
            print("KatzPoint RDD model is compatible only with ButtsKatz, Edmund and Waligorski ER models")
            return result_dose_Gy
    elif sim_setup.tst_model.rdd_model_code == libam.RDDModels.RDD_Geiss.value:
        rdd_parameters = [a0_m, 0., 0.]
    elif sim_setup.tst_model.rdd_model_code == libam.RDDModels.RDD_KatzSite.value:
        if not sim_setup.tst_model.er_model_code in {
                libam.AT_ERModels.ER_ButtsKatz.value, libam.AT_ERModels.ER_Edmund.value,
                libam.AT_ERModels.ER_Waligorski.value
        }:
            print("KatzSite RDD model is compatible only with ButtsKatz, Edmund and Waligorski ER models")
            return result_dose_Gy
        rdd_parameters = [a0_m, 1e-10, 0.]
    elif sim_setup.tst_model.rdd_model_code == libam.RDDModels.RDD_CucinottaPoint.value:
        rdd_parameters = [5e-11, 1e-10, 0.]
    elif sim_setup.tst_model.rdd_model_code == libam.RDDModels.RDD_KatzExtTarget.value:
        if not sim_setup.tst_model.er_model_code in {
                libam.AT_ERModels.ER_ButtsKatz.value, libam.AT_ERModels.ER_Edmund.value,
                libam.AT_ERModels.ER_Waligorski.value
        }:
            print("KatzExtTarget RDD model is compatible only with ButtsKatz, Edmund and Waligorski ER models")
            return result_dose_Gy
        rdd_parameters = [1e-10, a0_m, 1e-10]
    elif sim_setup.tst_model.rdd_model_code == libam.RDDModels.RDD_CucinottaExtTarget.value:
        rdd_parameters = [5e-11, a0_m, 1e-10]

    result_dose_Gy_tmp = [0.] * result_dose_Gy.size
    libam.AT_D_RDD_Gy(p_r_m=r_m.tolist(),
                      p_E_MeV_u=sim_setup.beam.start_E_MeV_u,
                      p_particle_no=sim_setup.beam.particle_code,
                      p_material_no=sim_setup.material.code,
                      p_rdd_model=sim_setup.tst_model.rdd_model_code,
                      p_rdd_parameter=rdd_parameters,
                      p_er_model=sim_setup.tst_model.er_model_code,
                      p_stopping_power_source_no=sim_setup.stopping_power_source_code,
                      p_D_RDD_Gy=result_dose_Gy_tmp)

    result_dose_Gy = np.array(result_dose_Gy_tmp)
    del result_dose_Gy_tmp

    return result_dose_Gy


def rdd_dose_Gy_basic(r_m: npt.NDArray,
                      a0_nm: float = 5.,
                      E_MeV_u: float = 150.,
                      particle_name: str = "1H",
                      material_name: str = "Water, Liquid",
                      rdd_model_name: str = "RDD_Geiss") -> npt.NDArray:

    simulation_setup = SimulationSetup()

    # all necessary parameters
    simulation_setup.beam.particle_name = particle_name
    simulation_setup.material.material_name = material_name
    simulation_setup.tst_model.rdd_model_name = rdd_model_name

    # we will calculate RDD just for a single energy value
    simulation_setup.beam.start_E_MeV_u = E_MeV_u
    simulation_setup.beam.stop_E_MeV_u = simulation_setup.beam.start_E_MeV_u
    simulation_setup.beam.num_E_MeV_u = 1

    # some of the options in simulation setup are not needed for RDD calculation,
    # instead of leaving default values, we set those items to `nan` (not-a-number) or to `None`

    # just a single value of a0 parameter, note that final `,` is needed in python to define a single-element tuple
    simulation_setup.tst_model.a0_nm = (a0_nm, )

    # model of response to reference radiation (gamma) is not needed for RDD calculations
    simulation_setup.gamma_response_model = None

    # there is no need to specify slab thickness for RDD calculation
    simulation_setup.beam.dose_gy = float('nan')

    # there is no need to specify slab thickness and saturation cross-section for RDD calculation
    simulation_setup.material.slab_thickness_um = float('nan')
    simulation_setup.saturation_cross_section_factor = float('nan')

    result_dose_Gy = rdd_dose_Gy(r_m=r_m, sim_setup=simulation_setup)

    return result_dose_Gy


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
        'E_MeV_u':
        np.linspace(start=sim_setup.beam.start_E_MeV_u,
                    stop=sim_setup.beam.stop_E_MeV_u,
                    num=sim_setup.beam.num_E_MeV_u),
        'a0_nm':
        sim_setup.tst_model.a0_nm
    }
    df: pd.DataFrame = pd.DataFrame.from_records(data=itertools.product(*data_dict.values()), columns=data_dict.keys())

    return df
