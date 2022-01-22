from helper import create_df, eloss_MeV, fluence_cm2, get_hpc, let_keV_um, run_igk
from printing import pretty_print
from settings import SimulationSetup


def demo():
    """Function to demontrate usage of all functions"""

    # adjust simulation setup
    simulation_setup = SimulationSetup()
    simulation_setup.start_E_MeV_u = 30.
    simulation_setup.stop_E_MeV_u = 60.
    simulation_setup.num_E_MeV_u = 4
    pretty_print(simulation_setup)

    test_E_MeV_u = 60.
    test_a0_nm = 100

    # test fluence calculation
    test_fluence_cm2 = fluence_cm2(E_MeV_u=test_E_MeV_u, sim_setup=simulation_setup)
    print(f"E = {test_E_MeV_u} MeV/u, fluence = {test_fluence_cm2:g} /cm2")

    # test LET calculation
    test_let_keV_um = let_keV_um(E_MeV_u=test_E_MeV_u, sim_setup=simulation_setup)
    print(f"E = {test_E_MeV_u} MeV/u, LET = {test_let_keV_um:.3f} keV/um")

    # test energy loss calculation
    test_eloss_MeV = eloss_MeV(E_MeV_u=test_E_MeV_u, sim_setup=simulation_setup)
    print(f"E = {test_E_MeV_u} MeV/u, Eloss = {test_eloss_MeV:.3f} MeV")

    # test IGK algorithm
    test_IGK_output = run_igk(E_MeV_u=test_E_MeV_u, a0_nm=test_a0_nm, sim_setup=simulation_setup)
    print(f"E = {test_E_MeV_u} MeV/u, a0 = {test_a0_nm} nm, IGK output = {test_IGK_output}")

    # test HPC calculation
    test_HPC = get_hpc(E_MeV_u=test_E_MeV_u, a0_nm=test_a0_nm, sim_setup=simulation_setup)
    print(f"E = {test_E_MeV_u} MeV/u, a0 = {test_a0_nm} nm, HPC = {test_HPC}")

    # test DataFrame creation
    test_df = create_df(simulation_setup)
    print(test_df.head())


if __name__ == "__main__":
    demo()