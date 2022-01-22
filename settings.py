from dataclasses import dataclass, field
from functools import lru_cache

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
    start_E_MeV_u: float = 1.5
    stop_E_MeV_u: float = 60.
    num_E_MeV_u: int = 400

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