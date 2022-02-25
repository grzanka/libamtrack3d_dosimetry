from dataclasses import dataclass, field
from functools import lru_cache
from typing import Tuple
import numpy as np

import pyamtrack.libAT as libam


def pure_attr_hash(obj):
    '''
        Calculate hash function from values of all attributes which are not properties
        This is needed for @lru_cache function to speedup calculations
        '''
    return hash((getattr(obj, attr_name) for attr_name in vars(obj).keys()))


@dataclass
class BeamSetup:
    particle_name: str = "1H"
    start_E_MeV_u: float = 1.5
    stop_E_MeV_u: float = 60.
    num_E_MeV_u: int = 400
    dose_gy: float = 0.3

    @property
    @lru_cache
    def particle_code(self) -> int:
        return libam.AT_particle_no_from_particle_name_single(self.particle_name)

    def __hash__(self):
        return pure_attr_hash(self)


@dataclass
class DetectorPropertySetup:
    material_name: str = "Aluminum Oxide"
    slab_thickness_um: float = 100.

    @property
    @lru_cache
    def code(self) -> int:
        return libam.AT_material_number_from_name(self.material_name)

    def __hash__(self):
        return pure_attr_hash(self)


@dataclass
class TrackStructureModel:
    er_model_name: str = "ER_Edmund"
    rdd_model_name: str = "RDD_Geiss"
    a0_nm: Tuple[float, ...] = (95., 50., 150.)

    @property
    @lru_cache
    def er_model_code(self) -> int:
        return libam.AT_ERModels[self.er_model_name].value

    @property
    @lru_cache
    def rdd_model_code(self) -> int:
        return libam.RDDModels[self.rdd_model_name].value

    def __hash__(self):
        return pure_attr_hash(self)


@dataclass
class GammaResponseModel:
    name: str = "GR_GeneralTarget"
    r: float = 44.
    smax: float = 27.6
    d01: float = 2.9
    c1: float = 1.
    m1: float = 1.
    d02: float = 4.66
    c2: float = 2.
    m2: float = 1.

    @property
    @lru_cache
    def code(self) -> int:
        return libam.AT_GammaResponseModels[self.name].value

    @property
    @lru_cache
    def parameters_vector(self) -> tuple:
        k1 = self.smax * (self.r / 100)
        k2 = self.smax * np.abs(1. - self.r / 100)
        result = (k1, self.d01, self.c1, self.m1, k2, self.d02, self.c2, self.m2, 0)
        return result

    def __hash__(self):
        return pure_attr_hash(self)


@dataclass
class SimulationSetup:
    '''
    TODO add list of available materials
    '''
    beam: BeamSetup = BeamSetup()
    material: DetectorPropertySetup = DetectorPropertySetup()
    gamma_response_model: GammaResponseModel = GammaResponseModel()
    tst_model: TrackStructureModel = TrackStructureModel()
    stopping_power_source_name: str = "PSTAR"
    saturation_cross_section_factor: float = 1.4

    @property
    @lru_cache
    def stopping_power_source_code(self) -> int:
        return libam.stoppingPowerSource_no[self.stopping_power_source_name].value

    def __hash__(self):
        return pure_attr_hash(self)
