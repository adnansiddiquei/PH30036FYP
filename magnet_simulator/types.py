import numpy as np
from typing import TypedDict, Union, List

TVector = np.ndarray  # a vector type [float, float, float] = [x, y, z]
TArrayOfVectors = np.ndarray  # an array of TVectors, [[x, y, z], [x, y, z], [...], ...]
TArrayOfScalars = np.ndarray  # an array of scalars, i.e., a 1D array


# this TypedDict represents the keys that are within the graphite_properties dict
class TGraphiteProperties(TypedDict, total=False):
    # absolute position of each section of the graphite w.r.t. centre of the Grid ([0, 0, 0])
    abs_section_positions: TArrayOfVectors

    # relative position of each section of the graphite w.r.t centre of MagnetArray (which is usually at [0,0,-5e-3])
    # on the grid's frame of reference
    rel_section_positions: TArrayOfVectors

    # the B-field at each section of the graphite due to the MagnetArray on the Grid
    B_ext: TArrayOfVectors

    section_magnetic_potential_energy_density: TArrayOfScalars

    # magnetic potential energy of each of the sections in the Graphite, computed by U = (-m.B_ext) where m is the
    # dipole moment of the section and B_ext is the external B-field at that section
    section_magnetic_potential_energy: TArrayOfScalars

    # total magnetic potential energy of the piece of Graphite, this is the sum of section_magnetic_potential_energy
    total_magnetic_potential_energy: Union[float, np.ndarray]

    # the magnetic force density (N/m^3) at each of the section positions of the graphite
    section_magnetic_force_density: TArrayOfVectors

    # the force on each of the sections of the Graphite due to MagnetArray
    section_magnetic_force: TArrayOfVectors

    # the total force on the Graphite due to the MagnetArray
    total_magnetic_force: TVector

    # this is the total force (magnetic and gravity) on each of the sections of the graphite
    section_force: TArrayOfVectors

    # total force on the Graphite due to the MagnetArray and gravity
    total_force: TVector

    # torque density at each section position
    section_torque_density: TArrayOfVectors

    # torque on each section of the graphite
    section_torque: TArrayOfVectors

    # total torque on graphite
    total_torque: Union[float, np.ndarray]

    # the shape of the graphite.section_positions array, this is used a lot to reshape other pieces of data into the
    # same shape as this so that calculations can be performed in a vectorised manner
    shape: tuple
