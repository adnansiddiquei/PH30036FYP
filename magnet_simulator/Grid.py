from typing import Union

import numpy as np
from scipy.constants import mu_0

from .MagnetArray import MagnetArray
from .Graphite import Graphite
from .types import TVector, TArrayOfVectors, TArrayOfScalars, TGraphiteProperties
from .utils import around, dot_product


class Grid:
    def __init__(self, magnet_array: MagnetArray, graphite: Graphite, magnet_array_origin: TVector = (0, 0, -5e-3)):
        self.magnet_array = magnet_array
        self.magnet_array_origin = np.array(magnet_array_origin)

        self.graphite = graphite
        self.graphite_centre = None

        # the following properties regarding the Graphite are calculated every time self.position_graphite is called
        self.graphite_properties: TGraphiteProperties = {
            'shape': self.graphite.section_positions.shape
        }

    def alter_graphite(self, new_position: TVector = None, new_rotation: float = None, compute_forces: bool = True,
                       compute_energies: bool = True, compute_torque: bool = True,
                       compute_force_components=(True, True, True)):
        """
        Moves the position of the Graphite and computes all related properties at the same time.

        :param new_position: The new position of the centre of the Graphite, if the graphite has moved.
        :param new_rotation: The new rotation of the Graphite, if the graphite has rotated.
        :param compute_forces: Whether to compute the forces on the Graphite.
        :param compute_energies: Whether to compute the energies on the Graphite.
        :param compute_torque: Whether to compute then torque on the Graphite.
        :param compute_force_components: Which axis to compute the forces for.
        """

        def compute_graphite_abs_section_positions(graphite_centre: TVector) -> TArrayOfVectors:
            """
            Compute self.graphite_properties['abs_section_positions']. This is the absolute position each of the
            sections of the Graphite w.r.t. to the co-ordinate [0,0,0] on the Grid.

            :param graphite_centre: The co-ordinates for the centre of the graphite.

            :return: An array with the same shape as self.graphite.section_positions.
            """
            graphite_centre: TArrayOfVectors = np.full(self.graphite_properties['shape'], graphite_centre)
            return graphite_centre + self.graphite.section_positions

        def compute_graphite_rel_section_positions(abs_section_positions: TArrayOfVectors) -> TArrayOfVectors:
            """
            Compute self.graphite_properties['rel_section_positions']. This is the relative position each of the
            sections of the Graphite w.r.t. to the centre of the MagnetArray (self.magnet_array_origin, this will
            usually be at co-ordinate [0, 0, 0] anyway).

            :param abs_section_positions: self.graphite_properties['abs_section_positions']

            :return: An array with the same shape as self.graphite.section_positions.
            """
            magnet_array_centre: TArrayOfVectors = np.full(self.graphite_properties['shape'], self.magnet_array_origin)
            return abs_section_positions - magnet_array_centre

        # TODO vectorise this so we can simply pass in the entire array rel_section_positions
        def compute_graphite_B_ext(rel_section_positions: TArrayOfVectors) -> TArrayOfVectors:
            """
            Computes the external B-field at each of the section positions due to the MagnetArray.

            :param rel_section_positions: self.graphite_properties['rel_section_positions']

            :return: An array with the same shape as self.graphite.section_positions.
            """
            return self.magnet_array.B(rel_section_positions)

        # TODO comment
        def compute_graphite_force_density(rel_section_positions: TArrayOfVectors, calc_components) -> TArrayOfVectors:
            return self.magnet_array.force_density(rel_section_positions, calc_components=calc_components)

        def compute_section_magnetic_forces(section_magnetic_force_density: TArrayOfVectors) -> TArrayOfVectors:
            """
            Computes the forces on each of the Graphite section positions due to the MagnetArray.

            :param section_magnetic_force_density: The force density at every section position of the graphite

            :return: An array with the same shape as self.graphite.section_positions.
            """
            return section_magnetic_force_density * self.graphite.section_volume

        def compute_section_magnetic_potential_energy_density(rel_section_positions: TArrayOfVectors,
                                                              B_ext: TArrayOfVectors) -> TArrayOfVectors:
            return self.magnet_array.energy_density(rel_section_positions, b_field=B_ext)

        def compute_magnetic_potential_energy(
                section_magnetic_potential_energy_density: TArrayOfScalars) -> TArrayOfScalars:
            """
            Compute the magnetic potential energy of each of the sections of the Graphite.

            :param section_magnetic_potential_energy_density:

            :return: An array of scalars with the potential energy of each section
            """
            return section_magnetic_potential_energy_density * self.graphite.section_volume

        def compute_section_torque_density(B_ext: TArrayOfVectors) -> TArrayOfVectors:
            # TODO might be wrong?
            magnetisation = (1 / mu_0) * self.graphite.magnetic_susceptibility * B_ext
            return -1 * np.cross(magnetisation, B_ext)

        if new_position is not None:
            self.graphite_centre = np.array(new_position)  # set the graphite_centre to the new_position

        if new_rotation is not None:
            self.graphite.reconfigure(rotation=new_rotation)  # set the rotation of the Graphite

        self.graphite_properties['abs_section_positions'] = compute_graphite_abs_section_positions(self.graphite_centre)

        self.graphite_properties['rel_section_positions'] = \
            compute_graphite_rel_section_positions(self.graphite_properties['abs_section_positions'])

        self.graphite_properties['B_ext'] = compute_graphite_B_ext(self.graphite_properties['rel_section_positions'])

        if compute_forces:
            # if the user has asked to compute forces on the Graphite
            self.graphite_properties['section_magnetic_force_density'] = compute_graphite_force_density(
                self.graphite_properties['rel_section_positions'], compute_force_components
            )

            self.graphite_properties['section_magnetic_force'] = compute_section_magnetic_forces(
                self.graphite_properties['section_magnetic_force_density']
            )

            self.graphite_properties['section_force'] = self.graphite_properties['section_magnetic_force'] + \
                np.full(self.graphite_properties['shape'], self.graphite.section_gravity_force)

            self.graphite_properties['total_magnetic_force'] = \
                np.sum(self.graphite_properties['section_magnetic_force'], axis=0)

            self.graphite_properties['total_force'] = \
                self.graphite_properties['total_magnetic_force'] + self.graphite.gravity_force

            self.graphite_properties['section_force'] = self.graphite_properties['section_force']
            self.graphite_properties['total_magnetic_force'] = self.graphite_properties['total_magnetic_force']
            self.graphite_properties['total_force'] = around(self.graphite_properties['total_force'])

        if compute_energies:
            # if the user has asked to compute the energies on the Graphite
            self.graphite_properties['section_magnetic_potential_energy_density'] = \
                compute_section_magnetic_potential_energy_density(
                    self.graphite_properties['rel_section_positions'],
                    self.graphite_properties['B_ext']
                )

            self.graphite_properties['section_magnetic_potential_energy'] = compute_magnetic_potential_energy(
                self.graphite_properties['section_magnetic_potential_energy_density']
            )

            self.graphite_properties['total_magnetic_potential_energy'] = \
                np.sum(self.graphite_properties['section_magnetic_potential_energy'])

        if compute_torque:
            # self.graphite_properties['section_torque_density'] = compute_section_torque_density(
            #     self.graphite_properties['B_ext']
            # )

            self.graphite_properties['section_torque'] = np.cross(self.graphite.section_positions,
                                                                  self.graphite_properties['section_magnetic_force'])

            # self.graphite_properties['section_torque'] = \
            #     self.graphite_properties['section_torque_density'] * self.graphite.section_volume

            self.graphite_properties['total_torque'] = np.sum(self.graphite_properties['section_torque'], axis=0)

    def B(self, r: TVector, method=1) -> TVector:
        """ See documentation for MagnetArray.B """
        rel_r = r - self.magnet_array_origin
        return self.magnet_array.B(r=rel_r, method=method)

    def F(self, r: TVector, m2: TVector) -> TVector:
        """ See documentation for MagnetArray.F """
        rel_r = r - self.magnet_array_origin
        return self.magnet_array.F(r=rel_r, m2=m2)

    def force_density(self, r: Union[TVector, TArrayOfVectors]) -> TArrayOfVectors:
        """ See documentation for MagnetArray.force_density """
        rel_r = r - self.magnet_array_origin
        return self.magnet_array.force_density(r=rel_r)

    def energy_density(self, r: Union[TVector, TArrayOfVectors]) -> TArrayOfVectors:
        """ See documentation for MagnetArray.energy_density """
        rel_r = r - self.magnet_array_origin
        return self.magnet_array.energy_density(r=rel_r)
