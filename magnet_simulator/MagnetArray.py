import numpy as np
from typing import Union
from .types import TVector, TArrayOfVectors
from .Magnet import Magnet
from .Holed10mmMagnet import Holed10mmMagnet
from .utils import around, dot_product
from .Graphite import graphite_properties

from scipy.constants import mu_0

TMagnet = Union[Magnet, Holed10mmMagnet]


class MagnetArray:
    """
    A wrapper class which represents an array of magnet with a certain configuration.
    """

    def __init__(self, magnet, magnet_up: TMagnet, magnet_down: TMagnet, config: str = None):
        """
        :param config: Can be None, '1x1', '2x2', '3x3' or '4x4'. If provided, then MagnetArray will automatically
        generate these arrays. NOTE - currently only the 2x2 config has been implemented.
        """
        self._magnets = np.array([])  # this will contain all the magnets in the magnet array

        self.magnet = magnet
        self.magnet_up = magnet_up
        self.magnet_down = magnet_down
        self.magnet_array_centre: TVector = np.array([])

        self.config = config

        if self.config:
            if config.lower() == '1x1':
                self._generate_1x1_config()
            if config.lower() == '1x2':
                self._generate_1x2_config()
            if config.lower() == '2x2':
                self._generate_2x2_config()
            if config.lower() == '3x1':
                self._generate_3x1_config()
            if config.lower() == '3x3':
                self._generate_3x3_config()
            if config.lower() == '3x3invert':
                self._generate_3x3invert_config()
            if config.lower() == '2x2up':
                self._generate_2x2up_config()
            if config.lower() == '2x2invert':
                self._generate_2x2invert_config()

    def remove_all_magnets(self):
        self._magnets = np.array([])
        self.magnet_array_centre = np.array([])

    def add_magnet(self, position: TVector, magnet: TMagnet):
        # add the magnet
        self._magnets = np.append(self._magnets, {
            'magnet': self.magnet(
                magnetisation=magnet.magnetisation, cubic_section_sizes=magnet.cubic_section_sizes, label=magnet.label
            ),
            'position': position
        })

        # re-compute the centre of the magnet array
        self.magnet_array_centre = around(
            np.array([
                self._magnets[i]['position'] for i in range(len(self._magnets))
            ]).mean(axis=0)
        )

    def B(self, r: Union[TVector, TArrayOfVectors], method=1) -> TVector:
        """
        Calculates magnetic field at position r produced by the magnet. Equation from here
        https://en.wikipedia.org/wiki/Magnetic_moment#Magnetic_field_of_a_magnetic_moment.

        :param r: A numpy array or list of shape (3, ). Vector to point we want the B-field for, this vector must be
                in the frame of reference of the magnet_array. I.e., the [0, 0, 0] for this vector must also be the
                [0, 0, 0] for the magnet_array.
        :param method: Either 1 or 2. Which method to use to calculate the B-field.
        """
        total_B = np.array([0, 0, 0])

        # loop over each magnet and calculate the contribution to the total_B due to each magnet
        for i, magnet_details in zip(range(len(self._magnets)), self._magnets):
            # relative_r is the relative vector from the centre of each individual magnet to vector r
            relative_r = np.array(r) - magnet_details['position']
            total_B = total_B + magnet_details['magnet'].B(r=relative_r, method=method)

        return around(total_B)

    def F(self, r: TVector, m2: TVector) -> TVector:
        """
        Calculates the force exerted on m2 (magnetic moment 2) as a result of every magnet in this MagnetArray.
        For more details on equation, see documentation on Magnet.F

        :param r: Relative vector [x, y, z] from centre of this MagnetArray to m2.
        :param m2: Magnetic moment of magnet at r.

        :return: Force experienced by m2.
        """
        total_F = np.array([0, 0, 0])

        for magnet_details in self._magnets:
            relative_r = np.array(r) - magnet_details['position']  # relative to centre of the current magnet
            total_F = total_F + magnet_details['magnet'].F(r=relative_r, m2=m2)

        return around(total_F)

    def dBd(self, axis: str, r: Union[TVector, TArrayOfVectors], power=1, delta=1e-7, multiplier=(1, 1, 1)):
        """
        Computes the value of dB/dz at the location(s) provided by the argument r. This function uses the central
        difference approximation (CDA), with step size dz, to estimate the value of the derivative. If you would like
        to calculate (dB^2)/dz instead, then pass power=2.

        :param axis: Either 'x', 'y' or 'z'. Whether to calc db/dx, db/dy or db/dz
        :param r: A single vector or an array of vectors to compute dB/dz at.
        :param delta: The step size to use for CDA.
        :param power: If you would like to calculate (dB^2)/dz instead of dB/dz, set this to 2.
        :param multiplier:

        :return: An array of shape r.shape, populated with the value of dB/dz at those points.
        """
        r = np.array(r)
        multiplier = np.array(multiplier)

        if axis.lower() == 'x':
            delta_broadcast = np.full(r.shape, [delta, 0, 0])  # change delta into an array of the same shape as r
        elif axis.lower() == 'y':
            delta_broadcast = np.full(r.shape, [0, delta, 0])  # change delta into an array of the same shape as r
        elif axis.lower() == 'z':
            delta_broadcast = np.full(r.shape, [0, 0, delta])  # change delta into an array of the same shape as r
        else:
            raise ValueError("'axis' must be one of 'x', 'y', 'z'")

        # these are r+delta_broadcast and r-delta_broadcast to be used for the central difference approximation
        r_plus_delta = r + delta_broadcast
        r_minus_delta = r - delta_broadcast

        # this is the value of dB/delta_broadcast at every co-ordinate passed in the r vector
        result = (1 / (2 * delta)) * (self.B(r_plus_delta) ** power - self.B(r_minus_delta) ** power)

        return result

    # TODO: comment
    def force_density(self, r: Union[TVector, TArrayOfVectors],
                      magnetic_susceptibility: TVector = graphite_properties['magnetic_susceptibility'],
                      delta=1e-8, calc_components=(True, True, True)):
        """
        Calculates the force density (N/m^3) for a piece of graphite at every position vector provided in r. The force
        density is calculated at each point assuming that the graphite has the provided magnetic susceptibility.

        :param r: A single, or array of position vectors to calculate the force density at.
        :param magnetic_susceptibility:
        :param delta:
        :param calc_components:
        :return:
        """
        r = np.array(r)
        magnetic_susceptibility = np.array(magnetic_susceptibility)

        if len(r.shape) == 1:
            r = np.array([r])

        delta_broadcast_x = np.full(r.shape, [delta, 0, 0])  # change delta into an array of the same shape as r
        delta_broadcast_y = np.full(r.shape, [0, delta, 0])  # change delta into an array of the same shape as r
        delta_broadcast_z = np.full(r.shape, [0, 0, delta])  # change delta into an array of the same shape as r

        if calc_components[0]:
            fx = (1 / (2 * mu_0)) * (1 / (2 * delta)) * (
                    np.sum(magnetic_susceptibility * (self.B(r + delta_broadcast_x) ** 2), axis=1) -
                    np.sum(magnetic_susceptibility * (self.B(r - delta_broadcast_x) ** 2), axis=1)
            )
        else:
            fx = np.zeros(len(r)) * np.nan

        if calc_components[1]:
            fy = (1 / (2 * mu_0)) * (1 / (2 * delta)) * (
                    np.sum(magnetic_susceptibility * (self.B(r + delta_broadcast_y) ** 2), axis=1) -
                    np.sum(magnetic_susceptibility * (self.B(r - delta_broadcast_y) ** 2), axis=1)
            )
        else:
            fy = np.zeros(len(r)) * np.nan

        if calc_components[2]:
            fz = (1 / (2 * mu_0)) * (1 / (2 * delta)) * (
                    np.sum(magnetic_susceptibility * (self.B(r + delta_broadcast_z) ** 2), axis=1) -
                    np.sum(magnetic_susceptibility * (self.B(r - delta_broadcast_z) ** 2), axis=1)
            )
        else:
            fz = np.zeros(len(r)) * np.nan

        f = np.array([fx, fy, fz]).T

        return f

    def energy_density(self, r: Union[TVector, TArrayOfVectors],
                       magnetic_susceptibility: TVector = graphite_properties['magnetic_susceptibility'],
                       b_field=None):
        """
        Calculates the energy density of a piece of graphite placed at each position in r.

        :param r: A single position, or an array of positions.
        :param magnetic_susceptibility: Magnetic susceptibility of graphite.
        :param b_field: b_field at the provided area, if it has already been calculated.

        :return: energy density at each location.
        """
        r = np.array(r)
        magnetic_susceptibility = np.array(magnetic_susceptibility)

        if len(r.shape) == 1:
            r = np.array([r])

        if b_field is None:
            b_field = self.B(r)

        magnetisation = (1 / mu_0) * (magnetic_susceptibility * b_field)
        energy_density = -1 * dot_product(magnetisation, b_field)

        return energy_density

    def _generate_1x1_config(self):
        self.remove_all_magnets()  # remove all existing magnets

        # add 1 up magnets to the MagnetArray at the centre
        self.add_magnet(position=np.array([0, 0, 0]), magnet=self.magnet_up)

    def _generate_1x2_config(self):
        self.remove_all_magnets()  # remove all existing magnets

        self.add_magnet(position=np.array([-5e-3, 0, 0]), magnet=self.magnet_up)
        self.add_magnet(position=np.array([5e-3, 0, 0]), magnet=self.magnet_down)

    def _generate_2x2_config(self):
        self.remove_all_magnets()  # remove all existing magnets

        # add 2 up magnets to the MagnetArray
        self.add_magnet(position=np.array([5e-3, 5e-3, 0]), magnet=self.magnet_up)
        self.add_magnet(position=np.array([-5e-3, -5e-3, 0]), magnet=self.magnet_up)

        # add 2 down magnets to the MagnetArray
        self.add_magnet(position=np.array([-5e-3, 5e-3, 0]), magnet=self.magnet_down)
        self.add_magnet(position=np.array([5e-3, -5e-3, 0]), magnet=self.magnet_down)

    def _generate_2x2up_config(self):
        self.remove_all_magnets()  # remove all existing magnets

        # add 2 up magnets to the MagnetArray
        self.add_magnet(position=np.array([5e-3, 5e-3, 0]), magnet=self.magnet_up)
        self.add_magnet(position=np.array([-5e-3, -5e-3, 0]), magnet=self.magnet_up)
        self.add_magnet(position=np.array([-5e-3, 5e-3, 0]), magnet=self.magnet_up)
        self.add_magnet(position=np.array([5e-3, -5e-3, 0]), magnet=self.magnet_up)

    def _generate_3x1_config(self):
        self.remove_all_magnets()  # remove all existing magnets

        # add 3 magnets in a column, with the surface of the top one being at [0, 0, 0]
        self.add_magnet(position=np.array([0, 0, 0]), magnet=self.magnet_up)
        self.add_magnet(position=np.array([0, 0, -5e-3]), magnet=self.magnet_up)
        self.add_magnet(position=np.array([0, 0, -15e-3]), magnet=self.magnet_up)

    def _generate_3x3_config(self):
        self.remove_all_magnets()  # remove all existing magnets

        # row 1
        self.add_magnet(position=np.array([-10e-3, 10e-3, 0]), magnet=self.magnet_up)
        self.add_magnet(position=np.array([0, 10e-3, 0]), magnet=self.magnet_down)
        self.add_magnet(position=np.array([10e-3, 10e-3, 0]), magnet=self.magnet_up)

        # row 2
        self.add_magnet(position=np.array([-10e-3, 0, 0]), magnet=self.magnet_down)
        self.add_magnet(position=np.array([0, 0, 0]), magnet=self.magnet_up)
        self.add_magnet(position=np.array([10e-3, 0, 0]), magnet=self.magnet_down)

        # row 3
        self.add_magnet(position=np.array([-10e-3, -10e-3, 0]), magnet=self.magnet_up)
        self.add_magnet(position=np.array([0, -10e-3, 0]), magnet=self.magnet_down)
        self.add_magnet(position=np.array([10e-3, -10e-3, 0]), magnet=self.magnet_up)

    def _generate_3x3invert_config(self):
        self.remove_all_magnets()  # remove all existing magnets

        # row 1
        self.add_magnet(position=np.array([-10e-3, 10e-3, 0]), magnet=self.magnet_down)
        self.add_magnet(position=np.array([0, 10e-3, 0]), magnet=self.magnet_up)
        self.add_magnet(position=np.array([10e-3, 10e-3, 0]), magnet=self.magnet_down)

        # row 2
        self.add_magnet(position=np.array([-10e-3, 0, 0]), magnet=self.magnet_up)
        self.add_magnet(position=np.array([0, 0, 0]), magnet=self.magnet_down)
        self.add_magnet(position=np.array([10e-3, 0, 0]), magnet=self.magnet_up)

        # row 3
        self.add_magnet(position=np.array([-10e-3, -10e-3, 0]), magnet=self.magnet_down)
        self.add_magnet(position=np.array([0, -10e-3, 0]), magnet=self.magnet_up)
        self.add_magnet(position=np.array([10e-3, -10e-3, 0]), magnet=self.magnet_down)

    def _generate_2x2invert_config(self):
        self.remove_all_magnets()  # remove all existing magnets

        # add 2 up magnets to the MagnetArray
        self.add_magnet(position=np.array([5e-3, 5e-3, 0]), magnet=self.magnet_down)
        self.add_magnet(position=np.array([-5e-3, -5e-3, 0]), magnet=self.magnet_down)

        # add 2 down magnets to the MagnetArray
        self.add_magnet(position=np.array([-5e-3, 5e-3, 0]), magnet=self.magnet_up)
        self.add_magnet(position=np.array([5e-3, -5e-3, 0]), magnet=self.magnet_up)
