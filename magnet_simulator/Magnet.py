import numpy as np

from .types import TVector, TArrayOfVectors
from scipy.constants import mu_0, pi
from .utils import normalise, magnitude, create_grid, around, dot_product

from typing import Union


class Magnet:
    def __init__(self, magnetisation: TVector, cubic_section_sizes: float = None, cubic_section_count: int = None,
                 dimensions: TVector = (0.01, 0.01, 0.01), label=''):
        """
        Defines a new magnet. The magnet will be simulated using lots of smaller cubic magnets, The dimensions of the
        smaller cubic magnet is defined in cubic_section_sizes.

        :param magnetisation: A vector defining the magnetisation any volume of the magnet.
        :param cubic_section_sizes: A scalar defining the dimensions of each dimension of the smaller cubic magnets that
                                    make up the larger magnet. Either this or cubic_section_count needs to be provided.
        :param cubic_section_count: An integer defining the number of smaller magnets that make up the larger magnet.
                                    Either this or cubic_section_sizes needs to be provided.
        :param dimensions: The [x, y, z] dimensions of the magnet.
        """
        self.dimensions = np.array(dimensions)
        self.magnetisation = np.array(magnetisation)
        self.label = label

        if cubic_section_sizes is None and cubic_section_count is None:
            raise ValueError("Both 'cubic_section_sizes' and 'cubic_section_count' cannot be None.")
        elif cubic_section_sizes is not None and cubic_section_count is not None:
            raise ValueError("Provide only one of either 'cubic_section_sizes' and 'cubic_section_count'.")
        elif cubic_section_count is not None:
            cubic_section_sizes = around(self.dimensions / cubic_section_count)

        self.cubic_section_sizes = cubic_section_sizes
        self.section_sizes = np.full(3, self.cubic_section_sizes)

        self.number_of_sections: TVector = (self.dimensions / self.section_sizes).astype(int)

        # magnetic moment of each section of the magnet:
        self.section_magnetic_moment: TVector = self.magnetisation * np.prod(self.section_sizes)

        self.section_positions = self._generate_section_positions()

    def _generate_section_positions(self):
        x_start = -(self.dimensions[0] / 2) + (self.section_sizes[0] / 2)
        x_end = (self.dimensions[0] / 2) - (self.section_sizes[0] / 2)

        y_start = -(self.dimensions[1] / 2) + (self.section_sizes[1] / 2)
        y_end = (self.dimensions[1] / 2) - (self.section_sizes[1] / 2)

        z_start = -(self.dimensions[2] / 2) + (self.section_sizes[2] / 2)
        z_end = (self.dimensions[2] / 2) - (self.section_sizes[2] / 2)

        x_range = np.linspace(x_start, x_end, self.number_of_sections[0])
        y_range = np.linspace(y_start, y_end, self.number_of_sections[1])
        z_range = np.linspace(z_start, z_end, self.number_of_sections[2])

        section_positions = create_grid(x_range, y_range, z_range)
        return section_positions

    def describe(self) -> dict:
        """
        Describes the properties of the magnet.
        """
        return {
            'magnetisation': self.magnetisation,
            'dimensions': self.dimensions
        }

    # TODO: make this function take in an array of r values and compute it all completely vectorised
    def B(self, r: Union[TVector, TArrayOfVectors], method=1):
        """
        Calculates magnetic field at position r produced by the magnet. Equation from here
        https://en.wikipedia.org/wiki/Magnetic_moment#Magnetic_field_of_a_magnetic_moment.

        :param r: A numpy array or list of shape (3, ) or (x, 3). These are the points we want the B-field for. These
                vectors must be position vectors w.r.t. the centre of this Magnet.
        :param method: Either 1 or 2. Which method to use to calculate the B-field. Use 1.
        """
        r = np.array(r)

        if method == 1:
            # this is the fully vectorised method, this is significantly faster

            if len(r.shape) == 1:
                # this block runs if only a TVector was passed in, i.e., a single vector [x, y, z]

                # duplicates r vector into array of self.section_positions.shape so that they can be broadcast together
                r = np.full(self.section_positions.shape, r)
                rel_r = r - self.section_positions  # relative vector to r from every section_position
                rel_r_magnitude = magnitude(rel_r, axis=1)  # find the magnitude each relative vector in rel_r
                rel_r_normalised = rel_r / rel_r_magnitude[:, None]  # normalise rel_r
                section_magnetic_moment = np.full(rel_r.shape, self.section_magnetic_moment)  # similar to first line
                # above

                # compute the B-field at point r using the completely vectorised computation below.
                # b_field_components is the B-field at point r due to each of the magnet sections, therefore
                # b_field_components is an array.
                b_field_components = mu_0 / (4 * pi) * (
                        ((3 * rel_r_normalised * dot_product(rel_r_normalised, section_magnetic_moment)[:, None]) -
                         section_magnetic_moment) / (rel_r_magnitude ** 3)[:, None])

                total_b = np.sum(b_field_components, axis=0)  # total B-field as a sum of all the components

                return around(total_b)
            elif len(r.shape) == 2:
                # this block runs if a TArrayofVectors was passed in, i.e., [[x, y, z], [x, y, z], ...]

                def reshape(array_of_vectors: TArrayOfVectors, last_element_shape: int):
                    # reshapes a given array of vectors such
                    return array_of_vectors.reshape([r.shape[0], self.section_positions.shape[0], last_element_shape])

                r = reshape(np.repeat(r, self.section_positions.shape[0], axis=0), 3)
                rel_r = r - self.section_positions  # relative vector to r from every magnet section_position
                rel_r_magnitude = magnitude(rel_r, axis=2)  # find the magnitude each relative vector in rel_r
                rel_r_normalised = rel_r / reshape(rel_r_magnitude, 1)  # normalise rel_r
                section_magnetic_moment = np.full(rel_r.shape, self.section_magnetic_moment)

                b_field_components = mu_0 / (4 * pi) * (
                        ((3 * rel_r_normalised * reshape(dot_product(rel_r_normalised, section_magnetic_moment), 1)) -
                         section_magnetic_moment) / reshape((rel_r_magnitude ** 3), 1)
                )

                total_b = around(np.sum(b_field_components, axis=1))  # total B-field as a sum of all the components

                return total_b
            else:
                raise ValueError("Parameter 'r' has the wrong shape. Please provide an array of shape (3, ) or (x, 3).")

        elif method == 2:
            # this is the looping method, this is significantly slower, don't use this method
            # this also requires that y
            total_B = np.array([0, 0, 0])

            for position in self.section_positions:
                relative_r = r - position  # relative to a section

                total_B = total_B + np.array((mu_0 / (4 * pi)) * (
                        ((3 * normalise(relative_r) * np.dot(normalise(relative_r), self.section_magnetic_moment)) -
                         self.section_magnetic_moment) / magnitude(relative_r) ** 3))

            return around(total_B)
        else:
            raise ValueError("'method' must be either 1 or 2.")

    # TODO: make this function take in an array of r and m2 values and compute everything completely vectorised
    def F(self, r: TVector, m2: TVector) -> TVector:
        """
        Calculates the force exerted on m2 (magnetic moment 2) by this magnet (m1) where the relative vector from the
        centre of this magnet and m2 is r.

        Equation: https://en.wikipedia.org/wiki/Force_between_magnets#Magnetic_dipole%E2%80%93dipole_interaction
        Same Equation: https://en.wikipedia.org/wiki/Magnetic_moment section "Forces between two magnetic dipoles"

        The first equation above is the one implemented below.

        :param r: Relative vector [x, y, z] from centre of this magnet to m2.
        :param m2: Magnetic moment of magnet at r.

        :return: Force experienced by m2.
        """
        r = np.array(r)
        m2 = np.array(m2)

        # duplicates r vector into array of self.section_positions.shape so that they can be broadcast together
        r = np.full(self.section_positions.shape, r)
        rel_r = r - self.section_positions  # get the relative vector to r from every section_position
        rel_r_magnitude = magnitude(rel_r, axis=1)  # find the magnitude each relative vector in rel_r

        m1 = np.full(self.section_positions.shape, self.section_magnetic_moment)
        m2 = np.full(self.section_positions.shape, m2)

        f_components = ((3 * mu_0) / (4 * pi * (rel_r_magnitude ** 5)[:, None])) * \
                       (
                               (dot_product(m1, rel_r)[:, None] * m2) +
                               (dot_product(m2, rel_r)[:, None] * m1) +
                               (dot_product(m1, m2)[:, None] * rel_r) -
                               (rel_r * (5 * dot_product(m1, rel_r) *
                                         dot_product(m2, rel_r) / (rel_r_magnitude ** 2))[:, None])
                       )

        total_f = np.sum(f_components, axis=0)

        return around(total_f)




