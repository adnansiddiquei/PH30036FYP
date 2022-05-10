import numpy as np
from decimal import Decimal

from .types import TVector
from .Magnet import Magnet
from .utils import around


class Holed10mmMagnet(Magnet):
    def __init__(self, magnetisation: TVector, cubic_section_sizes: float, label=None):
        """
        Generates a 10mm cube magnet that has a 2mm hole running through the y-axis of the centre of the magnet.

        :param magnetisation: Magnetisation of magnet in A/m
        :param cubic_section_sizes: Size of miniature magnets that simulate the larger magnet, all being cubic
        """
        self.dimensions = np.array([10e-3, 10e-3, 10e-3])  # dimensions of magnet, 10mm cube
        self.hole_size = 2e-3  # dimensions of hole running through y-axis of magnet
        self.cubic_section_sizes = cubic_section_sizes
        self.number_of_sections = (self.dimensions / self.cubic_section_sizes).astype(int)

        super().__init__(
            magnetisation,
            cubic_section_sizes=self.cubic_section_sizes,
            dimensions=self.dimensions,
            label=label
        )  # call the super classes __init__ function

        if (Decimal(f'{self.hole_size}') % Decimal(f'{cubic_section_sizes}')) != Decimal('0'):
            raise ValueError('2mm hole must be exactly divisible by "section_size".')
            # this is so that the 2mm hole can be modelled as a cuboid

        # magnetic moment of each section of the magnet:
        self.section_magnetic_moment: TVector = self.magnetisation * np.prod(self.section_sizes)

        self.section_positions = self._generate_section_positions()

        # now remove all the section positions that correspond to the 2mm hold running through the y-axis of the magnet
        self.section_positions, self.hole_positions = self._remove_2mm_hole()

    def _remove_2mm_hole(self):
        xyz_range = np.linspace(
            -(self.dimensions[0] / 2) + (self.section_sizes[0] / 2),
            (self.dimensions[0] / 2) - (self.section_sizes[0] / 2),
            self.number_of_sections[0]
        )  # this is the range for all xyz co-ordinates in the magnet grid
        # e.g., for cubic_section_size = 1e-3
        # xyz_range = [-0.0045, -0.0035, -0.0025, -0.0015, -0.0005,  0.0005,  0.0015, 0.0025,  0.0035,  0.0045]

        indexes_to_delete = np.arange(
            int(4e-3/self.cubic_section_sizes),
            len(xyz_range) - int(4e-3/self.cubic_section_sizes)
        )  # indexes in xyz_range to delete, in the above example, this would be [4, 5]

        coords_to_delete = around(np.array([xyz_range[i] for i in indexes_to_delete]))
        # coords_to_delete = [-0.0005,  0.0005] in the above example.
        # This means, if the 2mm hole runs through the y-axis then we want to delete the following points
        # [0.0005, y, 0.0005] , [0.0005, y, -0.0005], [-0.0005, y, 0.0005] , [-0.0005, y, -0.0005] for all y

        section_position_indexes_to_delete = np.array([]).astype(int)

        for index, section_position in zip(range(len(self.section_positions)), self.section_positions):
            if (section_position[0] in coords_to_delete) & (section_position[2] in coords_to_delete):
                section_position_indexes_to_delete = np.concatenate([
                    section_position_indexes_to_delete,
                    np.array([index])
                ])

        hole_positions = np.array([self.section_positions[i] for i in section_position_indexes_to_delete])
        section_positions = np.delete(self.section_positions, section_position_indexes_to_delete, axis=0)

        return section_positions, hole_positions

