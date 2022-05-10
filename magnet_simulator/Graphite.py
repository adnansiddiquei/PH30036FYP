import numpy as np
from scipy.constants import g

from .utils import create_grid, xy_rotate, around


# these are the properties of the pieces of Graphite that were used in the experiment
graphite_properties = {
    '1cm': {
        'dimensions': np.array([0.0103761, 0.0103761, 0.00104757]),  # m
        'density': 2147.4280  # kg/m^3,
    },
    '1.6cm': {
        'dimensions': np.array([0.016169167, 0.016169167, 0.00103671]),  # m
        'density': 2148.7555  # kg/m^3
    },
    'magnetic_susceptibility': np.array([0.85, 0.85, 4.5]) * -1e-4
}


class Graphite:
    def __init__(self, density=graphite_properties['1cm']['density'],
                 dimensions=graphite_properties['1cm']['dimensions'], number_of_sections=(10, 10, 1), rotation=0,
                 magnetic_susceptibility=graphite_properties['magnetic_susceptibility']):
        """
        :param density: kg per cubic metre (kg/m^3)
        :param dimensions: (x-length, y-length, z-length) in metres
        :param number_of_sections: how many sections to split the graphite into when integrating the force over the entire
                         graphite.
        """
        self.magnetic_susceptibility = magnetic_susceptibility
        self.density = density
        self.dimensions = np.array(dimensions)
        self.number_of_sections = np.array(number_of_sections)
        self.rotation = rotation

        self.mass = self.density * np.prod(self.dimensions)  # mass of entire piece of graphite

        self.gravity_force = np.array([0, 0, self.mass * -g])  # the force on the graphite due to gravity

        # [x, y, z] dimensions of each section
        self.section_dimensions = self.dimensions / self.number_of_sections

        # volume of each section
        self.section_volume = np.prod(self.section_dimensions)

        # mass of each section of the graphite after it has been split into the specified resolution
        self.section_mass = self.mass / np.prod(self.number_of_sections)

        # the force on a section of the graphite due to gravity
        self.section_gravity_force = np.array([0, 0, self.section_mass * -g])

        # generate the co-ordinates for all the sections
        self.section_positions = self._generate_section_positions()

        self.moment_of_inertia = (1/12) * self.mass * ((self.dimensions[0]**2) + (self.dimensions[1]**2))

    def reconfigure(self, rotation=None, number_of_sections=None, dimensions=None):
        self.rotation = rotation if rotation is not None else self.rotation
        self.number_of_sections = number_of_sections if number_of_sections is not None else self.number_of_sections
        self.dimensions = dimensions if dimensions is not None else self.dimensions

        self.__init__(self.density, self.dimensions, number_of_sections=self.number_of_sections, rotation=self.rotation,
                      magnetic_susceptibility=self.magnetic_susceptibility)

    def _generate_section_positions(self):
        x_start = -(self.dimensions[0] / 2) + (self.section_dimensions[0] / 2)
        x_end = (self.dimensions[0] / 2) - (self.section_dimensions[0] / 2)

        y_start = -(self.dimensions[1] / 2) + (self.section_dimensions[1] / 2)
        y_end = (self.dimensions[1] / 2) - (self.section_dimensions[1] / 2)

        z_start = -(self.dimensions[2] / 2) + (self.section_dimensions[2] / 2)
        z_end = (self.dimensions[2] / 2) - (self.section_dimensions[2] / 2)

        x_range = np.linspace(x_start, x_end, self.number_of_sections[0])
        y_range = np.linspace(y_start, y_end, self.number_of_sections[1])
        z_range = np.linspace(z_start, z_end, self.number_of_sections[2])

        section_positions = create_grid(x_range, y_range, z_range)
        section_positions = xy_rotate(section_positions, self.rotation)  # rotate all vectors
        section_positions = around(section_positions)

        return section_positions

    def describe(self):
        return {
            'dimensions': self.dimensions,
            'density': self.density,
            'mass': self.mass,
            'gravity_force': self.gravity_force,
            'number_of_sections': self.number_of_sections,
            'section_mass': self.section_mass,
            'section_dimensions': self.section_dimensions,
            'section_volume': self.section_volume,
            'section_gravity_force': self.section_gravity_force,
        }
