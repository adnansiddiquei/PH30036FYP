import numpy as np
from typing import Tuple, List
from scipy.constants import mu_0, g

import pandas as pd
from .types import TVector, TArrayOfVectors, TArrayOfScalars
from .utils import create_grid, create_xyz_plane, magnetisation as M, identify_axis

from .Graphite import Graphite, graphite_properties
from .Grid import Grid
from .Holed10mmMagnet import Holed10mmMagnet
from .MagnetArray import MagnetArray
from .NewtonRaphson import NewtonRaphson

from sklearn.linear_model import LinearRegression


class Simulation:
    """
    This class automatically simulates certain useful things given a populated Grid object.
    """

    def __init__(self, grid: Grid):
        self.grid = grid
        self.magnet_array = self.grid.magnet_array  # for easier referencing
        self.graphite = self.grid.graphite  # for easier referencing

    @staticmethod
    def quick_grid_setup(magnetisation=M['mean'], cubic_section_sizes=0.4e-3, config='2x2invert',
                         graphite_number_of_sections=(10, 10, 3), graphite_rotation=0, magnet=Holed10mmMagnet,
                         graphite_piece='1cm', magnet_array_origin=(0, 0, -5e-3), graphite_dimensions=None,
                         graphite_density=None) -> Tuple[Grid, Graphite, MagnetArray]:
        """
        Quickly configures a Grid with a piece of Graphite and a MagnetArray with the provided parameters. Returns all 3
        objects.
        """
        magnetisation = np.array(magnetisation)
        graphite_number_of_sections = np.array(graphite_number_of_sections)

        magnet_up = magnet(magnetisation=magnetisation, cubic_section_sizes=cubic_section_sizes, label='N')
        magnet_down = magnet(magnetisation=-magnetisation, cubic_section_sizes=cubic_section_sizes, label='S')
        magnet_array = MagnetArray(magnet=magnet, magnet_up=magnet_up, magnet_down=magnet_down, config=config)

        if graphite_dimensions is None:
            graphite_dimensions = graphite_properties[graphite_piece]['dimensions']

        if graphite_density is None:
            graphite_density = graphite_properties[graphite_piece]['density']

        graphite = Graphite(number_of_sections=graphite_number_of_sections, rotation=graphite_rotation,
                            dimensions=graphite_dimensions, density=graphite_density)

        grid = Grid(magnet_array=magnet_array, graphite=graphite, magnet_array_origin=magnet_array_origin)

        return grid, graphite, magnet_array

    @staticmethod
    def calc_steps(limits: Tuple[int, int], step_size: float, number_of_steps: int):
        """
        Used to calculate the values of intermediate steps for rotation or position. [TODO: Add a better description]
        """
        if step_size is not None and number_of_steps is not None:
            raise ValueError("Only provide one of 'step_size' or 'number_of_steps'.")
        elif step_size is not None:
            # i.e., if the user has provided a value for step_size, but not number_of_steps
            number_of_steps = round((limits[1] - limits[0]) / step_size) + 1
            return np.linspace(limits[0], limits[1], number_of_steps)
        elif number_of_steps is not None:
            return np.linspace(limits[0], limits[1], number_of_steps)
        elif step_size is None and number_of_steps is None:
            raise ValueError("Both 'step_size' and 'number_of_steps' cannot be None. One Parameter must be provided.")

    def magnetic_potential_energy_vs_rotation(self, limits=(0, 360), step_size: float = None,
                                              number_of_steps: int = None, graphite_position: TVector = None):
        """
        Calculates the magnetic potential energy of the graphite as it is rotated in the B-field.

        :param limits: The lower and upper limits for the rotation, in degrees.
        :param step_size: The step size, in degrees. Either this one or number_of_steps needs to be provided.
        :param number_of_steps: The number of steps to compute between limits[0] and limits[1]
        :param graphite_position: The position of the graphite while it is being rotated.

        :return:
        """
        all_rotations = Simulation.calc_steps(limits, step_size, number_of_steps)

        if graphite_position is not None:
            # re-position the Graphite if a new position was passed in, but don't bother computing anything yet
            self.grid.alter_graphite(
                new_position=graphite_position, compute_forces=False, compute_energies=False, compute_torque=False
            )

        # this will hold the total magnetic potential energy at each of the rotations
        magnetic_potential_energy = np.array([])

        for rotation in all_rotations:
            self.grid.alter_graphite(new_rotation=rotation, compute_forces=False, compute_torque=False)
            magnetic_potential_energy = np.append(
                magnetic_potential_energy, self.grid.graphite_properties['total_magnetic_potential_energy'])

        results = pd.DataFrame({
            'rotation': all_rotations,
            'magnetic_potential_energy': magnetic_potential_energy
        })

        return results

    def magnetic_potential_energy_vs_height(self, limits, xy_position=(0, 0), step_size: float = None,
                                            number_of_steps: int = None):
        """
        The potential energy of the graphite as it is lifted vertically through a series of z points at constant xy.

        This function also calculates the force acting on the graphite by using taking F = -dU/delta_broadcast using the CDA method
        to compute dU/delta_broadcast. This is computed using Eqn 5 / Eqn 6 in the "Determining magnetic susceptibilities of everyday
        materials using an electronic balance", Laumann. D., Heusler. S.

        :param limits: The lower and upper limits for the z co-ord.
        :param xy_position: The xy co-ords.
        :param step_size: The step size. Either this one or number_of_steps needs to be provided.
        :param number_of_steps: The number of steps to compute between limits[0] and limits[1]

        :return:
        """
        all_heights = Simulation.calc_steps(limits, step_size, number_of_steps)

        # this will hold the total magnetic potential energy at each of the rotations
        magnetic_potential_energy = np.array([])

        for height in all_heights:
            self.grid.alter_graphite(new_position=np.concatenate([xy_position, [height]]), compute_forces=False)
            magnetic_potential_energy = np.append(
                magnetic_potential_energy, self.grid.graphite_properties['total_magnetic_potential_energy'])

        results = pd.DataFrame({
            'height': all_heights,
            'magnetic_potential_energy': magnetic_potential_energy
        })

        # calculate the force on the graphite due to the magnets using the CDA method
        results['magnetic_force'] = \
            -(results['magnetic_potential_energy'].diff(2).shift(-1) / results['height'].diff(2).shift(-1)) / 2

        # calculate total force
        results['total_force'] = results['magnetic_force'] + self.graphite.gravity_force[2]

        return results

    # TODO: graph might be inverted?
    def torque_vs_rotation(self, graphite_position: TVector, limits=(0, 90), step_size: float = None,
                           number_of_steps: int = None, clockwise_rotation=False, log=False):
        all_rotations = Simulation.calc_steps(limits, step_size, number_of_steps)
        all_rotations = all_rotations * -1 if clockwise_rotation else all_rotations

        if graphite_position is not None:
            # re-position the Graphite if a new position was passed in, but don't bother computing anything yet
            self.grid.alter_graphite(
                new_position=graphite_position, compute_forces=False, compute_energies=False, compute_torque=False
            )

        # this will hold the total torque at each of the rotations
        torques = np.empty((0, 3), float)

        for rotation in all_rotations:
            self.grid.alter_graphite(new_rotation=rotation, compute_energies=False)
            torques = np.append(torques, [self.grid.graphite_properties['total_torque']], axis=0)

            if log:
                print(rotation, self.grid.graphite_properties['total_torque'])

        results = pd.DataFrame({
            'rotation': all_rotations * -1 if clockwise_rotation else all_rotations,
            'torque_x': torques[:, 0],
            'torque_y': torques[:, 1],
            'torque_z': torques[:, 2],
        })

        return results

    # TODO: check if this is correct -- 3x3 not working??
    def calc_equiforce_surface(self, x_range: Tuple[float, float], y_range: Tuple[float, float], resolution: int,
                               z_force: float = None, initial_guess: float = 2e-3, allowed_z_error: float = 0.0005e-3,
                               CDA_a=0.0001e-3, max_NR_iterations_per_point: int = 10) -> TArrayOfVectors:
        """
        This function finds a surface where the z-component of the force on the Graphite is constant.

        :param x_range: x_range to compute surface within. E.g., [-5e-3, 5e-3]
        :param y_range: y_range to compute surface within. E.g., [-5e-3, 5e-3]
        :param resolution: The resolution of the xy grid points, e.g., 10. This is the number of points to use in each
                        dimension.
        :param z_force: The z-force to solve the surface for. If left blank, this function will find the 0 force surface
        :param initial_guess: The initial guess in the z-coordinate for the first iteration.
        :param allowed_z_error: The allowed error in the computed value of z.
        :param CDA_a: The value of a to use for CDA (centered difference approximation)
        :param max_NR_iterations_per_point: The max number of newton raphson iterations allowed per point.

        :return: An array of vectors, where each vector represents a point where the graphite will have the same
            z-component of force.
        """
        x_values = np.linspace(x_range[0], x_range[1], resolution)
        y_values = np.linspace(y_range[0], y_range[1], resolution)
        xy_points = create_grid(x_values, y_values)

        equiforce_surface = np.empty((0, 3), float)

        if z_force is None:
            # if no z_force is specified then we can assume that we are looking for the 0 force surface, i.e, the
            # surface where the graphite would sit stably in the z-axis
            z_force = -self.graphite.gravity_force[2]

        for xy in xy_points:
            def f(z):
                self.grid.alter_graphite(new_position=np.concatenate([xy, [z]]), compute_energies=False,
                                         compute_torque=False, compute_force_components=(False, False, True))
                return self.grid.graphite_properties['total_magnetic_force'][2] - z_force

            nr = NewtonRaphson(f, initial_guess, allowed_z_error, CDA_a, max_NR_iterations_per_point)
            z_actual = nr.solve()

            equiforce_surface = np.append(equiforce_surface, [np.concatenate([xy, [z_actual]])], axis=0)
            initial_guess = z_actual

        return equiforce_surface

    def calc_levitation_height(self, xy_position, rotation: float, initial_guess: float = 2e-3,
                               allowed_z_error: float = 0.00005e-3, CDA_a=0.0001e-3,
                               max_NR_iterations_per_point: int = 20):
        self.grid.alter_graphite(new_position=np.array([0, 0, initial_guess]), new_rotation=rotation, compute_forces=False,
                                 compute_energies=False, compute_torque=False,
                                 compute_force_components=(True, False, False))

        def f(z):
            self.grid.alter_graphite(new_position=np.concatenate([xy_position, [z]]),
                                     compute_energies=False, compute_torque=False,
                                     compute_force_components=(False, False, True))
            return self.grid.graphite_properties['total_force'][2]

        nr = NewtonRaphson(f, initial_guess, allowed_z_error, CDA_a, max_NR_iterations_per_point)
        z_actual = nr.solve()

        return z_actual, nr

    def calc_horizontal_stability_zone(self, yz_position, rotation: float, initial_guess: float = 0.5e-3,
                                       allowed_x_error: float = 0.00005e-3, CDA_a=0.0001e-3,
                                       max_NR_iterations_per_point: int = 20):
        self.grid.alter_graphite(new_rotation=rotation, compute_forces=False, compute_energies=False,
                                 compute_torque=False, compute_force_components=(True, False, False))

        def f(x):
            self.grid.alter_graphite(new_position=np.concatenate([[x], yz_position]),
                                     compute_energies=False, compute_torque=False,
                                     compute_force_components=(True, False, False))
            return self.grid.graphite_properties['total_force'][0]

        nr = NewtonRaphson(f, initial_guess, allowed_x_error, CDA_a, max_NR_iterations_per_point)
        z_actual = nr.solve()

        return z_actual, nr

    def calc_levitation_height_vs_side_length(self, xy_position, rotation: float, side_lengths: List[float],
                                              initial_guesses: List[float] = None):
        if initial_guesses is None:
            initial_guesses = np.full_like(side_lengths, 2e-3)

        levitation_heights = np.array([])

        for side_length, initial_guess in zip(side_lengths, initial_guesses):
            self.graphite.reconfigure(
                rotation=rotation, dimensions=[side_length, side_length, self.graphite.section_dimensions[2]]
            )

            levitation_height, nr = self.calc_levitation_height(xy_position, rotation, initial_guess=initial_guess)

            levitation_heights = np.append(levitation_heights, levitation_height)

        results = pd.DataFrame({
            'side_length': side_lengths,
            'levitation_height': levitation_heights
        })

        return results

    def calc_magnetic_susceptibility(self, graphite_positions: TArrayOfVectors, mass_differences: TArrayOfScalars):
        """
        Provided an array of z heights

        :param graphite_positions:
        :param mass_differences:
        :return:
        """
        results = pd.DataFrame(columns=['height', 'mass_difference', 'magnetic_susceptibility'])

        for graphite_position, mass_difference in zip(graphite_positions, mass_differences):
            self.grid.alter_graphite(new_position=graphite_position, compute_forces=False, compute_energies=False,
                                     compute_torque=False)

            # compute (dB^2)/delta_broadcast for every graphite section
            dBdz = self.magnet_array.dBd('z', r=self.grid.graphite_properties['rel_section_positions'], power=2)

            # multiply each one by the volume of each section
            dBdzdv = dBdz * self.graphite.section_volume

            # this is the integral component in Eqn 7
            integral_component = np.sum(dBdzdv[:, 2])

            results = pd.concat([
                results,
                pd.DataFrame({
                    'height': graphite_position[2],
                    'mass_difference': mass_difference,
                    'magnetic_susceptibility': (2 * mu_0 * mass_difference * g) * (1 / integral_component)
                }, index=[0])
            ], ignore_index=True)

        return results

    def calc_b_field_map(self, x_range: Tuple[float, float], y_range: Tuple[float, float], resolution: int, z: float):
        """
        :param x_range:
        :param y_range:
        :param resolution:
        :param z:

        :return:
        """
        xyz_points = create_xyz_plane(x_range, y_range, resolution, z)

        b_field = self.grid.B(xyz_points)

        xyzb_points = [
            [
                grid_point[0], grid_point[1], grid_point[2], b
            ] for grid_point, b in zip(xyz_points, b_field)
        ]

        return xyzb_points

    def calc_potential_energy_density_map(self, x_range: Tuple[float, float], y_range: Tuple[float, float],
                                          resolution: int, z: float):
        xyz_points = create_xyz_plane(x_range, y_range, resolution, z)
        energy_densities = self.grid.energy_density(xyz_points)

        xyzu_points = [
            [
                grid_point[0], grid_point[1], grid_point[2], energy_density
            ] for grid_point, energy_density in zip(xyz_points, energy_densities)
        ]

        return xyzu_points

    def calc_force_density_map(self, x_range: Tuple[float, float], y_range: Tuple[float, float], resolution: int,
                               z: float):
        xyz_points = create_xyz_plane(x_range, y_range, resolution, z)

        magnetic_force_densities = self.grid.force_density(xyz_points)

        xyzf_points = [
            [
                grid_point[0], grid_point[1], grid_point[2], magnetic_force_density
            ] for grid_point, magnetic_force_density in zip(xyz_points, magnetic_force_densities)
        ]

        return xyzf_points

    def calc_torque_tangent(self, graphite_position: TVector, rotation: float, rad=False):
        limits = np.array([rotation * 0.99, rotation * 1.01])

        torques = self.torque_vs_rotation(
            graphite_position=graphite_position, limits=limits, number_of_steps=2
        )

        tz = torques['torque_z'].values

        if rad:
            limits = np.deg2rad(limits)

        reg = LinearRegression().fit(
            limits.reshape([len(limits), 1]),
            tz.reshape([len(tz), 1])
        )

        return reg

    def calc_f_tangent(self, graphite_position: TVector, rotation: float, axis: str, delta: float = 0.01e-3):
        axis = identify_axis(axis)
        compute_force_components = (
            True if axis == 0 else False,
            True if axis == 1 else False,
            True if axis == 2 else False
        )

        self.grid.alter_graphite(new_position=graphite_position, new_rotation=rotation, compute_energies=False,
                                 compute_torque=False, compute_force_components=compute_force_components)
        f0 = self.grid.graphite_properties['total_force'][axis]

        new_graphite_position = np.array([i for i in graphite_position])
        new_graphite_position[axis] += new_graphite_position[axis] + delta

        self.grid.alter_graphite(new_position=new_graphite_position, compute_energies=False, compute_torque=False,
                                 compute_force_components=compute_force_components)
        f1 = self.grid.graphite_properties['total_force'][axis]

        reg = LinearRegression().fit(
            [[graphite_position[axis]], [new_graphite_position[axis]]],
            [[f0], [f1]]
        )

        return reg
