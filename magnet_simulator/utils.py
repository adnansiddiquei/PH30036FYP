from collections.abc import Iterable
from typing import Union, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .types import TVector, TArrayOfVectors, TArrayOfScalars

matplotlib.rcParams['figure.figsize'] = (14, 8)

# optimum magnet magnetisation
magnetisation = {
    'mean': np.array([0, 0, 1.107e6]),
    'upper': np.array([0, 0, 1.1075e6]),
    'lower': np.array([0, 0, 1.1065e6]),
    'error': np.array([0, 0, 0.0005e6])
}


class LinearReg:
    def __init__(self):
        self.reg = None
        self.intercept_ = None
        self.coef_ = None

    def fit(self, x: TArrayOfScalars, y: TArrayOfScalars):
        x = np.array(x).reshape([len(x), 1])
        y = np.array(y).reshape([len(y), 1])

        self.reg = LinearRegression().fit(x, y)
        self.intercept_ = self.reg.intercept_[0]
        self.coef_ = self.reg.coef_[0][0]
        return self

    def predict(self, x: TArrayOfScalars):
        res = self.reg.predict(np.array(x).reshape([len(x), 1]))
        res.reshape(len(res))
        return res


def is_iterable(obj) -> bool:
    """
    Checks if the passed object is an iterable.
    """
    return isinstance(obj, Iterable)


def calc_magnetic_dipole_moment(volume: float, magnetisation: Union[TVector, float]) -> Union[TVector, float]:
    """
    Calculate the magnetic dipole moment of a magnetised volume V given the magnetisation M of the volume. M can
    be either either a scalar or a vector.
    """
    if is_iterable(magnetisation):
        return volume * np.array(magnetisation)
    else:
        return volume * magnetisation


def normalise(vector: TVector) -> TVector:
    """
    Calculate and return the unit vector for the vector passed in
    """
    return np.array(vector) / magnitude(vector)


def magnitude(vector: np.ndarray, axis=None) -> Union[float, np.ndarray]:
    """
    Calculates the magnitude of the vector.
    """
    return np.linalg.norm(vector, axis=axis)


def dot_product(a: np.ndarray, b: np.ndarray):
    """
    Returns dot product of an array of vectors a and b. Each vector in a is dot with the vector in b at the
    same positions.

    If a.shape == (x, 3), then b.shape must be (x, 3) and the returned array has shape (x, ).
    If a.shape == (y, x, 3) then b.shape must be (y, x, 3) and the returned array has shape (y, x)

    :param a: 2D array of shape (x, 3).
    :param b: 2D array of shape (x, 3).

    :return: 1D array of shape (x, ).
    """
    return np.sum(a * b, axis=(len(a.shape) - 1))


def xy_rotate(vectors: Union[TVector, TArrayOfVectors], theta: float) -> TArrayOfVectors:
    """
    Rotate an array of vectors (or single vector) about the [0, 0, z] axis by angle theta.

    :param vectors: Array of vectors, or single vector to rotate.
    :param theta: angle in degrees.

    :return: rotated array of vectors, or single vector
    """
    vectors = np.array(vectors)
    theta = np.deg2rad(theta)  # convert degrees to radians

    rotation_vector = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    return np.dot(rotation_vector, vectors.T).T


def create_grid(x_range, y_range, z_range=None):
    """
    Creates a 2D or 3D grid of points from the specified range.
    """
    if z_range is not None:
        xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
        xx, yy, zz = xx.flatten(), yy.flatten(), zz.flatten()
        points = np.stack([xx, yy, zz], axis=1)
    else:
        xx, yy = np.meshgrid(x_range, y_range)
        xx, yy = xx.flatten(), yy.flatten()
        points = np.stack([xx, yy], axis=1)

    points = around(np.array(points))

    return points


def create_heatmap(xyzb_points, z=None):
    xyzb_points = np.array(xyzb_points)

    if z is None:
        all_z = list(set(xyzb_points[:, 2]))

        if len(all_z) == 1:
            z = all_z[0]
        else:
            raise ValueError("'z' must be provided if multiple z-values exist in 'xyzb_points'.")

    extent = [
        np.min(xyzb_points[:, 0]),  # x-min
        np.max(xyzb_points[:, 0]),  # x-max
        np.min(xyzb_points[:, 1]),  # y-min
        np.max(xyzb_points[:, 1])  # y-max
    ]

    x_res = len(set(xyzb_points[:, 0]))
    y_res = len(set(xyzb_points[:, 1]))

    heatmap = np.zeros((x_res, y_res)) * np.nan

    xyzb_points_sorted = pd.DataFrame({
        'x': xyzb_points[:, 0],
        'y': xyzb_points[:, 1],
        'z': xyzb_points[:, 2],
        'b': xyzb_points[:, 3],
    }).sort_values(by=['y', 'x'], ascending=[False, True]).reset_index(drop=True)

    row = 0
    for y in range(y_res):
        for x in range(x_res):
            heatmap[y, x] = xyzb_points_sorted.iloc[row]['b']
            row += 1

    return heatmap, extent


def plot_heatmap(heatmap, extent, magnet_array=None, title=None, x_label=None, y_label=None, cbar_label=None,
                 cmap='bwr', cbar_on=True):
    plt.clf()
    plt.imshow(heatmap, extent=extent, cmap=cmap)

    if cbar_on:
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(cbar_label, rotation=90, fontsize=14)

    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xticks(fontsize=12), plt.yticks(fontsize=12)

    if magnet_array is not None:
        for magnet in magnet_array._magnets:
            pos = magnet['position']
            plt.text(pos[0], pos[1], magnet['magnet'].label, fontsize=36, ha='center', va='center')

        if magnet_array.config[0:3] == '2x2':
            plt.axvline(0, c='black'), plt.axvline(0.01, c='black'), plt.axvline(-0.01, c='black')
            plt.axhline(0, c='black'), plt.axhline(0.01, c='black'), plt.axhline(-0.01, c='black')

    # plt.show()


def around(value: np.ndarray):
    return np.around(value, 12)


def create_xyz_plane(x_range: Tuple[float, float], y_range: Tuple[float, float], resolution: int, z: float):
    """ Generates a grid of xy-points at constant z """
    x_values = np.linspace(x_range[0], x_range[1], resolution)
    y_values = np.linspace(y_range[0], y_range[1], resolution)
    xy_points = create_grid(x_values, y_values)
    xyz_points = np.array([[xy_point[0], xy_point[1], z] for xy_point in xy_points])

    return xyz_points


def identify_axis(axis: str) -> int:
    if axis.lower() == 'x':
        return 0
    elif axis.lower() == 'y':
        return 1
    elif axis.lower() == 'z':
        return 2
    else:
        raise ValueError("'component' must be one of either 'x', 'y' or 'z'.")


def reduce_xyz_vectors(xyzb_points, axis: str) -> np.ndarray:
    axis = identify_axis(axis)
    return np.array([[i[0], i[1], i[2], i[3][axis]] for i in xyzb_points])
