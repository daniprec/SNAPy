import datetime
from math import pi
from math import sin, cos, acos

import numpy as np

from snapy.constants import C_EI, DATE_EI


def direction_cosine_matrix(thetas: np.array) -> np.array:
    """The Direction Cosine Matrix (DCM) is a 3 by 3 matrix that defines the
    rotations between two reference frames.

    Parameters
    ----------
    thetas : numpy.array
        The Euler rotation angles (roll, pitch, yaw), in radians

    Returns
    -------
    c : numpy.array
        Direction Cosine Matrix

    """
    r1 = np.array(
        [
            [1, 0, 0],
            [0, cos(thetas[0]), sin(thetas[0])],
            [0, -sin(thetas[0]), cos(thetas[0])],
        ]
    )
    r2 = np.array(
        [
            [cos(thetas[1]), 0, -sin(thetas[1])],
            [0, 1, 0],
            [sin(thetas[1]), 0, cos(thetas[1])],
        ]
    )
    r3 = np.array(
        [
            [cos(thetas[2]), sin(thetas[2]), 0],
            [-sin(thetas[2]), cos(thetas[2]), 0],
            [0, 0, 1],
        ]
    )
    c = np.dot(np.dot(r1, r2), r3)
    return c


def rotate_frame(x: np.array, thetas: np.array) -> np.array:
    """Rotate to one reference frame to the other

    Parameters
    ----------
    x : numpy.array
        Reference frame
    thetas : numpy.array
        The Euler rotation angles (roll, pitch, yaw), in radians

    Returns
    -------
    x_c : numpy.array
        The new reference frame, rotated according to the Euler angles

    """
    c = direction_cosine_matrix(thetas)
    x_c = np.dot(c, x)
    return x_c


def eigen_axis_rotation(c: np.array) -> float:
    """Angle of DCM

    Parameters
    ----------
    c : numpy.array
        Direction Cosine Matrix

    Returns
    -------
    theta : float
        Angle in radians

    """
    try:
        theta = acos(0.5 * (c[0, 0] + c[1, 1] + c[2, 2] - 1))
    except IndexError:
        raise IndexError(
            f"The shape of DCM matrix must be (3,3). Given DCM was " f"{c.shape}"
        )

    return theta


def quaternions(e: np.array, theta: float) -> tuple:
    """Quaternion elements do not carry a direct intuitive meaning.
    The Quaternion representation however simplifies the kinematic and dynamic
    equations and does not suffer from singularities which do occur in Euler angle
    representations.

    Parameters
    ----------
    e : numpy.array
    theta : float

    Returns
    -------
    q : numpy.array
    q4 : float

    """
    q = e * sin(theta / 2)
    q4 = cos(theta / 2)
    return q, q4


def matrix_angular_rotation(w: np.array) -> np.array:
    """

    Parameters
    ----------
    w : numpy.array

    Returns
    -------
    omega : numpy.array

    """
    try:
        omega = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    except IndexError:
        raise IndexError(f"w vector must have length 3. Given w is length {len(w)}")
    return omega


def quartenions_dot(w: np.array, q: np.array, q4: float) -> tuple:
    """

    Parameters
    ----------
    w : numpy.array
    q : numpy.array
    q4 : float

    Returns
    -------
    q_dot : numpy.array
    q4_dot : float

    """
    q_dot = 0.5 * (q4 * w - np.cross(w, q))
    q4_dot = -0.5 * np.dot(np.transpose(w), q)
    return q_dot, q4_dot


def dynamic_equation(j: np.array, w_dot: np.array, w: np.array):
    """

    Parameters
    ----------
    j : numpy.array
        Body's interia matrix
    w_dot : numpy.array
    w : numpy.array

    Returns
    -------
    m_body : np.array
        External angular moment applied to the body's main axes

    """
    m = np.dot(j, w_dot) + np.cross(w_dot, np.dot(j, w))
    return m


def nadir_vector(x: np.array) -> np.array:
    """

    Parameters
    ----------
    x : numpy.array
        Position relative to ECI, in meters

    Returns
    -------
    u_e : numpy.array
        Nadir vector

    """
    u_e = -x / np.linalg.norm(x)
    return u_e


def rotate_ecef_dcm(c_ei: np.array, t: float) -> np.array:
    """Rotates the matrix c_ei (ECI to ECEF) according to Earth's rotation

    Parameters
    ----------
    c_ei : numpy.array
        Original c_ei matrix
    t : float
        Time in the future, in seconds

    Returns
    -------
    c_ei_new : numpy.array
        c_ei after t seconds

    """
    thetas = np.array([-2 * pi * t / 24 / 60 / 60, 0, 0])
    dcm = direction_cosine_matrix(thetas)
    c_ei_new = c_ei * dcm
    return c_ei_new


def compute_ecef_dcm(date: str) -> np.array:
    """Computes the rotation matrix from ECI to ECEF at a given date

    Parameters
    ----------
    date : str
        Date when the matrix is computed, in format "%Y/%m/%d %H:%M:%S"

    Returns
    -------
    c_ei : numpy.array
        Rotation matrix from ECI to ECEF
    
    """
    date_ei = datetime.datetime.strptime(DATE_EI, "%Y/%m/%d %H:%M:%S.%f")
    date = datetime.datetime.strptime(date, "%Y/%m/%d %H:%M:%S.%f")
    t = (date - date_ei).total_seconds()
    c_ei = rotate_ecef_dcm(C_EI, t)
    return c_ei
