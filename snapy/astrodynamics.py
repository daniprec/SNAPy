import numpy as np
from math import (sin, cos, acos)


def direction_cosine_matrix(theta1: float, theta2: float, theta3: float):
    """
    The Direction Cosine Matrix (DCM) is a 3 by 3 matrix
    that defines the rotations between two reference frames.
    theta are the Euler rotation angles (roll, pitch, yaw)
    """
    r1 = np.array([[1, 0, 0],
                    [0, cos(theta1), sin(theta1)],
                    [0, -sin(theta1), cos(theta1)]])
    r2 = np.array([[cos(theta2), 0, -sin(theta2)],
                    [0, 1, 0],
                    [sin(theta2), 0, cos(theta2)]])
    r3 = np.array([[cos(theta3), sin(theta3), 0],
                    [-sin(theta3), cos(theta3), 0],
                    [0, 0, 1]])
    c = np.dot(np.dot(r1, r2), r3)
    return c


def eigen_axis_rotation(c: np.array):
    assert c.shape == (3, 3)
    theta = acos(0.5 * (c[0,0] + c[1,1] + c[2,2] - 1))
    return theta


def quaternions(e: np.array, theta: float):
    """
    Quaternion elements do not carry a direct intuitive meaning.
    The Quaternion representation however simplifies the kinematic
    and dynamic equations and does not suffer from singularities
    which do occur in Euler angle representations.
    """
    q = e * sin(theta / 2)
    q4 = cos(theta / 2)
    return q, q4


def matrix_angular_rotation(w: np.array):
    omega = np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])
    return omega


def quartenions_dot(w: np.array, q: np.array, q4: float):
    q_dot = .5 * (q4 * w - np.cross(w, q))
    q4_dot = -.5 * np.dot(np.transpose(w), q)
    return q_dot, q4_dot


def dynamic_equation(J: np.array, w_dot: np.array, w: np.array):
    """
    Parameters
    ----------
    J : np.array
        Body's interia matrix
    Returns
    -------
    M : np.array
        External angular moment applied to the body's main axes
    """
    M = np.dot(J, w_dot) + np.cross(w_dot, np.dot(J, w))
    return M
    