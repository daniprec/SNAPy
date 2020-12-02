import numpy as np

from snapy.astrodynamics import rotate_frame


M_EARTH = 5.972 * 1e24  # kg
G = 6.67408 * 1e-11  # m3 kg-1 s-2
MU = 3.896 * 1e14  # m3 s−2


def gravity_acceleration(x: np.array):
    """
    Parameters
    ----------
    x : np.array
        In ECI
    """
    a = G * M_EARTH / np.dot(x, x)
    return a


def nadir_vector(x: np.array):
    """
    Parameters
    ----------
    x : np.array
        In ECI
    """
    u_e = - x / np.linalg.norm(x)
    return u_e


def gravitational_force(x: np.array, m_sat: float):
    """
    Parameters
    ----------
    x : np.array
        In ECI
    Returns
    -------
    f_g : np.array
        Gravitational force (N)
    """
    a = gravity_acceleration(x)
    nadir = nadir_vector(x)
    f_g = a * nadir * m_sat
    return f_g


def gravity_gradient(x: np.array, thetas: np.array,
                     inertia: np.array):
    """
    Returns
    -------
    m_gg : np.array
        Gravity gradient torque, in bodyframe (N m)
    """
    # Distance to the center (ECI)
    r = np.linalg.norm(x)
    # Bodyframe position
    x_body = rotate_frame(x, thetas)
    # Nadir vector (body)
    u_e = nadir_vector(x_body)
    # Gravity gradient torque
    m_gg = (3 * MU / r**3) * np.cross(u_e, np.dot(inertia, u_e))
    return m_gg


def gravity_gradient_smart(x: np.array, thetas: np.array,
                           inertia: np.array):
    """
    Same as gravity_gradient but reduces
    the number of computations
    """
    # Bodyframe position
    x_body = rotate_frame(x, thetas)
    # Nadir vector (body)
    u_e = nadir_vector(x_body)
    # Gravity gradient torque
    m_gg = ((3 * G * M_EARTH / (np.dot(x, x)**2))
            * np.cross(-x_body, np.dot(inertia, u_e)))
    return m_gg
    