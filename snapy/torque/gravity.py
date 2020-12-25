import numpy as np

from snapy.torque.astrodynamics import nadir_vector
from snapy.utils.constants import G, MU, M_EARTH


def gravity_acceleration(x: np.array, m_body: float = M_EARTH) -> np.array:
    """Acceleration due to some mass' gravity at a certain point in space
    By default the mass is the Earth, so coordinates are given in ECI

    Parameters
    ----------
    x : numpy.array
        Position relative to the mass, in meters
    m_body : float, optional
        Mass of the body producing the gravity, in kilograms. By default Earth's

    Returns
    -------
    a : numpy.array
        Acceleration relative to the mass, in meters per seconds squared

    """
    a = G * m_body / np.dot(x, x)
    return a


def gravitational_force(x: np.array, m_sat: float, m_body: float = M_EARTH) -> np.array:
    """Gravitational force due to a huge mass (usually Earth) being acting on the
    satellite

    Parameters
    ----------
    x : numpy.array
        Position relative to gravity pull, in meters
    m_sat : float
        Satellite's mass, in kilograms
    m_body : float, optional
        Mass of the body producing the gravity, in kilograms. By default 5.972 * 1e24 kg

    Returns
    -------
    f_g : np.array
        Gravitational force, in Newtons

    """
    a = gravity_acceleration(x, m_body=m_body)
    nadir = nadir_vector(x)
    f_g = a * nadir * m_sat
    return f_g


def gravity_torque(x: np.array, c_bi: np.array, inertia: np.array) -> np.array:
    """The gravity gradient angular moment

    Parameters
    ----------
    x : numpy.array
        Position relative to gravity pull, in meters
    c_bi : numpy.array
        Rotation matrix from ECI to body frame
    inertia : numpy.array
        Inertia matrix J, in kilograms times meter squared

    Returns
    -------
    m_gg : numpy.array
        Gravity gradient torque relative to body frame, in Newtons * meters

    """
    # Distance to the center (ECI)
    r = np.linalg.norm(x)
    # The position vector is rotated into the body frame
    x_body = np.dot(c_bi, x)
    # Nadir vector (body)
    u_e = nadir_vector(x_body)
    # Gravity gradient torque
    m_gg = (3 * MU / r ** 3) * np.cross(u_e, np.dot(inertia, u_e))
    return m_gg


def gravity_torque_smart(x: np.array, c_bi: np.array, inertia: np.array) -> np.array:
    """Same as gravity_torque but reduces the number of computations

    Parameters
    ----------
    x : numpy.array
        Position relative to gravity pull, in meters
    c_bi : numpy.array
        Rotation matrix from ECI to body frame
    inertia : numpy.array
        Inertia matrix J, in kilograms times meter squared

    Returns
    -------
    m_gg : numpy.array
        Gravity gradient torque relative to body frame, in Newtons * meters

    """
    # The position vector is rotated into the body frame
    x_body = np.dot(c_bi, x)
    # Nadir vector (body frame)
    u_e = nadir_vector(x_body)
    # Gravity gradient torque
    m_gg = (3 * G * M_EARTH / (np.dot(x, x) ** 2)) * np.cross(
        -x_body, np.dot(inertia, u_e)
    )
    return m_gg
