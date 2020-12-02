import numpy as np

from snapy.astrodynamics import rotate_frame

M_EARTH = 5.972 * 1e24  # kg
G = 6.67408 * 1e-11  # m3 kg-1 s-2
MU = 3.896 * 1e14  # m3 s−2


def gravity_acceleration(x: np.array, m: float = M_EARTH) -> np.array:
    """Acceleration due to some mass' gravity at a certain point in space
    By default the mass is the Earth, so coordinates are given in ECI

    Parameters
    ----------
    x : numpy.array
        Position relative to the mass, in meters
    m : float, optional
        Mass of the body producing the gravity, in kilograms. By default 5.972 * 1e24 kg

    Returns
    -------
    a : numpy.array
        Acceleration relative to the mass, in meters per seconds squared

    """
    a = G * m / np.dot(x, x)
    return a


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


def gravitational_force(x: np.array, m_sat: float, m_body: float = M_EARTH) -> np.array:
    """

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
    a = gravity_acceleration(x, m=m_body)
    nadir = nadir_vector(x)
    f_g = a * nadir * m_sat
    return f_g


def gravity_gradient(x: np.array, thetas: np.array, inertia: np.array) -> np.array:
    """

    Parameters
    ----------
    x : numpy.array
    thetas : numpy.array
    inertia : numpy.array

    Returns
    -------
    m_gg : numpy.array
        Gravity gradient torque relative to bodyframe, in Newtons * meters

    """
    # Distance to the center (ECI)
    r = np.linalg.norm(x)
    # Bodyframe position
    x_body = rotate_frame(x, thetas)
    # Nadir vector (body)
    u_e = nadir_vector(x_body)
    # Gravity gradient torque
    m_gg = (3 * MU / r ** 3) * np.cross(u_e, np.dot(inertia, u_e))
    return m_gg


def gravity_gradient_smart(
    x: np.array, thetas: np.array, inertia: np.array
) -> np.array:
    """Same as gravity_gradient but reduces the number of computations

    Parameters
    ----------
    x : numpy.array
    thetas : numpy.array
    inertia : numpy.array

    Returns
    -------
    m_gg : numpy.array
        Gravity gradient torque relative to bodyframe, in Newtons * meters

    """
    # Bodyframe position
    x_body = rotate_frame(x, thetas)
    # Nadir vector (body)
    u_e = nadir_vector(x_body)
    # Gravity gradient torque
    m_gg = (3 * G * M_EARTH / (np.dot(x, x) ** 2)) * np.cross(
        -x_body, np.dot(inertia, u_e)
    )
    return m_gg
