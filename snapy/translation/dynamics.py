import numpy as np


def change_in_velocity(force: np.array, m_sat: float, dt: float) -> np.array:
    """Estimate the change in velocity. This should be done integrating,
    but with small time deltas we expected that a raw computation will be close enough

    Parameters
    ----------
    force : numpy.array
        Force affecting the satellite, in ECI and Newtons
    m_sat : float
    dt : float

    Returns
    -------
    dv : numpy.array

    """
    acc = force / m_sat
    dv = acc * dt
    return dv


def compute_velocity_and_position(
    x: np.array, v: np.array, force: np.array, m_sat: float, dt: float
):
    """Estimate the velocity and position, given a force affecting the satellite during
    a small time delta

    Parameters
    ----------
    x : numpy.array
        Satellite position in ECI
    v : numpy.array
        Satellite velocity in ECI
    force : numpy.array
        Force affecting the satellite, in ECI and Newtons
    m_sat : float
    dt : float

    Returns
    -------
    v_new  : numpy.array
    x_new : numpy.array

    """
    dv = change_in_velocity(force, m_sat, dt)
    v_new = v + dv
    x_new = x + v_new * dt
    return v_new, x_new
