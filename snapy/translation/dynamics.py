import numpy as np


def change_in_velocity_and_position(force: np.array, m_sat: float, dt: float):
    """Estimate the change in velocity and position. This should be done integrating,
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
    dx : numpy.array

    """
    acc = force / m_sat
    dv = acc * dt
    dx = dv * dt
    return dv, dx
