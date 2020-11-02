import numpy as np


M_EARTH = 5.972 * 1e24  # kg
G = 6.67408 * 1e-11  # m3 kg-1 s-2
MU = 3.896 * 1e14  # m3 s−2

def gravitational_force_at_position(x, m_sat):
    x2 = np.dot(x, x)
    # Acceleration
    a = G * M_EARTH / x2
    # Nadir vector
    nadir = - x / np.linalg.norm(x)
    # Gravitational force
    f_grav = a * m_sat * nadir
    return f_grav


def gravity_gradient(x, inertia):
    # Distance to the center
    r = np.linalg.norm(x)
    # Nadir vector
    nadir = - x / r
    # Gravity gradient torque
    m_gg = (3 * MU / r**3) * np.cross(nadir, np.dot(inertia, nadir))
    return m_gg
