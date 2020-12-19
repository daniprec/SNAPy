"""
Physic constants
"""
import numpy as np

# Equatorial radius of Earth (m)
R_EARTH = 6.371 * 1e6
# Mass of Earth (kg)
M_EARTH = 5.972 * 1e24
# Dipole strength (Wb * m)
H0 = 7.9430e15 / R_EARTH**3
# Earth magnetic dipole (A * m2)
MAG_EARTH = np.array([-0.0653, 0.1865, -0.9803])

# Gravitational constant (m3 kg-1 s-2)
G = 6.67408 * 1e-11
# Geocentric gravitational constant (m3 s−2)
MU = G * M_EARTH

# Magnetic permeability of free space (m kg s-2 A-2)
MU0 = 1.25663706 * 1e-6

# ECI to ECEF
C_EI = np.array([
    [-0.1839, 0.9829, 0.0004],
    [-0.9829, -0.1839, 0.0020],
    [0.0020, 0.0000, 1.0000],
])  # Reference ECI to ECEF DCM at given date
DATE_EI = "2021/01/01 00:00:00"  # Time of reference DCM
