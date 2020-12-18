from math import pi

import numpy as np

from snapy.astrodynamics import nadir_vector
from snapy.constants import MU0, R_EARTH, H0, MAG_EARTH


def magnetic_dipole_coil(i: float, n: int, area: float) -> float:
    """Magnetic dipole for a current coil

    Parameters
    ----------
    i : float
        Current through the coil, in Amperes
    n : int
        Number of turns
    area : float
        Area of the coil, in meters squared

    Returns
    -------
    mag : float
        The magnetic dipole in Amperes * meters squared

    """
    mag = i * n * area
    return mag


def magnetic_dipole_magnet(b: np.array, vol: float):
    """The magnetic dipole of a permanent magnet, or any material

    Parameters
    ----------
    b : numpy.array
        Magnetic flux density of the magnet, in Tesla
    vol : float
        Volume of the material, in cubic meters

    Returns
    -------
    mag : numpy.array
        The magnetic dipole in Amperes * meters squared

    """
    mag = b * vol * MU0
    return mag


def earth_magnetic_field(x: np.array, c_ei: np.array) -> np.array:
    """Based on the development of the Dipole Model, the magnetic field at a certain
    point in orbit is calculated

    Parameters
    ----------
    x : numpy.array
        Position vector in ECI
    c_ei : numpy.array
        Rotation matrix from ECI to ECEF. It is time dependent as the ECEF frame
        rotates  about ECI

    Returns
    -------
    b : numpy.array
        Magnetic field in ECI, in Tesla

    """
    # Position vector is first rotated into ECEF
    x_ecef = np.dot(c_ei, x)
    # Unit vector at which the magnetic field is calculated in ECEF
    u_x = nadir_vector(x_ecef)
    # TODO: Ensure U_M is in ECEF
    u_m = MAG_EARTH
    # The magnetic field is computed in ECEF
    b = ((R_EARTH ** 3) * H0 / np.norm(x_ecef) ^ 3) * (3 * np.dot(u_m, u_x) * u_x - u_m)
    # The calculated value of the magnetic field is rotated to eci
    c_ie = np.linalg.inv(c_ei)
    b_eci = np.dot(c_ie, b)
    return b_eci


def magnetic_torque(mag: np.array, b_earth: np.array) -> np.array:
    """The torque produced by a magnetic dipole

    Parameters
    ----------
    mag : numpy.array
        Magnetic dipole moment in Amperes * squared meters
    b_earth : numpy.array
        Earth magnetic flux density vector in ECI

    Returns
    -------
    m_mag : numpy.array
        Magnetic torque

    """
    m_mag = np.cross(mag, b_earth)
    return m_mag


def hysteresis_loop(
    b: np.array,
    h: np.array,
    dhdt: np.array,
    dt: float,
    b_s: float,
    h_c: float,
    k: float,
    p: float,
    q0: float,
) -> np.array:
    """

    Parameters
    ----------
    b : numpy.array
        Magnetic field of the material, in Tesla
    h : numpy.array
        Outside magnetic field strength, in Amperes per meter
    dhdt : numpy.array
        dH / dt, in Amperes per meter per second
    dt : float
        Differential of time, in seconds
    b_s : float
        Saturation of the material, in Tesla
    h_c : float
        Coercitivity of the material, in Amperes per meter
    k : float
        Constant, in meters per Ampere
    p : float
        Exponent of the fractional distance F
    q0 : float
        Value of Q for F = 0

    Returns
    -------
    b_new : numpy.array
        New magnetic field of the material, in Tesla

    """

    # Value of H on the left boundary curve corresponding to B, HL (A / m)
    h_l = np.tan(pi * np.divide(b, b_s) / 2) / k - h_c

    # Boundary curve slope, BP (G * m / A)
    bp = 2 * k * b_s / pi * np.cos(pi * np.divide(b, b_s) / 2) ** 2

    # Fractional distance, F
    f = (h - h_l) / 2 / h_c
    # It is contained between [1, -1]
    f[f > 1] = 1
    f[f < -1] = -1
    # If dH/dt is negative, measure F from the right hand boundary
    f[dhdt < 0] = 1 - f

    # Sign of F is relevant, and as such f**p must maintain the sign of F
    # We save the signs of F
    f_sign = np.sign(f)
    f = np.abs(f)

    # Boundary slope multiplier, Q
    q = q0 + (1 - q0) * f_sign * np.power(f, p)

    # dB / dt (G / s)
    dbdt = q * bp * dhdt

    # Compute the new magnetic field of the material
    b_new = b + dbdt * dt

    return b_new


def hysteresis_torque(
    b_hyst: np.array,
    b_earth: np.array,
    dhdt: float,
    dt: float,
    c_bi: np.array,
    cfg_hyst: dict,
) -> np.array:
    """

    Parameters
    ----------
    b_hyst : numpy.array
        Magnetic field of the material, in Tesla
    b_earth : numpy.array
        Earth magnetic flux density vector in ECI
    dhdt : numpy.array
        dH / dt, in Amperes per meter per second
    dt : float
        Differential of time, in seconds
    c_bi : np.array
        Rotation matrix from the ECI frame to the body frame describing the attitude
        of the satellite
    cfg_hyst : dict
        Configuration of the hysteresis material

    Returns
    -------
    m_hyst : numpy.array
        Magnetic torque of the hysteresis material
    b_hyst : numpy.array
        New magnetic field of the material, in Tesla

    """
    # Rotate vector into body frame
    b_earth_body = np.dot(c_bi, b_earth)
    # Magnetic field density
    h_earth = (1 / MU0) * b_earth_body
    # New magnetic field of the material
    b_hyst = hysteresis_loop(
        b_hyst,
        h_earth,
        dhdt,
        dt,
        cfg_hyst["b_s"],
        cfg_hyst["h_c"],
        cfg_hyst["k"],
        cfg_hyst["p"],
        cfg_hyst["q0"],
    )
    # Magnetic moment of the hysteresis material
    mag_hyst = b_hyst * cfg_hyst["vol"] / MU0
    # Torque of the hysteresis material
    m_hyst = magnetic_torque(mag_hyst, b_earth_body)
    return m_hyst, b_hyst
