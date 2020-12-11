import numpy as np

from snapy.astrodynamics import nadir_vector
from snapy.constants import MU0, R_EARTH


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
    m : float
        The magnetic dipole in Amperes * meters squared

    """
    m = i * n * area
    return m


def magnetic_dipole_magnet(b: np.array, v: float):
    """The magnetic dipole of a permanent magnet, or any material

    Parameters
    ----------
    b : numpy.array
        Magnetic flux density of the magnet, in Tesla
    v : float
        Volume of the material, in cubic meters

    Returns
    -------
    m : numpy.array
        The magnetic dipole in Amperes * meters squared

    """
    m = b * v * MU0
    return m


def magnetic_field(x: np.array, h0: float, u_m: np.array, c_ei: np.array) -> np.array:
    """Based on the development of the Dipole Model, the magnetic field at a certain
    point in orbit is calculated

    Parameters
    ----------
    x : numpy.array
        Position vector in ECI
    h0 : float
        Dipole strength in Webers * meter
    u_m : numpy.array
        Unit vector along the magnetic dipole in ECEF
    c_ei : numpy.array
        Rotation matrix from ECI to ECEF. It is time dependent as the ECEF frame
        rotates  about ECI

    Returns
    -------
    b : numpy.array
        Magnetic field in ECI, in Tesla

    """
    # Position vector is first rotated into ECEF
    x_ecef = c_ei * x
    # Unit vector at which the magnetic field is calculated in ECEF
    u_x = nadir_vector(x_ecef)
    # The magnetic field is computed in ECEF
    b = ((R_EARTH ** 3) * h0 / np.norm(x_ecef) ^ 3) * (3 * np.dot(u_m, u_x) * u_x - u_m)
    # The calculated value of the magnetic field is rotated to eci
    c_ie = np.linalg.inv(c_ei)
    b_eci = c_ie * b
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


def hysteresis_torque(
    hysteresis_curve: str, v_hyst: np.array, b_earth: np.array, c_bi: np.array
) -> np.array:
    """

    Parameters
    ----------
    hysteresis_curve : str
        Name of the hysteresis loop function
    v_hyst : numpy.array
        Volume of the hysteresis material along the three axes, in cubic meters
    b_earth : numpy.array
        Earth magnetic flux density vector in ECI
    c_bi : np.array
        Rotation matrix from the ECI frame to the body frame describing the attitude
        of the satellite

    Returns
    -------
    m_hyst : numpy.array
        Magnetic torque of the hysteresis material

    """
    # Rotate vector into bodyframe
    b_earth_body = c_bi * b_earth
    # Magnetic field density
    h_earth = (1 / MU0) * b_earth_body
    # Approximated hysteresis loop model
    try:
        b_hyst = eval(hysteresis_curve)(h_earth)
    except ValueError:
        raise ValueError(f"Unrecognised hysteresis loop function: {hysteresis_curve}")
    # Magnetic moment of the hysteresis material
    mag_hyst = b_hyst * v_hyst / MU0
    # Torque of the hysteresis material
    m_hyst = magnetic_torque(mag_hyst, b_earth_body)
    return m_hyst
