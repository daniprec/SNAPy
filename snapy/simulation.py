import datetime

import numpy as np
import typer

from snapy.torque.astrodynamics import (
    angular_velocity_change,
    compute_ecef_dcm,
    euler_angles_change,
    direction_cosine_matrix,
    rotate_ecef_dcm,
)
from snapy.torque.gravity import gravity_torque_smart, gravitational_force
from snapy.torque.magnetism import (
    earth_magnetic_field,
    magnetic_torque,
    magnetic_field_change,
    hysteresis_torque,
)
from snapy.translation.dynamics import compute_velocity_and_position
from snapy.utils.conf import load_conf


class Simulation:
    def __init__(self, cfg):
        self.m_sat = cfg["structure"]["m"]
        self.inertia = cfg["structure"]["inertia"]

        self.cfg_magnets = cfg["magnets"]
        self.cfg_rods = cfg["rods"]
        self.b_rods = cfg["rods"]["b"]
        self.cfg_shield = cfg["shielding"]
        self.b_shield = cfg["shielding"]["b"]

        self.x = cfg["simulation"]["x"]
        self.v = cfg["simulation"]["v"]
        self.w = cfg["simulation"]["w"]
        self.dt = cfg["simulation"]["dt"]
        self.date_start = cfg["simulation"]["date"]
        self.date = datetime.datetime.strptime(self.date_start, "%Y/%m/%d %H:%M:%S")

        # Euler angles
        self.thetas = np.zeros(3)
        self._compute_rotation_matrices()

        # Magnetic field
        self.b_earth = earth_magnetic_field(self.x, self.c_ei)
        self._update_earth_magnetic_field()

        # Torques
        self.m = np.zeros(3)

    def _update_velocity_and_position(self):
        f_g = gravitational_force(self.x, self.m_sat)
        self.v, self.x = compute_velocity_and_position(
            self.x, self.v, f_g, self.m_sat, self.dt
        )

    def _update_date(self):
        self.date = self.date + datetime.timedelta(seconds=self.dt)

    def _compute_rotation_matrices(self):
        # Rotation matrices
        # ECI to body frame
        self.c_bi = direction_cosine_matrix(self.thetas)
        # ECI to ECEF
        self.c_ei = compute_ecef_dcm(self.date_start)

    def _update_rotation_matrices(self):
        # Rotation matrices
        # ECI to body frame
        self.c_bi = direction_cosine_matrix(self.thetas)
        # ECI to ECEF
        self.c_ei = rotate_ecef_dcm(self.c_ei, self.dt)

    def _update_earth_magnetic_field(self):
        self.b_earth_prev = self.b_earth
        self.b_earth = earth_magnetic_field(self.x, self.c_ei)
        self.dhdt = magnetic_field_change(self.b_earth_prev, self.b_earth, self.dt)

    def _compute_torques(self):
        # Gravitational torque
        m_gg = gravity_torque_smart(self.x, self.c_bi, self.inertia)

        # Permanent magnets torque
        m_mag = magnetic_torque(self.cfg_magnets["mag"], self.b_earth)

        # Hysteresis rods torque
        m_rods, self.b_rods = hysteresis_torque(
            self.b_rods, self.b_earth, self.dhdt, self.dt, self.c_bi, self.cfg_rods
        )

        # Magnetic shield torque
        m_shield, self.b_shield = hysteresis_torque(
            self.b_shield, self.b_earth, self.dhdt, self.dt, self.c_bi, self.cfg_shield
        )

        # All torques
        self.m = m_gg + m_mag + m_rods + m_shield

    def _rotate_satellite(self):
        self.w = angular_velocity_change(self.inertia, self.w, self.m, self.dt)
        self.thetas = euler_angles_change(self.thetas, self.w, self.dt)

    def step(self):
        self._update_velocity_and_position()
        self._update_date()
        self._update_rotation_matrices()
        self._update_earth_magnetic_field()
        self._compute_torques()
        self._rotate_satellite()


def _write_vector(vec: np.array, f: int):
    s = f"{round(vec[0], f)};{round(vec[1], f)};{round(vec[2], f)}"
    return s


def run_simulation(
    time: float = 86400,
    dt_store: float = 10,
    path_conf: str = "snapy/config.toml",
    path_output: str = "output.csv",
    round_float: int = 2,
):
    """Run the simulation

    Parameters
    ----------
    time : float, optional
        Time that the simulation will simulate, in seconds, by defaul 86400
    dt_store : float
        Time step at which data is stored, in seconds, by default 1
    path_conf : str, optional
    path_output : str, optional
    round_float : int, optional

    """
    # Load the config file
    cfg = load_conf(path_conf)
    # Start the simulation
    sim = Simulation(cfg)

    # Compute number of steps
    length = int(time / sim.dt)
    print(f"Doing {length} steps")
    dt_check = int(dt_store / sim.dt)

    f = round_float

    with open(path_output, "w") as text_file:
        text_file.write(
            "date;x;y;z;w_roll;w_pitch;w_yaw;" "theta_roll;theta_pitch;theta_yaw\n"
        )

    for i in range(length):
        if i % dt_check == 0:
            with open(path_output, "a") as text_file:
                text_file.write(
                    f"{sim.date};"
                    f"{_write_vector(sim.x, f)};"
                    f"{_write_vector(sim.w, f)};"
                    f"{_write_vector(sim.thetas, f)}\n"
                )
        sim.step()


if __name__ == "__main__":
    typer.run(run_simulation)
