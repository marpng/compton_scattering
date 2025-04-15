import numpy as np
import matplotlib.pyplot as plt

# constants for Task 3a
me_c2 = 0.511  # Electron rest mass energy in MeV
E_in = 0.6617  # Photon energy from 137Cs in MeV
sigma_E = 0.01  # Energy resolution of detector in MeV
mean = 0.0

def compton_energy(E_in, theta_deg):
    """Compute Compton scattered photon energy at given angle."""
    theta_rad = np.radians(theta_deg)
    E_out = E_in / (1 + (E_in / me_c2) * (1 - np.cos(theta_rad)))
    return E_out

def simulate_measurement(sigma_E=sigma_E):
    # Scattering angles from 10° to 80° in steps of 10
    angles_deg = np.arange(10, 90, 10)
    E_true = compton_energy(E_in, angles_deg)

    # Add Gaussian noise to simulate detector measurement
    noise = np.random.normal(loc=mean, scale=sigma_E, size=E_true.shape)
    E_measured = E_true + noise
    return (angles_deg, E_measured, E_true)

def save_and_plot(angles_deg, E_measured, E_true):
    np.savetxt("../data/simulated_data.csv", np.column_stack([angles_deg, E_measured]),
               delimiter=",", header="angle_deg,E_measured", comments='')


    plt.errorbar(angles_deg, E_measured, yerr=sigma_E, fmt='o', label="Measured")
    plt.plot(angles_deg, E_true, '--', label="True Energy", color='purple')
    plt.xlabel("Scattering Angle (degrees)")
    plt.ylabel("Photon Energy (MeV)")
    plt.title("Simulated Compton Scattering Measurement")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("../results/simulated_measurement.png")
    plt.show()

if __name__ == "__main__":
    angles, measured, true = simulate_measurement()
    save_and_plot(angles, measured, true)