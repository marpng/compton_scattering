"""
Fits simulated Compton scattering data to extract the electron mass.
Loads noisy energy vs. angle data, fits the Compton scattering formula
with electron mass as a free parameter
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants Task 3b
E_in = 0.6617  # MeV
sigma_E = 0.01  # MeV (assumed known detector resolution)

# Compton equation model
def compton_model(theta_deg, me):
    theta_rad = np.radians(theta_deg)
    return E_in / (1 + (E_in / me) * (1 - np.cos(theta_rad)))

def maximum_likelihood_fit(angles, E_measured, sigma_E=sigma_E):
    """
    Perform a maximum likelihood fit using the Compton model.
    Returns the fitted electron mass and its uncertainty.
    """
    popt, pcov = curve_fit(
        compton_model,
        angles,
        E_measured,
        p0=[0.5],
        sigma=np.full_like(E_measured, sigma_E),
        absolute_sigma=True,
        bounds=(0, np.inf)  # <-- Enforces physical range of mass
    )
    me_fit = popt[0]
    me_uncertainty = np.sqrt(pcov[0][0])
    return (me_fit, me_uncertainty)

def plot(angles, E_measured, me_fit, me_uncertainty):
    """
    Plot measured data and fitted Compton model.
    """
    print(f"Reconstructed electron mass: {me_fit:.5f} ± {me_uncertainty:.5f} MeV")
    theta_fit = np.linspace(10, 80, 500)
    E_fit = compton_model(theta_fit, me_fit)

    plt.errorbar(angles, E_measured, yerr=sigma_E, fmt='o', label='Measured')
    plt.plot(theta_fit, E_fit, label=f'Fit (me = {me_fit:.4f} MeV)', color='pink')
    plt.xlabel("Scattering Angle (degrees)")
    plt.ylabel("Photon Energy (MeV)")
    plt.title("Fit to Simulated Compton Scattering Data")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../results/fitted_mass.png")
    plt.show()

if __name__ == "__main__":
    data = np.loadtxt("../data/simulated_data.csv", delimiter=",", skiprows=1)
    angles = data[:, 0]
    E_measured = data[:, 1]
    fit, uncertainty = maximum_likelihood_fit(angles, E_measured)
    plot(angles, E_measured, fit, uncertainty)