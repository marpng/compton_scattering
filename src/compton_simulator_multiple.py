import numpy as np
import matplotlib.pyplot as plt
import compton_simulator1
import fit_electron_mass

N = 1000
sigma_E = 0.01  # Energy resolution of detector in MeV

def repetitions(sigma_E=sigma_E):
    masses = []
    uncertainties = []
    for i in range(N):
        try:
            angles_deg, E_measured, E_true = compton_simulator1.simulate_measurement(sigma_E)
            mass, uncertainty = fit_electron_mass.maximum_likelihood_fit(angles_deg, E_measured, sigma_E)

            # Skip unphysical or broken fits
            if mass <= 0 or uncertainty <= 0:
                continue

            masses.append(mass)
            uncertainties.append(uncertainty)
        except Exception as e:
            print(f"Fit failed on repetition {i}: {e}")
            continue
    return masses, uncertainties

def histogram(masses, fig_name="../results/histogram_repetition.png"):
    plt.hist(masses, bins=30)
    plt.xlabel("Electron Mass (MeV)")
    plt.ylabel("Count")
    plt.title("Distribution of 1000 simulations of electron mass (MeV)")
    plt.savefig(fig_name)
    plt.show()

def distribution(masses):
    mean = np.mean(masses)
    std = np.std(masses)
    print("\n--- Distribution Statistics ---")
    print(f"Mean:               {round(mean, 5)}")
    print(f"Standard Deviation: {round(std, 5)}")
    print("-" * 60)
    return mean, std

def pull_distribution(masses, uncertainties, fig_name="../results/histogram_pull.png"):
    """
    Calculate and plot the pull distribution for the reconstructed electron mass.

    The pull is defined as:
        pull = (measured_mass - true_mass) / uncertainty

    Args:
        masses (list): List of measured electron masses (in MeV).
        uncertainties (list): List of uncertainties associated with the measured masses.
        fig_name (str): Path to save the histogram plot.
     """
    pulls = []
    for i in range(len(masses)):
        pull = (masses[i] - 0.511) / uncertainties[i]
        pulls.append(pull)
    plt.hist(pulls, bins=30, density=True)
    plt.xlabel("Pull")
    plt.ylabel("Probability Density")
    plt.title("Distribution of Pull")
    plt.savefig(fig_name)
    plt.show()

    mean_pull, std_pull = np.mean(pulls), np.std(pulls)
    print("\n--- Pull Distribution Statistics ---")
    print(f"Mean of pull:               {mean_pull:.5f}")
    print(f"Standard deviation of pull: {std_pull:.5f}")
    print("-" * 60)

def uncertainty_vs_resolution():
    """
    Compare the uncertainty of the electron mass with different energy resolutions.

    This function simulates the measurement of the electron mass with two different energy resolutions
    (0.05 MeV and 0.1 MeV) and generates histograms for the mass distribution and pull distribution.
    """
    energy_resolutions = [0.05, 0.1]

    for sigma_E in energy_resolutions:
        fig_name_repetition = f"../results/histogram_repetition_{sigma_E}_MeV.png"
        fig_name_pull = f"../results/histogram_pull_{sigma_E}_MeV.png"

        print(f"\n=== Energy resolution: {sigma_E} MeV ===")
        masses, uncertainties = repetitions(sigma_E)
        histogram(masses, fig_name_repetition)
        mean, sd = distribution(masses)
        pull_distribution(masses, uncertainties, fig_name_pull)
        print("=" * 60)

if __name__ == "__main__":
    np.random.seed(42)
    masses, uncertainties = repetitions()
    histogram(masses)
    mean, sd = distribution(masses)
    pull_distribution(masses, uncertainties)
    uncertainty_vs_resolution()
