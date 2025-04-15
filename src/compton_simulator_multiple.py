import numpy as np
import matplotlib.pyplot as plt
import compton_simulator1
import fit_electron_mass

def repetitions():
    N = 1000
    masses = []
    uncertainties = []
    for i in range(N):
        angles_deg, E_measured, E_true = compton_simulator1.simulate_measurement()
        mass, uncertainty = fit_electron_mass.maximum_likelihood_fit(angles_deg, E_measured)
        masses.append(mass)
        uncertainties.append(uncertainty)
    return masses

def histogram(masses):
    plt.hist(masses, bins=30)
    plt.xlabel("Electron Mass (MeV)")
    plt.ylabel("Count")
    plt.title("Distribution of 1000 simulations of electron mass (MeV)")
    plt.savefig("../results/histogram_repetition.png")
    plt.show()

def distribution(masses):
    mean = np.mean(masses)
    std = np.std(masses)
    return mean, std

if __name__ == "__main__":
    np.random.seed(42)
    masses = repetitions()
    histogram(masses)
    mean, sd = distribution(masses)
    print(f"The mean of the distribution is: {round(mean, 5)}")
    print(f"The standard deviation of the distribution is: {round(sd, 5)}")