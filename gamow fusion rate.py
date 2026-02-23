import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# ==========================
# Physical Constants (SI)
# ==========================
k = 1.380649e-23
e = 1.602176634e-19
m_p = 1.6726219e-27
c = 299792458
alpha = 1 / 137.036

Z1 = Z2 = 1
m_r = m_p / 2

# Astrophysical S-factor for proton-proton fusion
S_pp = 4e-22  # MeV*barn (approx constant used in stellar physics)

# ==========================
# Gamow Energy
# ==========================
def gamow_energy():
    return 2 * m_r * c**2 * (np.pi * alpha * Z1 * Z2)**2

# ==========================
# Quantum tunneling
# ==========================
def tunneling_probability(E):
    Eg = gamow_energy()
    return np.exp(-np.sqrt(Eg / E))

# ==========================
# Maxwell Boltzmann
# ==========================
def maxwell_boltzmann(E, T):
    return (2 / np.sqrt(np.pi)) * (E**0.5 / (k * T)**1.5) * np.exp(-E / (k * T))

# ==========================
# Fusion cross-section
# ==========================
def cross_section(E):
    # simplified nuclear cross section
    return S_pp * np.exp(-np.sqrt(gamow_energy() / E)) / E

# ==========================
# Fusion rate
# ==========================
def fusion_rate(E, T):
    mb = maxwell_boltzmann(E, T)
    sigma = cross_section(E)
    integrand = sigma * mb
    rate = simpson(integrand, E)
    return rate, integrand

# ==========================
# Gamow Peak Simulation
# ==========================
def gamow_peak_plot():
    temperatures = [1e7, 1.55e7, 2e7, 5e7]

    E = np.linspace(1e-18, 5e-15, 2000)
    E_keV = E / (e * 1000)

    plt.figure(figsize=(12,7))

    for T in temperatures:
        rate, integrand = fusion_rate(E, T)
        integrand /= np.max(integrand)

        peak_energy = E_keV[np.argmax(integrand)]

        print("Temperature:", T)
        print("Gamow Peak:", peak_energy, "keV")
        print("Fusion Rate:", rate)
        print("------")

        plt.plot(E_keV, integrand, label=f"T={T:.2e} K")

    plt.title("Gamow Peak for Proton Fusion")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Normalized Fusion Probability")
    plt.legend()
    plt.grid(True)

    plt.savefig("gamow_peak_research.png", dpi=300)
    plt.show()

# ==========================
# Temperature Sweep Study
# ==========================
def temperature_study():

    temperatures = np.linspace(5e6, 1e8, 50)
    rates = []

    E = np.linspace(1e-18, 5e-15, 2000)

    for T in temperatures:
        rate, _ = fusion_rate(E, T)
        rates.append(rate)

    plt.figure(figsize=(10,6))
    plt.plot(temperatures, rates)
    plt.xlabel("Core Temperature (K)")
    plt.ylabel("Fusion Rate")
    plt.title("Fusion Rate vs Stellar Temperature")
    plt.grid(True)

    plt.savefig("fusion_rate_vs_temperature.png", dpi=300)
    plt.show()

# ==========================
# Run Research Simulation
# ==========================
gamow_peak_plot()
temperature_study()
