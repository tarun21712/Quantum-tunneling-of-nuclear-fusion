import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# ==========================
# Physical Constants (SI)
# ==========================
k = 1.380649e-23        # Boltzmann constant (J/K)
e = 1.602176634e-19     # Electron charge (C)
m_p = 1.6726219e-27     # Proton mass (kg)
c = 299792458           # Speed of light (m/s)
alpha = 1 / 137.036     # Fine structure constant
hbar = 1.054571817e-34

# Proton charges
Z1 = Z2 = 1

# Reduced mass
m_r = m_p / 2

# ==========================
# Gamow Energy
# ==========================
def gamow_energy():
    return 2 * m_r * c**2 * (np.pi * alpha * Z1 * Z2)**2

# ==========================
# Quantum Tunneling
# ==========================
def tunneling_probability(E):
    Eg = gamow_energy()
    return np.exp(-np.sqrt(Eg / E))

# ==========================
# Maxwell-Boltzmann Distribution
# ==========================
def maxwell_boltzmann(E, T):
    return (2 / np.sqrt(np.pi)) * (E**0.5 / (k * T)**1.5) * np.exp(-E / (k * T))

# ==========================
# Fusion Rate Calculation
# ==========================
def fusion_rate(E, T):
    prob = tunneling_probability(E)
    dist = maxwell_boltzmann(E, T)

    integrand = prob * dist

    # Numerical integration
    rate = simpson(integrand, E)

    return rate, integrand

# ==========================
# Simulation
# ==========================
def run_simulation():

    temperatures = [
        1e7,      # Small star
        1.55e7,   # Sun
        2e7,      # Hotter star
        5e7       # Massive star
    ]

    E_range = np.linspace(1e-18, 5e-15, 2000)
    E_keV = E_range / (e * 1000)

    plt.figure(figsize=(12,7))

    for T in temperatures:
        rate, integrand = fusion_rate(E_range, T)

        integrand /= np.max(integrand)

        peak_index = np.argmax(integrand)
        peak_energy = E_keV[peak_index]

        print("Temperature:", "{:.2e}".format(T), "K")
        print("Gamow Peak Energy:", "{:.2f}".format(peak_energy), "keV")
        print("Fusion Rate (relative):", "{:.3e}".format(rate))
        print("---------------")

        plt.plot(E_keV, integrand, label=f"T = {T:.2e} K")

    plt.title("Gamow Peak Shift with Stellar Temperature")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Normalized Fusion Probability")
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================
# Run Simulation
# ==========================
run_simulation()
