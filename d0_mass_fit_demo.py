import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Generate synthetic D0 mass data
rng = np.random.default_rng(42)

signal = rng.normal(1.865, 0.012, 12000)
background = rng.uniform(1.70, 2.05, 18000)
mass = np.concatenate([signal, background])

# Create histogram
counts, edges = np.histogram(mass, bins=100, range=(1.70, 2.05))
centers = (edges[:-1] + edges[1:]) / 2

# Gaussian signal + linear background
def fit_function(x, A, mu, sigma, c, slope):
    return A*np.exp(-0.5*((x-mu)/sigma)**2) + c + slope*(x-mu)

# Fit using SciPy
initial = [counts.max(), 1.865, 0.015, 200, 0]

popt, pcov = curve_fit(
    fit_function, centers, counts,
    p0=initial,
    bounds=([0, 1.80, 0.003, 0, -np.inf],
            [np.inf, 1.93, 0.06, np.inf, np.inf]),
    maxfev=20000
)

errors = np.sqrt(np.diag(pcov))
A, mu, sigma, c, slope = popt

# Plot
x = np.linspace(1.70, 2.05, 500)

plt.figure(figsize=(9, 6))
plt.errorbar(centers, counts, yerr=np.sqrt(counts),
             fmt='o', markersize=3, label='Toy data')

plt.plot(x, fit_function(x, *popt),
         linewidth=2, label='Gaussian + background fit')

plt.xlabel(r'Invariant mass (GeV/$c^2$)')
plt.ylabel('Counts')
plt.title(r'Toy $D^0$ invariant-mass fit')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('d0_mass_fit_demo.png', dpi=300)
plt.show()

print("\nD0 MASS FIT RESULTS")
print(f"Mass = {mu:.5f} +/- {errors[1]:.5f} GeV/c^2")
print(f"Width = {sigma:.5f} +/- {errors[2]:.5f} GeV/c^2")
print("Plot saved: d0_mass_fit_demo.png")
