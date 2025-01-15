import numpy as np
import matplotlib.pyplot as plt
import burnman
from burnman import equilibrate
from burnman.minerals import SLB_2011

# Initialize the phases
feomgo = SLB_2011.ferropericlase()

# Define temperature and composition
T = 2000.0  # Temperature in Kelvin
composition = {'Fe': 1.0, 'Mg': 1.0, 'O': 2.0}  # Bulk composition

feomgo.set_composition([0.50, 0.50])

# Define the assemblage with initial phase proportions
assemblage = burnman.Composite([feomgo], [1.0])

# Define equilibrium constraints
equality_constraints = [
    ('T', T),  # Fix temperature
    ('phase_fraction', (feomgo, 0.0))  # Ensure ferropericlase starts at 0%
]
free_compositional_vectors = [{'Mg': 1.0, 'Fe': -1.0}]  # Fe ↔ Mg exchange

# Perform the equilibrium calculation for the univariant point
sol, prm = equilibrate(
    composition, assemblage, equality_constraints, free_compositional_vectors, verbose=False
)

if not sol.success:
    raise Exception("Could not find a solution with the provided starting guesses.")

P_univariant = sol.assemblage.pressure  # Univariant pressure
phase_names = [sol.assemblage.phases[i].name for i in range(2)]
x_fe_mbr = [sol.assemblage.phases[i].molar_fractions[1] for i in range(2)]

print(f"Univariant pressure at {T:.0f} K: {P_univariant / 1.e9:.3f} GPa")
print("FeO concentrations at the univariant:")
for i in range(2):
    print(f"{phase_names[i]}: {x_fe_mbr[i]:.2f}")

# Generate the phase diagram by calculating over a range of compositions
output = []
for (m1, m2, x_fe_m1) in [[feomgo, np.linspace(x_fe_mbr[0], 0.999, 20)]]:
    assemblage = burnman.Composite([m1, m2], [1.0, 0.0])

    m1.set_composition([1.0 - x_fe_m1[0], x_fe_m1[0]])
    m2.set_composition([1.0 - x_fe_m1[0], x_fe_m1[0]])

    assemblage.set_state(P_univariant, T)

    equality_constraints = [
        ('T', T),
        ('phase_composition', (m1, [['Mg', 'Fe'], [0.0, 1.0], [1.0, 1.0], x_fe_m1])),
        ('phase_fraction', (m2, 0.0))
    ]

    sols, prm = equilibrate(
        composition, assemblage, equality_constraints, free_compositional_vectors, verbose=False
    )

    out = np.array([
        [sol.assemblage.pressure, sol.assemblage.phases[0].molar_fractions[1], sol.assemblage.phases[1].molar_fractions[1]]
        for sol in sols if sol.success
    ])
    output.append(out)

output = np.array(output)

# Plot the phase diagram
fig, ax = plt.subplots()

color = "blue"
for i in range(len(output)):
    ax.plot(output[i, :, 1], output[i, :, 0] / 1.e9, color=color)
    ax.plot(output[i, :, 2], output[i, :, 0] / 1.e9, color=color)
    ax.fill_betweenx(output[i, :, 0] / 1.e9, output[i, :, 1], output[i, :, 2], color=color, alpha=0.2)

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 20.0)
ax.set_xlabel("Molar fraction of FeO")
ax.set_ylabel("Pressure (GPa)")
plt.title("MgO-FeO Phase Diagram")
plt.show()
