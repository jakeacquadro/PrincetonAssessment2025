import numpy as np
import matplotlib.pyplot as plt
import burnman
from burnman import equilibrate
from burnman.minerals import SLB_2011

# Initialize the minerals we will use in this example.
mgo = SLB_2011.periclase()
feo = SLB_2011.wuestite()

P = 200.
composition = {'Fe': 0.5, 'Mg': 0.5,  'O': 1.0}
assemblage = burnman.Composite([mgo, feo], [0.5, 0.5])
equality_constraints = [('P', P),
                        ('phase_fraction', (mgo, 0.0)),
                        ('phase_fraction', (feo, 0.0))]
free_compositional_vectors = [{'Mg': 1., 'Fe': -1.}]

sol, prm = equilibrate(composition, assemblage, equality_constraints,
                        free_compositional_vectors,
                        verbose=False)
if not sol.success:
    raise Exception('Could not find solution for the univariant using '
                    'provided starting guesses.')

T_univariant = sol.assemblage.temperature
phase_names = [sol.assemblage.phases[i].name for i in range(3)]
x_fe_mbr = [sol.assemblage.phases[i].molar_fractions[1] for i in range(3)]

print(f'Univariant pressure at {P/1.e9:.3f} GPa:  {T_univariant:.0f} K')
print('Fe2SiO4 concentrations at the univariant:')
for i in range(3):
    print(f'{phase_names[i]}: {x_fe_mbr[i]:.2f}')

output = []
for (m1, m2, x_fe_m1) in [[mgo, feo, np.linspace(x_fe_mbr[0], 0.001, 20)],
                          [feo, mgo, np.linspace(x_fe_mbr[0], 0.999, 20)]]:

    assemblage = burnman.Composite([m1, m2], [1., 0.])

    # Reset the compositions of the two phases to have compositions
    # close to those at the univariant point
    m1.set_composition([1.-x_fe_mbr[1], x_fe_mbr[1]])
    m2.set_composition([1.-x_fe_mbr[1], x_fe_mbr[1]])

    # Also set the pressure and temperature
    assemblage.set_state(T_univariant, P)

    # Here our equality constraints are temperature,
    # the phase fraction of the second phase,
    # and we loop over the composition of the first phase.
    equality_constraints = [('P', P),
                            ('phase_composition',
                             (m1, [['Mg_A', 'Fe_A'],
                                   [0., 1.], [1., 1.], x_fe_m1])),
                            ('phase_fraction', (m2, 0.0))]

    sols, prm = equilibrate(composition, assemblage,
                            equality_constraints,
                            free_compositional_vectors,
                            verbose=False)

    # Process the solutions
    out = np.array([[sol.assemblage.pressure,
                     sol.assemblage.phases[0].molar_fractions[1],
                     sol.assemblage.phases[1].molar_fractions[1]]
                    for sol in sols if sol.success])
    output.append(out)

output = np.array(output)

fig = plt.figure()
ax = [fig.add_subplot(1, 1, 1)]

color='purple'
# Plot the line connecting the three phases
ax[0].plot([x_fe_mbr[0], x_fe_mbr[2]],
            [T_univariant/1.e9, T_univariant/1.e9], color=color)

for i in range(3):
    if i == 0:
        ax[0].plot(output[i,:,1], output[i,:,0]/1.e9, color=color, label=f'{P} K')
    else:
        ax[0].plot(output[i,:,1], output[i,:,0]/1.e9, color=color)

    ax[0].plot(output[i,:,2], output[i,:,0]/1.e9, color=color)
    ax[0].fill_betweenx(output[i,:,0]/1.e9, output[i,:,1], output[i,:,2],
                        color=color, alpha=0.2)

# ax[0].text(0.1, 6., 'olivine', horizontalalignment='left')
# ax[0].text(0.015, 14.2, 'wadsleyite', horizontalalignment='left',
#         bbox=dict(facecolor='white',
#                     edgecolor='white',
#                     boxstyle='round,pad=0.2'))
# ax[0].text(0.9, 15., 'ringwoodite', horizontalalignment='right')

ax[0].set_xlim(0., 1.)
ax[0].set_ylim(0.,20.)
ax[0].set_xlabel('p(MgO)')
ax[0].set_ylabel('Temperature (K)')
ax[0].legend()
plt.show()