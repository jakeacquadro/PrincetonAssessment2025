import torch
from ase.build import bulk
from ase.units import GPa
from mattersim.forcefield import MatterSimCalculator
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS

# Select device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running MatterSim on {device}")
"""
Cu (fcc)
"""
# Build Cu crystal structure (fcc)
cu_fcc = bulk("Cu", "fcc", a=3.58)  # Lattice constant

# Attach MatterSim calculator
cu_fcc.calc = MatterSimCalculator(device=device)

# Apply pressure indirectly by relaxing the unit cell
target_pressure = 300 * GPa  # Convert GPa to ASE units (eV/Å³)

# Use ASE's UnitCellFilter to allow cell relaxation under pressure
ucf = UnitCellFilter(cu_fcc, scalar_pressure=target_pressure)

# Optimize structure under the given pressure
optimizer = BFGS(ucf)
optimizer.run(fmax=0.01)  # Relax until forces are below 0.01 eV/Å

# Compute properties after relaxation
energy = cu_fcc.get_potential_energy()
forces = cu_fcc.get_forces()
stress = cu_fcc.get_stress(voigt=False)

# Output results
print(f"Energy (eV)                 = {energy}")
print(f"Energy per atom (eV/atom)   = {energy / len(cu_fcc)}")
print(f"Forces on first atom (eV/A) = {forces[0]}")
print(f"Stress[0][0] (eV/A^3)       = {stress[0][0]}")
print(f"Stress[0][0] (GPa)          = {stress[0][0] / GPa}")

"""
Cu (bcc)
"""
# Build Cu crystal structure (bcc)
cu_bcc = bulk("Cu", "bcc", a=2.84)  # Lattice constant

# Attach MatterSim calculator
cu_bcc.calc = MatterSimCalculator(device=device)

# Apply pressure indirectly by relaxing the unit cell
target_pressure = 300 * GPa  # Convert GPa to ASE units (eV/Å³)

# Use ASE's UnitCellFilter to allow cell relaxation under pressure
ucf = UnitCellFilter(cu_bcc, scalar_pressure=target_pressure)

# Optimize structure under the given pressure
optimizer = BFGS(ucf)
optimizer.run(fmax=0.01)  # Relax until forces are below 0.01 eV/Å

# Compute properties after relaxation
energy = cu_bcc.get_potential_energy()
forces = cu_bcc.get_forces()
stress = cu_bcc.get_stress(voigt=False)

# Output results
print(f"Energy (eV)                 = {energy}")
print(f"Energy per atom (eV/atom)   = {energy / len(cu_bcc)}")
print(f"Forces on first atom (eV/A) = {forces[0]}")
print(f"Stress[0][0] (eV/A^3)       = {stress[0][0]}")
print(f"Stress[0][0] (GPa)          = {stress[0][0] / GPa}")
