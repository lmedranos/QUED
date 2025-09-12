import os

from argparse import ArgumentParser

import numpy as np

from ase.io import read
from ase.atoms import Atoms
from ase.data import atomic_numbers
from ase.calculators.dftb import Dftb
from ase.units import Hartree, Bohr

# get xyz and Z arrays from xyz file
def get_atomic_numbers_n_coordinates(inputFile, index=0):
    ofile = open('tmp.xyz', 'w')
    with open(inputFile, 'r') as ifile:
        lines = ifile.readlines()
        line0 = lines[0]

        # Count total conformers
        total_conformers = sum(1 for line in lines if line.startswith(line0))
        if index >= total_conformers:
            return np.zeros(10), np.zeros(10)

        n = -1
        for line in lines:
            # count one conformer
            if line.startswith(line0): n += 1

            # find <index>th conformer
            if n==index:    ofile.write(line)
            elif n>index:   break
            else:           continue
    ofile.close()
    
    # get atomic number and positions of conformer
    atoms = read('tmp.xyz')
    Z = atoms.get_atomic_numbers()
    xyz = atoms.get_positions()
    os.remove('tmp.xyz')
    return Z, xyz

# extracts frontier orbitals (HOMOs/LUMOs) from band.out 
# and returns them as a sorted NumPy array
def get_band_energies(iorbs=4):
    # Reads the band.out file produced by DFTB+
    # This file contains eigenvalues of the molecular orbitals
    ifile = open('band.out', 'r')
    lines = ifile.readlines()
    ifile.close()

    # c6p = list of [energy, occupation] pairs
    c6p = []
    active = False
    for line in lines:
        if not active:
            # Looks for the first k-point, spin channel section in band.out
            if "KPT" in line:
                # Once inside that section (active=True), it parses the eigenvalues
                active = True
                nl = 0
        else:
            nl += 1
            tokens = line.split()
            # Each valid line has 3 fields: index, eigenvalue, occupation
            if len(tokens) == 3:
                c6p.append([float(tokens[1]), float(tokens[2])])
            else:
                # Stops when the section ends
                if nl>3:
                    active = False
    c6p = np.array(c6p)
    
    # Identifies the HOMO–LUMO gap
    # HOMO = last occupied orbital (occupation ≠ 0)
    # LUMO = first unoccupied orbital (occupation = 0)
    HOMOs, LUMOs = [], []
    if len(c6p) >= 8:
        for i in range(len(c6p)):
            if c6p[i][1] == 0.0:
                # Collects <iorbs> HOMOs and <iorbs> LUMOs around the gap
                for j in range(iorbs):
                    HOMOs.append(c6p[i - j - 1][0])
                    LUMOs.append(c6p[i + j][0])
                break
        # 8 is a flag saying “I found a full set”
        return np.concatenate(([8], np.sort(HOMOs+LUMOs)), axis=None)
    else:
        # If too few orbitals (<8), it just returns whatever was found, preceded by the count
        return np.concatenate(([len(c6p)], np.sort(c6p[:,0])), axis=None)

# extracts the Mulliken charges from detailed.out
def get_atom_charges():
    # Reads the detailed.out file produced by DFTB+
    ifile = open('detailed.out', 'r')
    lines = ifile.readlines()
    ifile.close()

    active = False
    c6p = []
    for line in lines:
        if not active:
            # Scans through detailed.out until it finds the block starting with "Atom Charge"
            if "Atom           Charge" in line:
                active = True
                nl = 0
        else:
            nl += 1
            c6pr = line.split()
            if len(c6pr) == 2:
                # Collects the second column = atomic charge
                c6p.append(float(c6pr[1]))
            else:
                # Stops when it encounters a line that doesn’t have exactly 2 entrie
                if nl > 3:
                    active = False
    # returns a Python list of per-atom charges
    return c6p

# calculate QM properties
# to get a descriptor-like array containing DFTB properties for the molecule:
# scalar properties (energies, electron count)
# 3D dipole moment
# orbital eigenvalues
# atomic charges
def calculate_dftb_props(Z_mol, xyz_mol):
    # Build the molecule
    amol = Atoms(Z_mol, xyz_mol)

    # Set up the DFTB calculator
    # SCC (self-consistent charge) enabled, up to 2000 iterations
    # 3rd-order correction enabled
    # Electronic filling with finite temperature smearing (250 K)
    # Broyden mixer with parameter 0.4 (to stabilize SCC convergence)
    # Hubbard correction damping with exponent 4.05
    # MBD (many-body dispersion) correction with specific parameters
    # Forces are requested
    calc = Dftb(label='current_dftb',
                atoms=amol,
                # run_manyDftb_steps=True,
                Hamiltonian_SCC = 'Yes',
                Hamiltonian_ThirdOrderFull = 'Yes',
                Hamiltonian_HCorrection_ = 'Damping',
                Hamiltonian_HCorrection_Exponent = 4.05,
                Hamiltonian_Dispersion_ = 'MBD',
                Hamiltonian_Dispersion_Beta = 0.83,
                Hamiltonian_Dispersion_KGrid = '1 1 1',
                Hamiltonian_Dispersion_NOmegaGrid = 25,
                Hamiltonian_Dispersion_ReferenceSet = 'ts',
                Options_='',
                Options_UseOmpThreads = 'Yes',
                ParserOptions_='',
                ParserOptions_ParserVersion=13)
    
    # Attach the calculator to the molecule
    amol.calc = calc

    try:
        # Run calculation
        calc.calculate(amol)

        # Extract properties
        # band energies
        eig = get_band_energies()
        # Mulliken charges
        charges = get_atom_charges()

        # Read DFTB+ output file
        ifile = open('detailed.out', 'r')
        lines = ifile.readlines()
        ifile.close()
        # Parse line by line for quantities of interest
        for line in lines:
            if line.strip().startswith('Fermi level:'):
                FermiEne = float(line.split()[2]) * Hartree
            if line.strip().startswith('Band energy:'):
                BandEne = float(line.split()[2]) * Hartree
            if line.strip().startswith('Input / Output electrons (q):'):
                NumElec = float(line.split()[5])
            if line.strip().startswith('Energy H0:'):
                h0Ene = float(line.split()[2]) * Hartree
            if line.strip().startswith('Energy SCC:'):
                sccEne = float(line.split()[2]) * Hartree
            if line.strip().startswith('Energy 3rd:'):
                thirdEne = float(line.split()[2]) * Hartree
            if line.strip().startswith('Repulsive energy:'):
                repEne = float(line.split()[2]) * Hartree
            if line.strip().startswith('Dispersion energy:'):
                mbdEne = float(line.split()[2]) * Hartree
            if "Dipole moment:" in line:
                lsp = line.split()
                if lsp[-1] == 'au':
                    # converted from a.u. to Debye using 1 au = 2.541746 ~ 1/0.39343
                    DipMom = np.array([ float(lsp[-4]), float(lsp[-3]), float(lsp[-2])])/0.393430238326893
                break

        # Collect into a single vector
        qm_props = np.concatenate(([FermiEne, BandEne, NumElec, h0Ene, sccEne, thirdEne, repEne, mbdEne], DipMom, eig, charges), axis=None)
    # If anything fails, it returns an array of zeros
    except  Exception as e:
        qm_props = np.zeros(10)
        print(f"DFTB calculation failed: {e}")
    return qm_props

# pad charges for QM descriptor
def pad_charges(chg, max_atoms=203):
    if len(chg)!=max_atoms:
        return np.concatenate((chg, np.zeros(max_atoms-len(chg))), axis=None)
    else:
        return chg

# get QM descriptor
def get_props(Z, xyz, max_atoms):
    # calculate QM properties
    DFTBprops = calculate_dftb_props(Z, xyz)
    # extract global properties: energy terms and number of electrons
    TBglobal = DFTBprops[:8]
    # extract norm of dipole moment
    TBdip = np.array([np.linalg.norm(DFTBprops[8:11])])
    # get electronic molecular orbital energies
    nEig = int(DFTBprops[11])
    TBeig = DFTBprops[12:12+nEig]
    # get Mulliken charges and padding
    TBchg = DFTBprops[12+nEig:]
    TBchg = pad_charges(TBchg, max_atoms)
    # concatenate properties
    props = np.concatenate((TBglobal, TBdip, TBeig, TBchg))
    return props

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_conformers', type=str,
                        help='Relative or absolute path to xyz file of conformers '
                        'or the directory with the xyz files. ')
    parser.add_argument('-n', '--max_conformers', type=int, default=10,
                        help='Maximum number of conformers per molecule to consider. ' 
                        'Defult: 10. ')
    parser.add_argument('-o', '--output_path', type=str, default='./qmprops',
                        help='Path to output npz files with calculations. ')
    return parser

if __name__ == '__main__':
    arguments = get_parser().parse_args()

    conformers_path = arguments.input_conformers
    if conformers_path.endswith('xyz'):
        xyzFiles = [conformers_path]
    else:
        xyzFiles = [os.path.join(conformers_path, x) for x in os.listdir(conformers_path)]

    output_directory = arguments.output_path
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print(f'XYZ files will be saved in {output_directory}')

    for inputFile in xyzFiles:
        label = inputFile.split('/')[-1].split('.')[0]

        for i in range(arguments.max_conformers):
            Z_mol, xyz_mol = get_atomic_numbers_n_coordinates(inputFile, i)
            if Z_mol.sum()==0: 
                print(f"{inputFile}: Maximum of conformers reached: {i} conformers")
                break
            
            # atomic coordinates, here in Ångström
            # atomic numbers
            molecule = dict(Z=Z_mol, xyz=xyz_mol)

            DFTBprops = calculate_dftb_props(Z_mol, xyz_mol)
            if len(DFTBprops) < 12:
                print(f"{inputFile}: Not enough data for conformer {i}")
            else:
                # FermiEne
                molecule.update(dict(FermiEne=np.array([float(DFTBprops[0])])))
                # BandEne
                molecule.update(dict(BandEne=np.array([float(DFTBprops[1])])))
                # NumElec
                molecule.update(dict(NumElec=np.array([float(DFTBprops[2])])))
                # h0Ene
                molecule.update(dict(h0Ene=np.array([float(DFTBprops[3])])))
                # sccEne
                molecule.update(dict(sccEne=np.array([float(DFTBprops[4])])))
                # 3rdEne
                molecule.update(dict(Ene3rd=np.array([float(DFTBprops[5])])))
                # RepEne
                molecule.update(dict(repEne=np.array([float(DFTBprops[6])])))
                # mbdEne
                molecule.update(dict(mbdEne=np.array([float(DFTBprops[7])])))
                # Dipole Moment
                molecule.update(dict(TBdip=DFTBprops[8:11]))

                # orbital energies
                nEig = int(DFTBprops[11])    # Number of retrieved orbital energies
                if nEig < 8: print(f"{inputFile}: {nEig} orbital energies in conformer {i}")
                molecule.update(dict(TBeig=DFTBprops[12:12+nEig]))
                
                # Mulliken charges
                molecule.update(dict(TBchg=DFTBprops[12+nEig:]))
                print(f"{inputFile}: QM properties extracted for conformer {i}")

            outputFilename = f"{output_directory}/Geom-m{label}-c{i}.npz"
            np.savez_compressed(outputFilename, **molecule)
            print(f"Done - {outputFilename}\n")
