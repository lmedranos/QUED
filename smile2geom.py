import os
import itertools
from argparse import ArgumentParser

import pandas as pd

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_dataset_path', type=str,
                        help='Relative or absolute path to the initial dataset.'
                        'String type. Preferably a csv.' )
    parser.add_argument('-x', '--smiles_label', type=str,
                        help='Label used in the input file to refer to the column of SMILEs.' )
    parser.add_argument('-y', '--target_label', type=str, default='',
                        help="Label used in the input file to refer to the column of the target property. Default: '' (no target property)")
    parser.add_argument('-o', '--output_path', type=str, default='./geometries',
                        help='Path to output xyz files directory. ')
    return parser

# get atom types in a SMILE
def get_atom_types(smile):
    mol = Chem.MolFromSmiles(smile)
    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return atom_types
    
# generate xyz files from SMILEs
# also returns number of heavy atoms and total number of atoms
def generate_xyz_from_smiles(smiles, filename):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Invalid SMILES")
            return False, 0, 0
        n_heavy = mol.GetNumAtoms()

        # Generate 3D coordinates
        mol = Chem.AddHs(mol)
        n_total = mol.GetNumAtoms()
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)

        # Write XYZ file
        Chem.rdmolfiles.MolToXYZFile(mol, filename)
        return True, n_heavy, n_total
    except (RuntimeError, ValueError) as e:
        print(f"Error occurred for SMILES: {smiles}. Error message: {e}")
        return False, n_heavy, n_total

if __name__ == '__main__':
    arguments = get_parser().parse_args()

    # read <input_dataset_path> csv
    df = pd.read_csv(arguments.input_dataset_path)
    smiles = df[arguments.smiles_label].tolist()

    # create output directory for xyz files
    output_directory = arguments.output_path
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print(f'XYZ files will be saved in {output_directory}')

    if_xyz = []
    chemical_species = []
    number_heavy_atoms = [] 
    number_total_atoms = []
    for i, smile in enumerate(smiles):
        # get chemical species in the molecule (for reference)
        tmp = get_atom_types(smile)
        chemical_species.append(tmp)

        # output filename: <molecule_index_in_database>.xyz
        # stored in <output_directory>
        ofilename = f'{output_directory}/{i}.xyz'
        # create xyz file
        created, n_heavy, n_total = generate_xyz_from_smiles(smile, ofilename)
        if_xyz.append(int(created))
        number_heavy_atoms.append(n_heavy)
        number_total_atoms.append(n_total)
    print(f'From {len(smiles)} SMILEs -> {if_xyz.count(1)} XYZ files were generated')

    # create a csv file summarizing relevant information of the database for future filtering
    summary = {'molecule_index': list(range(len(smiles))), 'smile': smiles, 'xyz': if_xyz}
    if len(arguments.target_label)>0:
        summary.update({arguments.target_label: df[arguments.target_label].tolist()})
    summary.update({'numHeavyAtoms': number_heavy_atoms, 'totalNumAtoms': number_total_atoms})
    summary = pd.DataFrame(summary)

    # get chemical species present in the whole database
    database_species = list(set(itertools.chain.from_iterable(chemical_species)))
    for specie in database_species:
        summary[specie] = 0

    # count number of atoms per chemical specie
    for i, smile in enumerate(smiles):
        for specie in database_species:
            summary.loc[i, specie] = chemical_species[i].count(specie)
    
    # save csv file
    summary.to_csv('summary.csv', index=False)
    print('File summary.csv was created')
    print('Done.')
