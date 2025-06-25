import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

# read original list of smiles from the TDCommons-LD50 database
df = pd.read_csv('./ld50-tdcommons.csv')

# all atom types found in the original database
all_atom_types = ['N', 'C', 'O', 'Br', 'S', 'P', 'Si', 'Cl', 'F', 'I']

# count number of atoms of a specific chemical species in a molecule 
def count_type(molecule, symbol):
    mol = Chem.MolFromSmiles(molecule)
    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if symbol in atom_types:
        return atom_types.count(symbol)
    else:
        return 0
    
for symbol in all_atom_types:
    df[symbol] = df['Drug'].apply(lambda x: count_type(x,symbol))

# filter atoms with 'Si'
df = df[df.Si==0]

# generate xyz files
def generate_xyz_from_smiles(smiles, filename):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Invalid SMILES")
            return False

        # Generate 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)

        # Write XYZ file
        Chem.rdmolfiles.MolToXYZFile(mol, filename)
        return True
    except (RuntimeError, ValueError) as e:
        print(f"Error occurred for SMILES: {smiles}. Error message: {e}")
        return False

for id, smile in enumerate(df['Drug']):
    filename = f'./{id}.xyz'

    _ = generate_xyz_from_smiles(smile, filename)
    if not _:
        print(f"ERROR. ID: {id}, SMILES: {smile}")