import os
import shutil
from argparse import ArgumentParser

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_conformers_directory', type=str,
                        help='Relative or absolute path to the directory with '
                        'the crest_conformers.xyz files.' )
    parser.add_argument('-o', '--output_path', type=str, default='./conformers',
                        help='Path to output xyz files of conformers. ')
    return parser

# Maximum number of lowest-xTB-energy conformers to be selected
max_conformers = 10

if __name__ == '__main__':
    arguments = get_parser().parse_args()

    input_directory = arguments.input_conformers_directory
    output_directory = arguments.output_path
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print(f'XYZ files will be saved in {output_directory}')

    for filename in os.listdir(input_directory):
        inputFile = os.path.join(input_directory, filename)
        outputFile = os.path.join(output_directory, filename)

        ofile = open(outputFile, 'w')
        with open(inputFile, 'r') as ifile:
            lines = ifile.readlines()
            line0 = lines[0]

            # Count total conformers
            total_conformers = sum(1 for line in lines if line.startswith(line0))

            # keep only <max_conformers> of
            # lowest-xTB-energy conformers
            if total_conformers<=max_conformers:
                shutil.copy(inputFile, outputFile)
            else:
                n = 0 
                for line in lines:
                    if line.startswith(line0):
                        n += 1
                        if n>max_conformers:
                            break
                    ofile.write(line)
        ofile.close()
        print(f"File {filename}: wrote {min(max_conformers, total_conformers)} conformers from {total_conformers} total")