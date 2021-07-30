import numpy as np
import sys
from random import seed, shuffle
import sdf_iterator as sdf_iterator


def read_file(file_path):
    """
    TA provided helper function to read an .sdf file.
    Given the file, this produces a list of all the molecules
    in a file (see molecule.py for the contents of a molecule).

    :param file_path: string, file path of data
    :return: a list of molecules. The nodes are only a list of atomic numbers.
    """
    iterator = sdf_iterator.SdfIterator(file_path)
    mol = iterator.read_one_molecule()
    molecules = []
    while mol is not None:
        molecules.append(mol)
        mol = iterator.read_one_molecule()
    return molecules


def get_data(file_path, random_seed=None, test_fraction=0.1, number_of_elements = 119):
    """
    Loads the NCI dataset from an sdf file.

    After getting back a list of all the molecules in the .sdf file,
    there's a little more preprocessing to do. First, you need to one hot
    encode the nodes of the molecule to be a 2d numpy array of shape
    (num_atoms, 119) of type np.float32 (see molecule.py for more details).
    After the nodes field has been taken care of, shuffle the list of
    molecules, and return a train/test split of 0.9/0.1.

    :param file_path: string, file path of data
    :param random_seed the random seed for shuffling
    :return: train_data, test_data. Two lists of shuffled molecules that have had their
    nodes turned into a 2d numpy matrix, and of split 0.9 to 0.1.

    Example of numpy
    > np.eye(5)
      -> array([[1., 0., 0., 0., 0.],
               [0., 1., 0., 0., 0.],
               [0., 0., 1., 0., 0.],
               [0., 0., 0., 1., 0.],
               [0., 0., 0., 0., 1.]])  # It is 5x5 matrix
    > m_nodes = [3, 2, 3, 0]  # list of 4 items
    > np.eye(5)[[3, 2, 3, 0], :]
      -> array([[0., 0., 0., 1., 0.],   # for m_nodes[0]=3
               [0., 0., 1., 0., 0.],    # for m_nodes[1]=2
               [0., 0., 0., 1., 0.],    # for m_nodes[2]=3
               [1., 0., 0., 0., 0.]])   # for m_nodes[3]=0
                                        # It is 4x5 matrix
    """
    # TODO: Read in the molecules file
    molecules = read_file(file_path)

    # TODO: Convert each molecule's nodes (atoms) to a one-hot vector
    for m in molecules:
        m.nodes = np.eye(number_of_elements, dtype=np.float32)[m.nodes, :]
    # TODO: Call `seed(random_seed)` and shuffle the molecules with `shuffle`
    seed(random_seed)
    shuffle(molecules)
    # TODO: Split the data into training and testing sets with test_fraction

    train_data = molecules[: -(int(test_fraction * len(molecules)) + 1)]
    test_data = molecules[int((1 - test_fraction) * len(molecules)):]
    return train_data, test_data

if __name__ == '__main__':

    train_data, test_data = get_data(file_path=r'data\1-balance.sdf')
    print(f'length of training = {len(train_data)}')
    print(f'length of test data = {len(test_data)}')

    np.set_printoptions(threshold=np.inf)
    print(f'mol example \n '
          f'num nodes = {len(train_data[0].nodes)} \n'
          f'nodes = {train_data[0].nodes} \n '
          f'num egdes = {len(train_data[0].edges)} \n'
          f'edges = {train_data[0].edges} \n '
          f'label = {train_data[0].label}')