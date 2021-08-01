import pandas as pd
from rdkit import Chem
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np
import os
from tqdm import tqdm


class MoleculeDataset(Dataset):
    '''To create customization datasets by inheriting the Dataset class from torch-geometric
        - Need to overridden some functions from the template from torch-geometric
    '''
    def __init__(self, root, filename, path_proccessed,
                 test=False, transform=None, pre_transform=None):
        """
        root: root folder. Eg. if the root = 'data\' then torch-geometric will
              create two sub folders: data\raw: for raw data
                                      data\processed: for data after processing
        filename: name of input file
        Note: if running in Colab:
                root = '.'
                filename is file_path = (os.path.join(path, filename))
        """
        self.filename = filename
        self.path_proccessed = path_proccessed
        self.test = test
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)


    @property
    def raw_file_names(self):
        """ If this file exists in raw directory, the download is not triggered.
            (The download func. is not implemented here)
        """
        return self.filename


    @property
    def processed_file_names(self):
        """ If these files are found in raw directory, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]


    def download(self):
        pass


    def process(self):
        '''
        To construct graph from input data
        '''
        # Get input data
        self.data = pd.read_csv(self.raw_paths[0])

        # Run for each row (each SMILES). tqdm is a progress bar
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            molecule = Chem.MolFromSmiles(row["smiles"])
            # Get node features
            node_feats = self._get_node_features(molecule)
            # Get edge features
            edge_feats = self._get_edge_features(molecule)
            # Get adjacency info
            edge_index = self._get_adjacency_info(molecule)
            # Get labels info
            label = self._get_labels(row["HIV_active"])

            # Create data object
            data = Data(x=node_feats,           # node features
                        edge_index=edge_index,  # adjacency info
                        edge_attr=edge_feats,   # edged features
                        y=label,
                        smiles=row["smiles"]  # add additional SMILES string. good for debugging
                        )
            if self.test:
                file_path = os.path.join(self.path_proccessed, f'data_test_{index}.pt')
                print(f'file_path = {file_path}')
                torch.save(data, file_path)  # in Colab
                # torch.save(data, os.path.join(self.processed_dir, f'data_test_{index}.pt'))
            else:
                torch.save(data, os.path.join(self.path_proccessed, f'data_{index}.pt'))  # in Colab
                # torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))

    def _get_node_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [Number of Nodes (atoms), Node Feature size]
        """
        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number
            node_feats.append(atom.GetAtomicNum())
            # Feature 2: Atom degree
            node_feats.append(atom.GetDegree())
            # Feature 3: Formal charge
            node_feats.append(atom.GetFormalCharge())
            # Feature 4: Hybridization
            node_feats.append(atom.GetHybridization())
            # Feature 5: Aromaticity
            node_feats.append(atom.GetIsAromatic())
            # Feature 6: Total Num Hs
            node_feats.append(atom.GetTotalNumHs())
            # Feature 7: Radical Electrons
            node_feats.append(atom.GetNumRadicalElectrons())
            # Feature 8: In Ring
            node_feats.append(atom.IsInRing())
            # Feature 9: Chirality
            node_feats.append(atom.GetChiralTag())

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)


    def _get_edge_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())
            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)


    def _get_adjacency_info(self, mol):
        """
        We could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features.
        Note: Need to transform edge_indices
        Eg. edge_indices = [[1, 2], [3, 4], [5, 6]] # Node 1-> Node 2, 3->4, 5->6
            Transform to edge_indices = [[1, 3, 5],
                                         [2, 4 ,6]]
        """
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]  # 2 directions i -> j and i <- j

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices


    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)


    def len(self):
        '''Returns the number of examples in your dataset.'''
        return self.data.shape[0]


    def get(self, idx):
        """ Implements the logic to load a single graph.
            - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.path_proccessed, f'data_test_{idx}.pt'))  # in Colab
            # data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.path_proccessed, f'data_{idx}.pt'))  # in Colab
            # data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data


def test_MoleculeDataset_class(path_raw, path_proccessed):
    '''
    Eg.
    print('Access to one molelcule')
    mol1 = dataset[1]
    print(mol1) # -> Data(edge_attr=[88, 2], edge_index=[2, 88], smiles="C(=Cc1ccccc1)C1=[O+][Cu-3]2([O+]=C(C=Cc3ccccc3)CC(c3ccccc3)=[O+]2)[O+]=C(c2ccccc2)C1", x=[39, 9], y=[1])
                # Here only a shape for Node features,  Edge features, Ajacency list and label

    # Node features
    print(mol1.x) # -> tensor([[ 6.,  2.,  0.,  3.,  0.,  1.,  0.,  0.,  0.],  # features of node 0: vector of length=9
                               [ 6.,  2.,  0.,  3.,  0.,  1.,  0.,  0.,  0.],  # features of node 1: vector of length=9
                               ...
                               [ 6.,  2.,  0.,  4.,  0.,  2.,  0.,  1.,  0.]]) # features of node 39: vector of length=9

    # Edge features
    print(mol1.edge_attr) # -> tensor([[2.0000, 0.0000],
                                        [2.0000, 0.0000],
                                        [1.0000, 0.0000],
                                        ...
                                        [1.5000, 1.0000]])

    # Ajacency matrix (in fact here is Ajacency list)
    print(mol1.edge_index) # -> tensor([[ 0,  1,  1,..., 15, 28, 23],
                                        [ 1,  0,  2,..., 20, 23, 28]])
                             node 0 ->1, node 1->0, node 1->2, ..., node 28->23, node 23->28

    # Label of this molecule
    print(mol1.y) # -> tensor([0]): inactive to HIV

    '''
    dataset = MoleculeDataset(root=".",
                              filename=os.path.join(path_raw, 'HIV_test_2_rows.csv'),
                              path_proccessed=path_proccessed,
                              test=True)  # test=False meaning running for training data
    print(f'dataset = {dataset}')
    print(f'type(dataset) = {type(dataset)}')
    print(f'filename = {dataset.raw_file_names}')
    # print('Access to one molelcule')
    # mol1 = dataset[1]
    # print(mol1)
    # print(mol1.x)
    # print(mol1.edge_attr)
    # print(mol1.edge_index)
    # print(mol1.y)

    # If calling dataset.process(), it will save to a folder
    # 1 molecule -> 1 file (*.pt)
    # dataset.process()
    print('End')


if __name__ == '__main__':
    test_MoleculeDataset_class()

    # # Train data - Need to run only one time
    # dataset = MoleculeDataset(root='.', filename=os.path.join(path, 'HIV_train_oversampled.csv'))
    # print(dataset)
    # filename = dataset.raw_file_names
    # print(f'filename = {filename}')
    # dataset.process()  # to save as *.pt file for every molecule

    # # Test data - Need to run only one time
    # dataset = MoleculeDataset(root='.',
    #                           filename=os.path.join(path_raw, 'HIV_test.csv'),
    #                           test=True)
    # print(f'dataset = {dataset}')
    # print(f'type(dataset) = {type(dataset)}')
    # filename = dataset.raw_file_names
    # print(f'filename = {filename}')
    # dataset.process()  # to save as *.pt file for every molecule