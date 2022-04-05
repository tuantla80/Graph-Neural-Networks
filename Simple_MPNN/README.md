## Message Passing Neural Networks 
Pytorch Implementation of Message Passing Neural Networks (MPNN).  
Ref: 
1. Gilmer *et al.*, [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf), arXiv, 2017.  
[Tensorflow implementation by Gilmer](https://github.com/brain-research/mpnn)  
2. [CS147 - Deep Learning - Brown University](https://brown-deep-learning.github.io/dl-website-2020/projects/public/hw5-mpnns/hw5-mpnns.html)  
3. [Deep Graph Library (DGL)](https://www.dgl.ai/)  

### 1. Data structure of a molecule: molecule.py  
- nodes: a list contains the atoms in molecule which are represented by their atomic numbers (not their symbols)  
- edges: a list of tuple and each tuple is the bond between ith and jth nodes  
- label: a number (1: active against cancer, 0 if not)  
### 2. Read sdf file: sdf_iterator.py
- An .sdf file contains many molecules with the marker = '$$$$' to separate each molecule.  
- Read each molecule in sdf file to get the data structure of molecule (nodes, edges and label)  
### 3. Get training and test data: preprocessing.py  
- Read sdf file and store output as a list of molecule (as indicated in 2.)
- Transform nodes of a molecule to 2D numpy array (num_atoms, 119) in which each atomic atom is transformed to 1 hot vector with length=119  
  119 is number of elements of periodic table. 
- Train/test split of a list of molecules. Ex 0.9/0.1   

### 4. MPPN Model for node features only thus it is a simple model: mpnn.py  
- Implemented MPNN mode using DGL
