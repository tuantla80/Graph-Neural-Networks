# Graph-Neural-Networks
Pytorch Implementation of Graph Neural Networks  
## I. Dataset  
- HIV dataset at data/raw/HIV.csv. Total ~ 40K molecules.  
  ```
    | smiles                                               | activity | HIV_active  
    | CCC1=[O+][Cu-3]2([O+]=C(CC)C1)[O+]=C(CC)CC(CC)=[O+]2 |    CI    |    0
    | CC(=O)N1c2ccccc2Sc2c1ccc1ccccc21                     |    CI    |    0  
    | O=C(O)Cc1ccc(SSc2ccc(CC(=O)O)cc2)cc1                 |    CM    |    1  
  ```  
  - The first column is SMILES string of a molecule. 
  - The second column is activity
  - The third column is HV_active, which is a binary class. If HIV_active = 1, the molecule is able to inhibit the HIV (human immunodeficiency virus).  
- <b>Train/test split at sample.py</b>
  - Since class HIV_active is unbalance, in training datasets we use oversampling method to make it to be balanced.
## II. Preprocess  
- File preprocess.py. To transform molecules to graph data.  
- For each molecule:  
  - Calculate node features of each atom. Such as the below properties.  
    ```
     Atomic number
     Atom degree
     Formal charge
     Hybridization
     Aromaticity
     Total Num Hs
     Radical Electrons
     In Ring
     Chirality  
    ```  
  - Calculate the edged features between two connected atoms. Such as the below properties.  
    (Note: Need specify for both directions. Eg. Node i -> Node j and Node j -> Node i)
    ```  
    Bond type
    Rings
    ```    
  - Calculate the adjacency list of connections.   
  (Note: Need specify for both directions. Eg. Node i -> Node j and Node j -> Node i)  
  
  ## III. Model  
  - File model.py  
  - Using 3 GAT (Graph Attention Convolution) layers and 3 TopKPooling layers.  
    - They are hyperparameters and depending on the size of molecule graphs  
