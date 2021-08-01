# Graph-Neural-Networks
Pytorch Implementation of Graph Neural Networks  
## I. Dataset  
- HIV dataset at data/HIV.csv  
  ```
    | smiles                                               | activity | HIV_active  
    | CCC1=[O+][Cu-3]2([O+]=C(CC)C1)[O+]=C(CC)CC(CC)=[O+]2 |    CI    |    0
    | CC(=O)N1c2ccccc2Sc2c1ccc1ccccc21                     |    CI    |    0  
    | O=C(O)Cc1ccc(SSc2ccc(CC(=O)O)cc2)cc1                 |    CM    |    1  
  ```  
  - The first column is SMILES string of a molecule. 
  - The second column is activity
  - The third column is HV_active, which is a binary class. If HIV_active = 1, the molecule is able to inhibit the HIV (human immunodeficiency virus).
  
