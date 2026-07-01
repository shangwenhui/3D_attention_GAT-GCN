import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset

# -------------------- Atom Feature Functions --------------------
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

# -------------------- Molecular Graph Generation with Multi-Conformer Support --------------------
def smile_to_graph_with_3d(smile, numConfs=50):
    """
    Generate multiple conformers, select the one with the lowest energy,
    and extract features and edge distances.
    Args:
        smile: SMILES string
        numConfs: number of conformers to generate (default 50)
    Returns:
        c_size, features, edge_index, coords, edge_s
        Returns None if failed.
    """
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print(f"Failed to parse SMILES: {smile}")
        return None

    mol = Chem.AddHs(mol)
    if mol is None:
        print(f"Failed to add hydrogens to SMILES: {smile}")
        return None

    # 1. Generate multiple conformers using ETKDGv3 method
    #    EmbedMultipleConfs will try multiple embeddings until numConfs valid conformers are generated
    try:
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=numConfs,
                                          params=AllChem.ETKDGv3())
    except Exception as e:
        print(f"Embedding failed for {smile}: {e}")
        return None

    if not cids:
        print(f"No conformers generated for {smile}")
        return None

    # 2. Optimize each conformer with MMFF force field and compute energy
    energies = []
    for cid in cids:
        # Optimize conformer
        try:
            mmff_status = MMFFOptimizeMolecule(mol, confId=cid)
        except Exception:
            mmff_status = -1  # mark as failed
        if mmff_status == 0:  # 0 indicates success
            # Get MMFF force field properties
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            if mp is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
                if ff is not None:
                    energy = ff.CalcEnergy()
                    energies.append((cid, energy))
                else:
                    energies.append((cid, float('inf')))
            else:
                energies.append((cid, float('inf')))
        else:
            energies.append((cid, float('inf')))

    if not energies:
        print(f"All conformers optimization failed for {smile}")
        return None

    # 3. Select the conformer with the lowest energy
    best_cid = min(energies, key=lambda x: x[1])[0]

    # 4. Extract features based on the best conformer
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        features.append(atom_features(atom))

    # Edge list (undirected edges)
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    # Convert to directed graph (for PyG message passing)
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    # Extract 3D coordinates (best conformer)
    coords = []
    conf = mol.GetConformer(best_cid)
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])

    # 5. Compute edge distance features (edge_s)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    coords_t = torch.tensor(coords, dtype=torch.float)
    row, col = edge_index_t[0], edge_index_t[1]
    dist = torch.norm(coords_t[row] - coords_t[col], dim=1, keepdim=True)   # [E, 1]

    # Normalize to [0,1]
    if dist.max() > 0:
        edge_s = dist / dist.max()
    else:
        edge_s = dist

    return c_size, features, edge_index, coords, edge_s


# -------------------- PyTorch Geometric Dataset Class --------------------
class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='data',
                 xd=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None):
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, y, smile_graph):
        assert (len(xd) == len(y)), "The two lists must be the same length!"
        data_list = []
        skipped = 0
        for i in range(len(xd)):
            smiles = xd[i]
            labels = y[i]
            result = smile_graph.get(smiles)   # reuse precomputed dictionary
            if result is None:
                # if not in dict, compute on the fly
                result = smile_to_graph_with_3d(smiles)
                if result is None:
                    skipped += 1
                    continue
            c_size, features, edge_index, coords, edge_s = result

            GCNData = Data(x=torch.Tensor(features),
                           edge_index=torch.LongTensor(edge_index).t().contiguous(),
                           y=torch.FloatTensor([labels]),
                           pos=torch.Tensor(coords),
                           edge_s=edge_s)   # 3D distance features
            data_list.append(GCNData)

        if len(data_list) == 0:
            print("No valid molecules to process.")
            return

        print(f'Graph construction done. Skipped {skipped} molecules. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# -------------------- Main Program --------------------
if __name__ == "__main__":
    # Adjustable number of conformers
    NUM_CONFS = 50   # you can modify this value as needed

    # File paths
    train_csv = 'data/train.csv'
    val_csv   = 'data/val.csv'
    test_csv  = 'data/test.csv'

    # Read CSVs
    df_train = pd.read_csv(train_csv)
    df_val   = pd.read_csv(val_csv)
    df_test  = pd.read_csv(test_csv)

    # Drop NaN
    df_train = df_train[df_train['compound_iso_smiles'].notna()]
    df_val   = df_val[df_val['compound_iso_smiles'].notna()]
    df_test  = df_test[df_test['compound_iso_smiles'].notna()]

    train_drugs = list(df_train['compound_iso_smiles'])
    val_drugs   = list(df_val['compound_iso_smiles'])
    test_drugs  = list(df_test['compound_iso_smiles'])

    # Preprocess all unique SMILES (generate multiple conformers and select best)
    all_smiles = set(train_drugs + val_drugs + test_drugs)
    smile_graph = {}
    print("Preprocessing molecules with multiple conformers...")
    for idx, smile in enumerate(all_smiles):
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(all_smiles)} molecules")
        result = smile_to_graph_with_3d(smile, numConfs=NUM_CONFS)
        if result is not None:
            smile_graph[smile] = result

    # Generate training set .pt
    processed_train = 'data/processed/train.pt'
    if not os.path.isfile(processed_train):
        train_Y = list(df_train['affinity'])
        train_data = TestbedDataset(root='data', dataset='train',
                                    xd=train_drugs, y=train_Y, smile_graph=smile_graph)
        print(processed_train, 'created')
    else:
        print(processed_train, 'already exists')

    # Generate validation set .pt
    processed_val = 'data/processed/val.pt'
    if not os.path.isfile(processed_val):
        val_Y = list(df_val['affinity'])
        val_data = TestbedDataset(root='data', dataset='val',
                                  xd=val_drugs, y=val_Y, smile_graph=smile_graph)
        print(processed_val, 'created')
    else:
        print(processed_val, 'already exists')

    # Generate test set .pt
    processed_test = 'data/processed/test.pt'
    if not os.path.isfile(processed_test):
        test_Y = list(df_test['affinity'])
        test_data = TestbedDataset(root='data', dataset='test',
                                   xd=test_drugs, y=test_Y, smile_graph=smile_graph)
        print(processed_test, 'created')
    else:
        print(processed_test, 'already exists')

    print("Data preprocessing completed.")