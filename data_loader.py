import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem

# 1. Feature Extraction Functions
def smiles_to_fingerprint(smiles: str, radius=2, nBits=2048):
    """Converts SMILES string into Morgan Fingerprint (2048-bit vector)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp, dtype=np.float32)

def protein_to_aac(sequence: str):
    """Converts Protein Amino Acid Sequence into normalized frequency vector (20 standard AAs)."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    seq_len = len(sequence)
    if seq_len == 0:
        return None
    counts = {aa: sequence.count(aa) for aa in amino_acids}
    aac_vector = [counts[aa] / seq_len for aa in amino_acids]
    return np.array(aac_vector, dtype=np.float32)

# 2. PyTorch Dataset Class
class DTIDataset(Dataset):
    def __init__(self, df):
        self.drugs = []
        self.targets = []
        self.labels = []

        for _, row in df.iterrows():
            d_fp = smiles_to_fingerprint(row['SMILES'])
            t_aac = protein_to_aac(row['Protein_Seq'])
            
            if d_fp is not None and t_aac is not None:
                self.drugs.append(d_fp)
                self.targets.append(t_aac)
                self.labels.append(row['Label'])

        self.drugs = torch.tensor(np.array(self.drugs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.drugs[idx], self.targets[idx], self.labels[idx]

# 3. Create Sample Synthetic Dataset for Testing
def get_sample_dataloader(batch_size=4):
    sample_data = {
        'SMILES': [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'CC(=O)NC1=CC=C(O)C=C1'  # Paracetamol
        ],
        'Protein_Seq': [
            'MSLSLVTASALCRWRWKNGFGRWLPRAP',
            'MAGALRAGSLAPLLLLGAALLSGWAGGA',
            'MSLSLVTASALCRWRWKNGFGRWLPRAP',
            'MAGALRAGSLAPLLLLGAALLSGWAGGA'
        ],
        'Label': [1, 0, 1, 0]  # 1 = Interacting, 0 = Non-interacting
    }
    df = pd.DataFrame(sample_data)
    dataset = DTIDataset(df)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    loader = get_sample_dataloader()
    for drugs, targets, labels in loader:
        print(f"Drug Vector Shape: {drugs.shape}")    # Expected: [batch, 2048]
        print(f"Target Vector Shape: {targets.shape}")  # Expected: [batch, 20]
        print(f"Labels Shape: {labels.shape}")        # Expected: [batch, 1]
        break