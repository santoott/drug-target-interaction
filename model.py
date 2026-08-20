import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepDTI(nn.Module):
    def __init__(self, drug_dim=2048, protein_dim=20, hidden_dim=128):
        super(DeepDTI, self).__init__()
        
        # 1. Drug Feature Extractor Branch
        self.drug_fc = nn.Sequential(
            nn.Linear(drug_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, hidden_dim),
            nn.ReLU()
        )
        
        # 2. Protein Feature Extractor Branch
        self.protein_fc = nn.Sequential(
            nn.Linear(protein_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, hidden_dim),
            nn.ReLU()
        )
        
        # 3. Combined Classifier Branch (Interaction Predictor)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Binary output (logits)
        )

    def forward(self, drug, protein):
        # Extract features for both branches
        d_feat = self.drug_fc(drug)
        p_feat = self.protein_fc(protein)
        
        # Concatenate drug and protein features
        combined = torch.cat((d_feat, p_feat), dim=1)
        
        # Predict interaction
        out = self.classifier(combined)
        return out

if __name__ == "__main__":
    # Test model forward pass with dummy inputs
    model = DeepDTI()
    dummy_drug = torch.randn(4, 2048)
    dummy_protein = torch.randn(4, 20)
    
    predictions = model(dummy_drug, dummy_protein)
    print("Model Output Shape:", predictions.shape)  # Expected: [4, 1]
    print("Sample Raw Output (Logits):\n", predictions.detach().numpy())