# %%
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch
from torch.utils.data import DataLoader



def status_to_score(status):
    mapping = {
        1.0: 1.0,  # neuf
        2.0: 0.7,  # très bon état
        3.0: 0.5,  # bon état
        4.0: 0.2   # mauvais état
    }
    return mapping.get(status, 0.0)

class VintedDescriptionStatusDataset(Dataset):
    def __init__(self, dataframe):
        self.descriptions = dataframe["Description"].fillna("").tolist()
        self.statuses = dataframe["ItemStatus"].fillna(0).tolist()
        self.labels = [status_to_score(s) for s in self.statuses]

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        return (
            self.descriptions[idx],
            str(self.statuses[idx]),  # en string pour concaténation avec le texte
            torch.tensor(self.labels[idx], dtype=torch.float)
        )


# %%
class DescriptionStatusToScoreModel(nn.Module):
    def __init__(self, model_name="xlm-roberta-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()  # car score entre 0 et 1
        )

    def forward(self, descriptions, statuses):
        inputs = [f"{d} [SEP] {s}" for d, s in zip(descriptions, statuses)]
        tokens = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        output = self.encoder(**tokens)
        cls_token = output.last_hidden_state[:, 0, :]  # [CLS]
        return self.regressor(cls_token).squeeze()
    
def predict_score_for_single_item(model, description, status):
    model.eval()
    with torch.no_grad():
        score = model([description], [status]).item()
    return score


if __name__ == "__main__":
    # Load the model and tokenizer
    model = DescriptionStatusToScoreModel()
    model.load_state_dict(torch.load("model_description_status.pt"))
    model.eval()

    # Load the dataset
    df = pd.read_csv("manteaux_labelise.csv")

    # Predict scores
    df1 = predict_scores_from_dataframe(model, df)
