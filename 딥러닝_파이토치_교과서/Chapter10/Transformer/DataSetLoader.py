from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import os
import pandas as pd

os.chdir("딥러닝_파이토치_교과서\Chapter10\Transformer")

train_df = pd.read_csv("./train.txt", sep="\t")
train_df, val_df = train_df.iloc[:40000, :], train_df.iloc[40000:, :]

test_df = pd.read_csv("./test.txt", sep="\t")

print(len(train_df), len(val_df), len(test_df))


class TransformerDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]
        return text, label

    def __len__(self):
        return len(self.df)


def return_dataloaders():
    train_dataset = TransformerDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

    val_dataset = TransformerDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=0)

    test_dataset = TransformerDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)

    return train_loader, val_loader, test_loader
