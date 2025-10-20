"""
Script for work data preparation
"""
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch


class SequenceDataset(Dataset):
    def __init__(self, data_path, task_type='classification'):
        """
        Dataset for characters sequence.
        
        Args:
            data (pd.DataFrame): DataFrame with worws 'sequence' and 'target'.
            task_type (str): 'classification' or 'next_token'. If 'classification',
                            the target is the class of the whole sequence;
                            If 'next_token', target is the next char.
        """
        self.data = pd.read_csv(data_path)
        self.task_type = task_type
        all_chars = set("".join(self.data['sequence'].values))
        all_targets = self.data['target'].unique()
        self.char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
        self.target_to_idx = {target: idx for idx, target in enumerate(all_targets)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.idx_to_target = {idx: target for target, idx in self.target_to_idx.items()}


        self.sequences = [self.encode_sequence(seq) for seq in self.data['sequence']]
        
        if self.task_type == 'classification':
            # Codifica il target come classe
            self.targets = [self.target_to_idx[target] for target in self.data['target']]
        elif self.task_type == 'next_token':
            # Codifica il target come prossimo carattere
            self.targets = [self.char_to_idx[target] for target in self.data['target']]
        else:
            raise ValueError("task_type deve essere 'classification' o 'next_token'")

    def encode_sequence(self, sequence):
        return [self.char_to_idx[char] for char in sequence]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return sequence, target

