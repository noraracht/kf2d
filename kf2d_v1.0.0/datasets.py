import torch
import pandas as pd
import numpy as np

class Dataset(torch.utils.data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """
    def __init__(self, df, list_IDs, label_idx):
        'Initialization'
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.kmer_frame = df
        self.list_IDs = list_IDs
        self.label_idx = label_idx


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, idx):
        'Generates one sample of data'
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # Select sample
        ID = self.list_IDs[idx]
        row_num = self.label_idx[ID]

        #print(row_num)

        # Load data and get label
        #X = torch.tensor(self.kmer_frame.iloc[row_num, :].values, dtype=torch.float64)
        #X = self.kmer_frame.iloc[list(row_num), :].values
        X = self.kmer_frame.iloc[row_num, :].values
        y = row_num

        # X = self.label_idx[ID]
        # y = self.list_IDs[idx]

        # print(X)
        # print(y)

        return X, y


# class Dataset(torch.utils.data.Dataset):
#     """
#     Characterizes a dataset for PyTorch
#     """
#     def __init__(self, df, list_IDs, label_idx):
#         'Initialization'
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#         """
#         self.kmer_frame = df
#         self.list_IDs = list_IDs
#         self.label_idx = label_idx
#
#
#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.list_IDs)
#
#     def __getitem__(self, idx):
#         'Generates one sample of data'
#         # if torch.is_tensor(idx):
#         #     idx = idx.tolist()
#
#         # Select sample
#         ID = self.list_IDs[idx]
#         row_num = self.label_idx[ID]
#
#         # Load data and get label
#         #X = torch.tensor(self.kmer_frame.iloc[row_num, :].values, dtype=torch.float64)
#         X = self.kmer_frame.iloc[row_num, :].values
#         y = row_num
#
#         # print(X)
#         # print(y)
#
#         return X, y


# class Lambda_Dataset(torch.utils.data.Dataset):
#   """
#   This is a custom dataset class.
#   """
#   def __init__(self, X):
#     self.X = X
# #    self.Y = Y
# #     if len(self.X) != len(self.Y):
# #       raise Exception("The length of X does not match the length of Y")
#
#   def __len__(self):
#     return len(self.X)
#
#   def __getitem__(self, index):
#     # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
#     _x = self.X[index]
#     _y = index
#
#     return _x, _y
#

