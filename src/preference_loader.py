''' 
preference_loader.py

Used this article as a reference
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel


Also used an answer by Shani_Gamrian on the pytorch forums here
https://discuss.pytorch.org/t/typeerror-batch-must-contain-tensors-numbers-dicts-or-lists-found-object/14665/3
'''
import torch
from torch.utils import data
from torchvision import transforms
import readPreflib
import numpy as np
import pairwise

class Dataset(data.Dataset):

    def __init__(self, filename):
        # use readpreflib here
        num_votes, self.num_alts, votes = readPreflib.readtoMatrix(filename)
        # define self.len
        self.len = num_votes
        # Vote matrix
        L = []
        for num, vote in votes:
            row = np.zeros(self.num_alts, dtype=int)
            for i, v in enumerate(vote):
                row[i] = v
            for i in range(num):
                L.append(row)
        self.votes = np.vstack(L)
        # Pairwise matrix
        L_Pairwise = []
        for i in L:
            mat = pairwise.process_vote(i)
            L_Pairwise.append(torch.tensor(mat, dtype=torch.float))
        self.pairs = L_Pairwise
        
    
    def __len__(self):
        # One feature
        return self.len

    def __getitem__(self, index):
        # Generates one sample of data
        vec = self.pairs[index]
        return vec
        