''' 
preference_loader.py

Used this article as a reference
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
'''
import torch
from torch.utils import data
import readPreflib

class Dataset(data.Dataset):

    def __init__(self, filename):
        # use readpreflib here
        num_votes, votes = readPreflib.readtoMatrix(filename)
        # define self.len
        self.len = num_votes
        self.votes = []
        for num, vote in votes:
            self.votes.extend([vote] * num)
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.votes[index]