# pairwise.py

# The goal of this class is to take a batch of preferences and return either
# a matrix of their pairwise probabilities or a flattened vector of them

import readPreflib as pref
import numpy as np
import math

def pairwise_from_votes(votes, num_candidates):

    n = num_candidates
    occurance_matrix = np.full(shape = (n,n), dtype = int, fill_value = 0)
    prob_matrix = np.full(shape = (n,n), dtype = float, fill_value = 0)
    for vote_ in votes:
        num_occurances, vote = vote_
        # print(vote)
        for i, v in enumerate(vote):
            after = vote[i+1:]
            # print('~',after)
            if after:
                for a in after:
                    occurance_matrix[v-1][a-1] += num_occurances
    for i, inner in enumerate(occurance_matrix):
        for j, num in enumerate(inner):
            if i != j:
                a_succ_b = occurance_matrix[i][j]
                b_succ_a = occurance_matrix[j][i]
                # print(a_succ_b, b_succ_a)
                prob_matrix[i][j] = a_succ_b / (a_succ_b + b_succ_a)
    prob_matrix = np.triu(prob_matrix, k = 1)
    return prob_matrix

# Converts an upper triangular matrix to a vector
def matrix_to_vec(matrix):
    vec = []
    n = len(matrix[0])
    vec_length = n * (n-1) / 2
    offset = 1
    for inner in matrix:
        vec.extend(inner[offset:])
        offset += 1
    return np.array(vec)

def vec_to_matrix(vec_):
    vec = list(vec_)
    m = len(vec)
    n = math.floor(math.sqrt(2*m))
    print(n)
    prob_matrix = np.full(shape = (n+1,n+1), dtype = float, fill_value = 0)
    row_offset = 1
    for row in prob_matrix:
        for i in range(row_offset,n+1):
            row[i] = vec.pop(0)
        row_offset += 1
    return prob_matrix



if __name__ == '__main__':
    print('Executing main thread in pairwise.py')
    np.set_printoptions(precision=3)
    candidates, votes = pref.readinSOIwfreqs('data_in/Practice/ED-02-Logo.soi')
    prob = pairwise_from_votes(votes, len(candidates))
    print(prob)
    vec = matrix_to_vec(prob)
    print(vec)
    print(vec_to_matrix(vec))