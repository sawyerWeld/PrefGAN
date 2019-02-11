# pairwise.py

# The goal of this class is to take a batch of preferences and return either
# a matrix of their pairwise probabilities or a flattened vector of them

import readPreflib as pref
import numpy as np
import math

# Acts on a set of SOI vote tuples
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

# Acts on a single np array
# This one gives a negative one if the 1 would go in the lower triangular bit
def vote_vector_to_pairwise(vote): 
    n = len(vote)
    occurance_matrix = np.full(shape = (n,n), dtype = int, fill_value = 0)
    for i, v in enumerate(vote):
        after = vote[i+1:]
        for a in after:
            if a != 0:
                if v < a:
                    occurance_matrix[v-1][a-1] = 1
                else:
                    occurance_matrix[a-1][v-1] = -1
    return occurance_matrix

# previous iterations did not think about the fact that 
# a vote [1 0 0] does not indicate nothing, it indicates
# that 1 succ 2 and 1 succ 3
def pairwise_matrix_singular(vote):
    n = len(vote)
    occurance_matrix = np.full(shape = (n,n), dtype = int, fill_value = 0)
    
    for i, v in enumerate(vote):
        if v == 0:
            continue
        # list of alts that the current alt is better than
        better_than = [i+1 for i in range(n)]
        before = vote[:i+1]
        for b in before:
            better_than.remove(b)
        for p in better_than:
            occurance_matrix[v-1][p-1] = 1
            occurance_matrix[p-1][v-1] = -1
    return occurance_matrix

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
    data_type = vec_.dtype
    vec = list(vec_)
    m = len(vec)
    n = math.floor(math.sqrt(2*m))
    prob_matrix = np.full(shape = (n+1,n+1), dtype = data_type, fill_value = 0)
    row_offset = 1
    for row in prob_matrix:
        for i in range(row_offset,n+1):
            row[i] = vec.pop(0)
        row_offset += 1
    return prob_matrix


def process_vote(vote):
    mat = pairwise_matrix_singular(vote)
    # print(mat)
    return matrix_to_vec(mat)

# deprecated?
# This is what should be used from preference loader
# just a wrapper function
def process_vote_depr(vote):
    mat = vote_vector_to_pairwise(vote)
    vec = matrix_to_vec(mat)
    return vec

if __name__ == '__main__':
    print('Executing main thread in pairwise.py')
    np.set_printoptions(precision=3)
    candidates, votes = pref.readinSOIwfreqs('data_in/Practice/ED-02-Logo.soi')
    a = np.array([2, 0, 0, 0, 0, 0, 0, 0])
    vec = process_vote(a)
    print(vec)
    print('mat now')
    #print(matrix_to_vec(vec))
    # prob = pairwise_from_votes(votes[0], len(candidates))
    # print(prob)
    # vec = matrix_to_vec(prob)
    # print(vec)
    print(vec_to_matrix(vec))