from graphviz import Digraph
import pairwise

# Given a vector of the form generated in pairwise.py for 
# easy reading into NNs, produce a diagram of the represented graph
def vec_to_graph(vec, name='no_name_graph', save=False, fromTorch=True):
    matrix = None
    if fromTorch:
        matrix = pairwise.vec_to_matrix(vec.numpy())
    else:
        matrix = pairwise.vec_to_matrix(vec)
    n_cands = len(matrix[0])
    dot = Digraph(comment='Preference Graph',format='png')
    # init nodes
    for i, row in enumerate(matrix):
        dot.node(chr(i+97), 'alt {}'.format(i+1))
    # init edges
    for i, row in enumerate(matrix):
        # only care about the upper triangluar part
        li = row[i+1:]
        for j, alt in enumerate(li):
            # math got confusing
            a = i+1
            b = i+j+2
            p_a = chr(a+96)
            p_b = chr(b+96)
            if alt == 1:
                dot.edge(p_a, p_b)
            elif alt == -1:
                dot.edge(p_b, p_a)
    file_output = '../diagrams/graph_views/{}'.format(name)
    if save:
        dot.render(file_output,view=False)
    return dot


def vote_to_graph(vote, name='no_name_graph', save=False):
    if 0 in vote:
        raise Exception('There should be no 0 values in vote vector')
    return vec_to_graph(pairwise.process_vote(vote), name, save, fromTorch=False)