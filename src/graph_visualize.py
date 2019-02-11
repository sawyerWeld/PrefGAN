from graphviz import Digraph
import pairwise

# Given a vector of the form generated in pairwise.py for 
# easy reading into NNs, produce a diagram of the represented graph
def vec_to_graph(vec, name='no_name_graph'):
    matrix = pairwise.vec_to_matrix(vec.numpy())
    n_cands = len(matrix[0])
    dot = Digraph(comment='Preference Graph')
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
    dot.render('../diagrams/graph_views/{}.gv'.format(name),view=True)