from typing import Union
from ase import Atoms
from pymatgen.core import Structure
from oganesson.descriptors.descriptors import Descriptors
from oganesson.ogstructure import OgStructure
from typing import Union
from m3gnet.graph._converters import RadiusCutoffGraphConverter
from m3gnet.graph._compute import *
from m3gnet.utils._tf import *
from oganesson.ogstructure import OgStructure

def same(Ri, Rj):
    '''
    By Tri
    '''
    return tf.math.reduce_all(tf.math.equal(Ri,Rj),axis=1,keepdims=True)

def extract_vectors_from_graph(graph):
    '''
    By Tri
    '''
    pair_vectors_i, pair_vectors_j, atom_i, atom_j = get_pair_vector_from_graph(graph)

    edge1 = graph[Index.TRIPLE_BOND_INDICES][:, 0]
    edge2 = graph[Index.TRIPLE_BOND_INDICES][:, 1]
    R1 = tf.gather(pair_vectors_i, edge1)
    Z1 = tf.gather(atom_i, edge1)
    R2 = tf.gather(pair_vectors_j, edge1)
    Z2 = tf.gather(atom_j, edge1)
    R3 = tf.gather(pair_vectors_i, edge2)
    Z3 = tf.gather(atom_i, edge2)
    new_R3 = tf.where(same(R3, R1),tf.gather(pair_vectors_j,edge2[:int(len(R1))]) ,R3)
    new_Z3 = tf.where(same(R3, R1),tf.gather(atom_j,edge2[:int(len(R1))]) ,Z3)
    return R1,R2,new_R3,Z1,Z2,new_Z3

class Triplets(Descriptors):
    def __init__(self, structure: Union[Atoms, Structure, str, OgStructure]) -> None:
        super().__init__(structure)


    def describe(self):
        r = RadiusCutoffGraphConverter()
        mg = r.convert(self.structure)
        graph = tf_compute_distance_angle(mg.as_list())
        R1,R2,R3,Z1,Z2,Z3 = extract_vectors_from_graph(graph)
        
        descriptors_list = []
        return descriptors_list
