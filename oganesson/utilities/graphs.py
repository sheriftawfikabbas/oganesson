from m3gnet.graph._compute import *
from m3gnet.utils._tf import *

def get_pair_vector_from_graph(graph: list):
    """
    Given a graph list return pair vectors that form the bonds
    Args:
        graph (List): graph list, obtained by MaterialGraph.as_list()

    Returns: pair vector tf.Tensor

    """
    atom_positions = graph[Index.ATOM_POSITIONS]
    lattices = graph[Index.LATTICES]
    pbc_offsets = graph[Index.PBC_OFFSETS]
    bond_atom_indices = graph[Index.BOND_ATOM_INDICES]
    n_bonds = graph[Index.N_BONDS]
    if lattices is not None:
        lattices = tf.gather(lattices, get_segment_indices_from_n(n_bonds))
        offset_vec = tf.keras.backend.batch_dot(tf.cast(pbc_offsets, DataType.tf_float), lattices)
    else:
        offset_vec = tf.constant([[0.0, 0.0, 0.0]], dtype=DataType.tf_float)
    diff = (
        tf.gather(atom_positions, bond_atom_indices[:, 1])
        + offset_vec
    )
    return tf.cast(tf.gather(atom_positions, bond_atom_indices[:, 0]), DataType.tf_float), \
        tf.cast(diff, DataType.tf_float),\
        tf.gather(graph[Index.ATOMS], graph[Index.BOND_ATOM_INDICES][:, 0]), tf.gather(
            graph[Index.ATOMS], graph[Index.BOND_ATOM_INDICES][:, 1])
