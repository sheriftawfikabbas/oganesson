import tensorflow as tf
import numpy as np
from m3gnet.models import M3GNet, Potential
from m3gnet.trainers import PotentialTrainer
import pandas as pd
import pickle

with open('MPF.2021.2.8/19470599/block_0.p', 'rb') as f:
    data = pickle.load(f)

# with open('MPF.2021.2.8/19470599/block_1.p', 'rb') as f:
#     data.update(pickle.load(f))

df = pd.DataFrame.from_dict(data)

cnt = 0

cnt = 0
dataset_train = []
for idx, item in df.items():
    for iid in range(len(item['energy'])):
        dataset_train.append({"atoms": item['structure'][iid], "energy": item['energy'][iid] / len(
            item['force'][iid]), "force": np.array(item['force'][iid]),  "stress": np.array(item['stress'][iid])})
dataset_val = dataset_train[80000:90000]
dataset_test = dataset_train[90000:]
dataset_train = dataset_train[:80000]

structures = []
energies = []
forces = []
stresses = []
val_structures = []
val_energies = []
val_forces = []
val_stresses = []
for item in dataset_train:
    structures += [item['atoms']]
    energies += [item["energy"]]
    forces += [item['force']]
    stresses += [item['stress']]
for item in dataset_val:
    val_structures += [item['atoms']]
    val_energies += [item["energy"]]
    val_forces += [item['force']]
    val_stresses += [item['stress']]


m3gnet = M3GNet(is_intensive=False)
potential = Potential(model=m3gnet)

trainer = PotentialTrainer(
    potential=potential, optimizer=tf.keras.optimizers.Adam(1e-3)
)

trainer.train(
    structures,
    energies,
    forces,
    stresses,
    validation_graphs_or_structures=val_structures,
    val_energies=val_energies,
    val_forces=val_forces,
    val_stresses=val_stresses,
    epochs=100,
    fit_per_element_offset=True,
    save_checkpoint=False,
    verbose=2
)
