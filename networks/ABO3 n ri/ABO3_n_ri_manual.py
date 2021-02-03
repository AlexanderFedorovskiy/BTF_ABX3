import numpy as np
import networks.seqnet as sq
from tensorflow import keras

model = keras.models.load_model('M_abo3_n_ri_tr97.81ts98.31')
model.summary()