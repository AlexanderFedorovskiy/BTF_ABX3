import numpy as np
import networks.seqnet as sq
from tensorflow import keras

model = keras.models.load_model('M_abx3_n_ri_tr86.92ts91.07')
model.summary()