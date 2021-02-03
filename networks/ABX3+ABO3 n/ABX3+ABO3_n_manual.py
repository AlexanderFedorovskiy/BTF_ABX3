import numpy as np
import networks.seqnet as sq
from tensorflow import keras

model = keras.models.load_model('M_abx3+abo3_n_tr82.77ts86.96')
model.summary()