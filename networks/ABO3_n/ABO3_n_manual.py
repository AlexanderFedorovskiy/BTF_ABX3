import numpy as np
import networks.seqnet as sq
from tensorflow import keras

model = keras.models.load_model('M_abo3_n_tr83.94ts86.44')
model.summary()