import pickle

import numpy as np

with open('samples_2.pkl', 'rb') as f:
    a = pickle.load(f)

with open('samples_3.pkl', 'rb') as f:
    b = pickle.load(f)

c = np.concatenate((a, b))
print(c.shape)

with open('room_samples.pkl', 'wb') as f:
    pickle.dump(c, f, protocol=pickle.HIGHEST_PROTOCOL)
