import numpy as np
import pickle

data = dict(np.load("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/SMPL_MALE.npz", allow_pickle=True))
with open("SMPL_NEUTRAL.pkl", "wb") as f:
    pickle.dump(data, f)
