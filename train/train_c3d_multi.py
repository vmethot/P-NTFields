import sys

sys.path.append(".")
from models import model_res_sigmoid_multi as md

modelPath = './Experiments/C3D_multi'
dataPath = './datasets/c3d/'
model = md.Model(modelPath, dataPath, 3, 8, device="cuda:0")
model.train()
