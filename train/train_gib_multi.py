import sys

sys.path.append('.')
from models import model_res_sigmoid_multi as md

modelPath = "./Experiments/Gib_multi"
dataPath = "./datasets/gibson/"
model = md.Model(modelPath, dataPath, 3, 2, device="cuda:0")
model.train()
