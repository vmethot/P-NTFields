import sys

sys.path.append('.')
from timeit import default_timer as timer

import igl
import numpy as np
import open3d as o3d
import torch
from torch import Tensor
from torch.autograd import Variable

from models import model_res_sigmoid_multi as md

GIB_ID = 0  # 1 also works

modelPath = "./Experiments/Gib"
dataPath = "./datasets/gibson/"
world_model = md.Model(modelPath, dataPath, 3, 2, device="cuda")
world_model.load("./Experiments/Gib_multi/Model_Epoch_10000_ValLoss_1.221157e-01.pt")  #
world_model.network.eval()
max_x, max_y, max_z = 0, 0, 0
v, faces = igl.read_triangle_mesh(
    "datasets/gibson/" + str(GIB_ID) + "/mesh_z_up_scaled.off"
)
vertices = torch.tensor(20 * v, dtype=torch.float32, device="cuda")
faces = torch.tensor(faces, dtype=torch.long, device="cuda")
triangles = vertices[faces].unsqueeze(dim=0)
B = np.load("datasets/gibson/" + str(GIB_ID) + "/B.npy")
B = Variable(Tensor(B)).to('cuda')

for ii in range(5):
    XP = np.array([[-6, -7, -6, 2, 7, -2.5]])
    XP = Variable(Tensor(XP)).to("cuda") / 20.0
    distance = torch.norm(XP[:, 3:6] - XP[:, 0:3])
    start = timer()
    point0 = []
    point1 = []
    point0.append(XP[:,0:3].clone())
    point1.append(XP[:, 3:6].clone())
    iter = 0
    while distance > 0.06:
        gradient = world_model.Gradient(XP.clone(), B)
        XP = XP + 0.03 * gradient
        distance = torch.norm(XP[:, 3:6] - XP[:, 0:3])
        point0.append(XP[:,0:3].clone())
        point1.append(XP[:,3:6].clone())
        iter = iter + 1
        if(iter>500):
            break
    print(f"planned {iter} iterations in {round(timer() - start, 4)} seconds")
point1.reverse()
xyz = 20 * torch.cat(point0 + point1).to("cpu").data.numpy()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
mesh = o3d.io.read_triangle_mesh(
    "datasets/gibson/" + str(GIB_ID) + "/mesh_z_up_scaled.off"
)
mesh.scale(20, center=(0, 0, 0)).compute_vertex_normals()
o3d.visualization.draw_geometries([mesh,pcd])
