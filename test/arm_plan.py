import sys
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch

sys.path.append('.')
from timeit import default_timer as timer

import open3d as o3d

from models import model_res_sigmoid as md


def Arm_FK(sampled_points, out_path_ ,path_name_,end_effect_):

    d = "cuda" if torch.cuda.is_available() else "cpu"
    robot_definition_file_path = Path(out_path_) / (path_name_ + ".urdf")
    with open(robot_definition_file_path) as robot_definition:
        chain = pk.build_serial_chain_from_urdf(robot_definition.read(), end_effect_)
        chain = chain.to(dtype=torch.float, device=d)
    th_batch = torch.tensor(2 * np.pi * sampled_points, requires_grad=True).cuda()
    tg_batch = chain.forward_kinematics(th_batch, end_only=False)
    p_list = []
    iter = 0
    point_size = 0
    mesh_list = []
    for tg in tg_batch:
        if iter > 1:
            mesh = o3d.io.read_triangle_mesh(
                out_path_ + "/meshes/visual/" + tg.replace("_link", "") + ".obj"
            )
            mesh_list.append(mesh)
            v = np.asarray(mesh.vertices)
            nv = np.ones((v.shape[0],4))
            point_size = point_size + v.shape[0]
            nv[:,:3] = v
            m = tg_batch[tg].get_matrix()
            t = torch.from_numpy(nv).float().cuda()
            p = torch.matmul(m[:], t.T)
            p = torch.permute(p, (0, 2, 1))
            p_list.append(p)
            del m, p, t, nv, v
        iter += 1
    whole_mesh = o3d.geometry.TriangleMesh()
    for ii in range(len(mesh_list)):
        mesh = mesh_list[ii]
        p = p_list[ii].detach().cpu().numpy()
        for jj in range(len(p)):
            pp = p[jj]
            mesh.vertices = o3d.utility.Vector3dVector(pp[:,:3])
            whole_mesh += mesh
    whole_mesh.compute_vertex_normals()
    return whole_mesh

modelPath = "./Experiments/UR5"
dataPath = "./datasets/arm/UR5"
model = md.Model(modelPath, dataPath, 6, device="cuda")
model.load("./Experiments/UR5/Model_Epoch_10000_ValLoss_3.526511e-03.pt")

for ii in range(10):
    XP = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.3, 0.4, 1.1, 0.5, -0.5, 0.0]]
    ).cuda()
    XP = torch.tensor(
        [[-2.2, 0.4, 1.1, 0.5, -0.5, 0.9, -1.3, 0.4, 1.1, 0.5, -0.5, 0.0]]
    ).cuda()
    BASE = torch.tensor(
        [
            [
                0,
                -0.5 * np.pi,
                0,
                -0.5 * np.pi,
                0,
                0,
                0,
                -0.5 * np.pi,
                0,
                -0.5 * np.pi,
                0,
                0,
            ]
        ]
    ).cuda()
    XP += BASE
    XP /= 2 * np.pi
    distance = torch.norm(XP[:, 6:] - XP[:, :6])
    point0 = []
    point1 = []
    point0.append(XP[:,:6])
    point1.append(XP[:,6:])
    start = timer()
    iter = 0
    while distance > 0.03:
        gradient = model.Gradient(XP.clone())
        XP = XP + 0.015 * gradient
        distance = torch.norm(XP[:, 6:] - XP[:, :6])
        point0.append(XP[:, :6])
        point1.append(XP[:, 6:])
        iter = iter + 1
        if iter > 300:
            break
    print(f"planned {iter} iterations in {round(timer() - start, 4)} seconds")
    point1.reverse()
    xyz = torch.cat(point0 + point1).to("cpu").data.numpy()
xyz0 = np.zeros((2, 6))
xyz0[0, :] = xyz[0, :]
xyz0[1, :] = xyz[-1, :]
whole_mesh = Arm_FK(xyz[0::1, :], "datasets/arm/UR5", "UR5", "wrist_3_link")

def length(path):
    size = path.shape[0]
    l = 0
    for i in range(size - 1):
        l += np.linalg.norm(path[i + 1, :] - path[i, :])
    return l
print(length(xyz))

mesh_path = Path("datasets/arm/") / "UR5" / "untitled_scaled.off"
obstacle = o3d.io.read_triangle_mesh(str(mesh_path))

vertices = np.asarray(obstacle.vertices)
obstacle.vertices = o3d.utility.Vector3dVector(vertices)
obstacle.compute_vertex_normals()
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.1, origin=[0, 0, 0]
)

o3d.visualization.draw_geometries([obstacle, whole_mesh, mesh_frame])
