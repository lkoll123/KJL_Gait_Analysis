import joblib
import torch
import numpy as np
from smplx import SMPL
import cv2 as cv

SMPL_MODEL_PTH = 'dataset/body_models/smpl'
OUTPUT_FILE_PATH = 'output/demo/Normal_Andrew1/wham_output.pkl'
SLAM_FILE_PATH = 'output/demo/Normal_Andrew1/slam_results.pth'


target_subject_id = 0





wham_results = joblib.load(OUTPUT_FILE_PATH)[target_subject_id]






pose = wham_results["pose"]
pose_world = wham_results["pose_world"]
trans_world = wham_results["trans_world"]
betas = wham_results["betas"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
body_model = SMPL(SMPL_MODEL_PTH, gender='neutral', num_betas=10).to(device)

toTensor = lambda x: torch.from_numpy(x).float().to(device)

smpl_kwargs = dict(
    global_orient=toTensor(pose_world[..., :3]),
    body_pose=toTensor(pose_world[..., 3:]),
    betas=toTensor(betas),
    transl=toTensor(trans_world)
)

smpl_output = body_model(**smpl_kwargs)



joints = smpl_output.joints
vertices = smpl_output.vertices


print(len(joints))
for i in range(0, 182):
    print(f"left {i}: {joints[i][7]}")
    #print(f"right {i}: {joints[i][8][1]}")

for i in range(0, 182):
    #print(f"left {i}: {joints[i][7]}")
    print(f"right {i}: {joints[i][8]}")
    










