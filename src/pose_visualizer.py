import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2 as cv
import torch

# Example 3D pose coordinate data (x, y, z)


def read_tensors_from_file(filename):
    with open(filename, 'r') as file:
        data = file.read()

    # Split the data by the tensor keyword and filter out any empty strings
    tensor_strings = [s for s in data.split('tensor(') if s]

    tensors = []
    for tensor_str in tensor_strings:
        # Extract the content inside the brackets and remove the 'device='cuda:0' part
        tensor_content = tensor_str.split(')', 1)[0].strip()
        tensor_content = tensor_content.replace(", device='cuda:0'", "")
        tensor = eval(f'torch.tensor({tensor_content})')
        tensors.append(tensor)

    return tensors


read_tensors = read_tensors_from_file('/Users/lukakoll/Desktop/SimCenterData/normal_Andrew_Keypoints.txt')
for tensor in read_tensors:
    print(tensor)

    pose_coordinates = tensor  

    # Create a new figure for plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = [point[0] for point in pose_coordinates]
    y = [point[2] for point in pose_coordinates]
    z = [point[1] for point in pose_coordinates]

    # Scatter plot the points
    ax.scatter(x, y, z, c='r', marker='o')

    # Define the keypoints to connect with a line (e.g., between keypoints 1 and 12)
    keypoints_to_connect = [(0, 1), (0, 2), (1, 4), (2, 5), (4, 7), (5, 8), (7, 10), (8, 11),
                            (0, 3), (3, 6), (6, 9), (9, 12), (9, 13), (9, 14), (13, 16), (14, 17),
                            (16, 18), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23), (12, 15)]

    # Plot the connections
    for start, end in keypoints_to_connect:
        point1 = pose_coordinates[start]
        point2 = pose_coordinates[end]
        ax.plot(
            [point1[0], point2[0]],
            [point1[2], point2[2]],
            [point1[1], point2[1]],
            color='b'
        )

    # Set labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()
    cv.waitKey(1)
