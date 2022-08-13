from copyreg import pickle
import os
import numpy as np

#Check if data has been processed, and if it hasn't process it.
def pc_and_sample(path):
    files = sorted(os.listdir(path))
    files = [(os.path.join(path, f)) for f in files if "demo" in f]
    
    for path in files:
        with open(path, 'rb') as file:
            data, attributes = pickle.load(file)

            #Check if the data has been processed before,
            if(attributes["rdgnn_ready"] == False):
                create_pointcloud(data)
                sample_pointcloud(data)


def create_pointcloud(data: dict):
    total_objects = 0
    
    for timestep_idx in range(len(data["depth"])):
        points = []

        color = []
        for i_o in range(total_objects):
            points.append([])
        cam_width = 512
        cam_height = 512

        # Retrieve depth and segmentation buffer
        depth_buffer = data["depth"][timestep_idx]
        seg_buffer = data["segmentation"][timestep_idx]
        view_matrix = data["view_matrix"][0]
        projection_matrix = data["projection_matrix"][0]

        # Get camera view matrix and invert it to transform points from camera to world space
        #print(view_matrix)
        vinv = np.linalg.inv(np.matrix(view_matrix))

        # Get camera projection matrix and necessary scaling coefficients for deprojection
        proj = projection_matrix
        fu = 2/proj[0, 0]
        fv = 2/proj[1, 1]

        # Ignore any points which originate from ground plane or empty space
        depth_buffer[seg_buffer == 0] = -10001

        centerU = cam_width/2
        centerV = cam_height/2
        for i in range(cam_width):
            for j in range(cam_height):
                if depth_buffer[j, i] < -10000:
                    continue
                # This will take all segmentation IDs. Can look at specific objects by
                # setting equal to a specific segmentation ID, e.g. seg_buffer[j, i] == 2
                
                for i_o in range(total_objects):
                    if seg_buffer[j, i] == i_o + 1:
                        u = -(i-centerU)/(cam_width)  # image-space coordinate
                        v = (j-centerV)/(cam_height)  # image-space coordinate
                        d = depth_buffer[j, i]  # depth buffer value
                        X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                        p2 = X2*vinv  # Inverse camera view to get world coordinates
                        points[i_o].append([p2[0, 2], p2[0, 0], p2[0, 1]])
                        color.append(0)
        for i_o in range(total_objects):
            points[i_o] = np.array(points[i_o])
        return points #np.array(points1), np.array(points2), np.array(points3)


def sample_pointcloud(data: dict):
    pass