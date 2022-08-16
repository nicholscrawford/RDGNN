import pickle
import os
import numpy as np
from farthest_point_sampling import farthest_point_sampling
import sys
from colorama import Fore, Back, Style

#Check if data has been processed, and if it hasn't process it.
def pc_and_sample(path_list):
    print(f"{Fore.LIGHTGREEN_EX}{Back.BLACK}[INFO] Generating point clouds and sampling{Style.RESET_ALL}")
    for path in path_list:
        files = sorted(os.listdir(path))
        files = [(os.path.join(path, f)) for f in files if "demo" in f]
        
        #Fun little progress bar vars
        n = len(files)
        i = 1

        for path in files:
            with open(path, 'rb') as file:
                data, attributes = pickle.load(file)
                
                #Check if the data has been processed before,
                if "rdgnn_ready" not in attributes.keys():
                    create_pointcloud(data)
                    sample_pointcloud(data)

                    attributes["rdgnn_ready"] = True

                    #Resave file
                    with open(path, 'wb') as savefile:
                        pickle.dump((data, attributes), savefile)

                #Print progress bar
                sys.stdout.write('\r')
                sys.stdout.write("[{:{}}] {:.1f}%".format("="*i, n, (100/(n)*i)))
                sys.stdout.flush()
                i += 1
    print()
    print(f"{Fore.LIGHTGREEN_EX}{Back.BLACK}[INFO] Data processed{Style.RESET_ALL}")


def create_pointcloud(data: dict):
    data["point_clouds"] = []

    for timestep_idx in range(len(data["depth"])):
        points = []

        data["point_clouds"].append({})

        for object_name in data["objects"].keys():
            data["point_clouds"][timestep_idx][object_name] = []

        color = []
        for i_o in range(len(data["objects"])):
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
                
                for i_o in range(len(data["objects"])):
                    #Assumes object order corresponds to their seg number, which I think is true, but could be good to confirm.
                    if seg_buffer[j, i] == i_o + 1:
                        u = -(i-centerU)/(cam_width)  # image-space coordinate
                        v = (j-centerV)/(cam_height)  # image-space coordinate
                        d = depth_buffer[j, i]  # depth buffer value
                        X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                        p2 = X2*vinv  # Inverse camera view to get world coordinates
                        points[i_o].append([p2[0, 2], p2[0, 0], p2[0, 1]])
                        color.append(0)
        for i_o in range(len(points)):
            points[i_o] = np.array(points[i_o])
        
        idx = 0
        for object_name in data["objects"].keys():
            data["point_clouds"][timestep_idx][object_name] = points[idx]
            idx += 1


def sample_pointcloud(data: dict):
    
    #Sampling number should probably be configurable, especially since it is needed by model. Alternatively, model could just
    #read what it is from data.
    sampling_number = 128

    for timestep_idx in range(len(data["depth"])):
        for object_name in data["objects"]:
            pc = data["point_clouds"][timestep_idx][object_name]
            farthest_indices,_ = farthest_point_sampling(pc, sampling_number)
            data["point_clouds"][timestep_idx][object_name] = farthest_indices
