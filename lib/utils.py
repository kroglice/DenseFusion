import logging
import numpy as np
import open3d as o3d
import copy

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    l.addHandler(streamHandler)
    return l


def cloud_to_dims(obj_cloud):

    obj_cloud = copy.deepcopy(obj_cloud)
    centroid = np.array([
        (np.amax(obj_cloud[:, 0]) + np.amin(obj_cloud[:, 0])) / 2,
        (np.amax(obj_cloud[:, 1]) + np.amin(obj_cloud[:, 1])) / 2,
        (np.amax(obj_cloud[:, 2]) + np.amin(obj_cloud[:, 2])) / 2,
    ])

    object_transform = np.eye(4)
    object_transform[0, 3] = -centroid[0]
    object_transform[1, 3] = -centroid[1]
    object_transform[2, 3] = -centroid[2]

    # Transform our point cloud. I.e. center and rotate to new frame.
    for i in range (obj_cloud.shape[0]):
        obj_cloud[i] = np.dot(object_transform, [obj_cloud[i][0], obj_cloud[i][1], obj_cloud[i][2], 1])[:3]

    width = np.amax(obj_cloud[:,0]) - np.amin(obj_cloud[:,0])
    height = np.amax(obj_cloud[:,1]) - np.amin(obj_cloud[:,1])
    depth = np.amax(obj_cloud[:,2]) - np.amin(obj_cloud[:,2])
    whd = [width, height, depth]

    w_from_center = max(abs(np.amax(obj_cloud[:,0])), abs(np.amin(obj_cloud[:,0])))
    h_from_center = max(abs(np.amax(obj_cloud[:,1])), abs(np.amin(obj_cloud[:,1])))
    d_from_center = max(abs(np.amax(obj_cloud[:,2])), abs(np.amin(obj_cloud[:,2])))
    sdf_whd = [w_from_center, h_from_center, d_from_center]

    bbox, connected_idxs = compute_3d_bbox(centroid, sdf_whd)
    return {'centroid': centroid, 'dim': whd, 'dim_from_centroid': sdf_whd, 'bbox': bbox,
            'connected_idxs': connected_idxs}


def compute_3d_bbox(centroid, dim_from_centroid):
    """
    Compute 3d bounding box of an object, given centroid and dimension form the centroid
    :param centroid:
    :param dim_from_centroid:
    :return: list of 8 corners of bbox
    """
    c = np.asarray(centroid).copy()
    dim = np.asarray(dim_from_centroid).copy()
    bbox = [c for _ in range(8)]
    bbox[:4] -= dim
    bbox[4:] += dim
    for i in range(3):
        bbox[i+1][i] += 2*dim[i]
    for i in range(3):
        bbox[i+5][i] -= 2*dim[i]

    connected_idx = []
    for i in range(5):
        for j in range(1, 8):
            a = bbox[i]
            b = bbox[j]
            if len(set(a).intersection(set(b)))==2:
                connected_idx.append([i, j])

    return bbox, connected_idx