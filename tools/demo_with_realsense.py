# Adapted image acquisition from realsense demo code

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
import numpy.ma as ma
import random
# Import OpenCV for easy image rendering
import cv2
import open3d as o3d
import torch
from torch.autograd import Variable
from torchvision import transforms
import sys
sys.path.append("..")
from vanilla_segmentation.segnet import SegNet as segnet
from vanilla_segmentation.data_controller import SegDataset
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import mask_to_bbox, get_bbox
from lib.network import PoseNet, PoseRefineNet
from lib.utils import cloud_to_dims, iterative_points_refine
from visualize_bbox import PoseYCBDataset_visualize
import argparse
import collections
import time

cropped_w, cropped_h = 160, 160
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='', help="dataset root dir (''YCB_Video Dataset'')")
parser.add_argument('--batch_size', type=int, default=3, help="batch size")
parser.add_argument('--img_w', type=int, default=640, help="image width")
parser.add_argument('--img_h', type=int, default=480, help="image height")
parser.add_argument('--n_epochs', type=int, default=600, help="epochs to train")
parser.add_argument('--log_interval', type=int, default=20, help="epochs to train")
parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
parser.add_argument('--logs_path', default='logs/', help="path to save logs")
parser.add_argument('--segnet_model_save_path', default='../vanilla_segmentation/trained_models', help="path to save models")
parser.add_argument('--log_dir', default='logs/', help="path to save logs")
parser.add_argument('--segnet_model', default='model_44_0.11587538284994661.pth', help="resume segnet model name")
parser.add_argument('--pose_model_save_path', default='../trained_models/ycb', help="path to save models")
parser.add_argument('--pose_model', type=str, default='pose_model_26_0.012863246640872631.pth',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default='pose_refine_model_69_0.009449292959118935.pth',  help='resume PoseRefineNet model')
opt = parser.parse_args()


def get_intrinsic_matrix(frame, img_w=640, img_h=480):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = o3d.camera.PinholeCameraIntrinsic(img_w, img_h, intrinsics.fx,
                                            intrinsics.fy, intrinsics.ppx,
                                            intrinsics.ppy)
    return out


def get_intrinsic_info(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    return intrinsics.ppx, intrinsics.ppy, intrinsics.fx, intrinsics.fy


def pc_with_o3d(color_frame, depth_image, clipping_distance_in_meters, depth_scale):
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    color_image = np.asanyarray(color_frame.get_data())
    color_image = color_image[:,:,::-1].astype("uint8") # convert bgr to rgb
    # color_image = color_image.astype("uint8")
    depth_image_o3d = o3d.geometry.Image(depth_image)
    color_image_o3d = o3d.geometry.Image(color_image)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        get_intrinsic_matrix(color_frame))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image_o3d,
        depth_image_o3d,
        depth_scale=1.0 / depth_scale,
        depth_trunc=clipping_distance_in_meters,
        convert_rgb_to_intensity=False)

    temp = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic)
    temp.transform(flip_transform)
    # pcd.points = temp.points
    # pcd.colors = temp.colors
    return temp


def bbox_of_obj_semantic(obj_semantic, obj_label, image=None):
    max_v = np.max(obj_semantic)
    min_v = np.min(obj_semantic)
    threshold = max(0.0, 0.8*(max_v - min_v) + min_v)
    roi = ma.getmaskarray(ma.masked_greater_equal(obj_semantic, threshold)).astype(np.uint8)
    roi_idx = np.where(obj_semantic >= threshold)
    # color = (0, 0, 255)
    color = tuple(np.random.randint(100, 255, (0, 3)))
    if len(roi_idx[0]) > 0:
        r_min = np.min(roi_idx[0])
        r_max = np.max(roi_idx[0])
        c_min = np.min(roi_idx[1])
        c_max = np.max(roi_idx[1])
        # print(r_min, r_max, c_min, c_max)
        # TODO: make this work for semantic map, draw bbox in original image with label
        if image is not None:
            cv2.rectangle(image, (c_min, r_min), (c_max, r_max), color, thickness=3)
            cv2.putText(image, obj_label, (c_min, r_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return [r_min, r_max, c_min, c_max]
    else:
        return None


def bbox_obj_mask(mask_idx, obj_label, image=None, color=None):
    # mask_idx = ma.masked_equal (obj_masks, obj_idx)
    color = [np.random.randint(0, 255) for _ in range(3)] if color is None else color
    obj_mask = ma.getmaskarray(mask_idx).astype(np.uint8)
    obj_mask = removeSmallComponents(obj_mask, 200)
    bbox = mask_to_bbox(obj_mask)
    r_min, r_max, c_min, c_max = get_bbox(bbox)
    if image is not None:
        cv2.rectangle(image, (c_min, r_min), (c_max, r_max), color, 3)
        cv2.putText(image, obj_label, (c_min, r_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color, thickness=2)
    return r_min, r_max, c_min, c_max, obj_mask


def removeSmallComponents(image, threshold):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    img2 = np.zeros((output.shape), dtype=np.uint8)
    # for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2


def find_idx_with_name(list_obj, obj_name):
    obj_idx = None
    for k, v in list_obj.items():
        if obj_name in v:
            obj_idx = k
            break

    return obj_idx


def generate_colors(n):
    rgb_values = []
    r = int(np.random.random() * 255)
    g = int(np.random.random() * 255)
    b = int(np.random.random() * 255)
    step = 255 / n
    for _ in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 255
        g = int(g) % 255
        b = int(b) % 255
        rgb_values.append((r,g,b))
    return rgb_values


class ObjectPoseEstimate(object):
    def __init__(self):
        self.visualize_with_o3d = False
        self.visualize_with_o3d = False
        self.num_points = 500

        # ycb_dataset = PoseDataset_ycb('test', num_points, False, opt.dataset_root, 0.0, True)
        self.ycb_dataset = PoseYCBDataset_visualize('test', self.num_points, False, opt.dataset_root, 0.0, True)
        self.list_obj = self.ycb_dataset.list_obj
        self.num_objects = len(self.list_obj)

        # Create models (segmentation and pose estimation) and load trained checkpoints
        self.segmenter = segnet()
        self.segmenter = self.segmenter.cuda()

        self.estimator = PoseNet(num_points = self.num_points, num_obj = self.num_objects)
        self.estimator.cuda()
        self.refiner = PoseRefineNet(num_points = self.num_points, num_obj = self.num_objects)
        self.refiner.cuda()
        self.estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.pose_model_save_path, opt.pose_model)))
        self.refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.pose_model_save_path, opt.refine_model)))
        self.estimator.eval()
        self.refiner.eval()

        # rgb_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.list_obj[0] = 'background'  # SegNet output has dimension of num_objects + background
        self.rgb_norm = self.ycb_dataset.norm

        if opt.segnet_model != '':
            checkpoint = torch.load('{0}/{1}'.format(opt.segnet_model_save_path, opt.segnet_model))
            self.segmenter.load_state_dict(checkpoint)
        self.segmenter.eval()

        obj_name = ['tomato']
        # self.obj_idx = find_idx_with_name(self.list_obj, obj_name)
        # self.model_points = self.ycb_dataset.cld[self.obj_idx]
        self.obj_idx = []
        self.model_points = []
        for name in obj_name:
            idx = find_idx_with_name(self.list_obj, name)
            self.obj_idx.append(idx)
            self.model_points.append(self.ycb_dataset.cld[idx])
        print('object idx {}'.format(self.obj_idx))


        # Create a pipeline
        self.pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream different resolutions of color and depth streams
        config = rs.config()
        config.enable_stream(rs.stream.depth, opt.img_w, opt.img_h, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, opt.img_w, opt.img_h, rs.format.bgr8,
                             30)  # color image streamed in bgr format

        # Start streaming
        profile = self.pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", self.depth_scale)
        self.cam_cx, self.cam_cy, self.cam_fx, self.cam_fy = None, None, None, None

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        self.clipping_distance_in_meters = 2  # 1 meter
        clipping_distance = self.clipping_distance_in_meters / self.depth_scale

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        if self.visualize_with_o3d:
            # open3d visualization
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()

            pcd = o3d.geometry.PointCloud()
            frame_count = 0

        self.rgb_colors = generate_colors(len(self.list_obj))
        self.streamer = self.stream_rgbd()

    def stream_rgbd(self):
        """
        Grasp and stream rgbd image obtained from realsense
        :return: a generator of [bgr image, depth image]
        """
        while True:
            # Get frameset of color and depth
            frames = self.pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            self.cam_cx, self.cam_cy, self.cam_fx, self.cam_fy = get_intrinsic_info(color_frame)
            self.ycb_dataset.update_cam_info([self.cam_cx, self.cam_cy, self.cam_fx, self.cam_fy])

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            # obtain rgb and depth image from the buffer
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())     # in bgr format
            yield [color_image, depth_image]

    def stream_rgbd_(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        self.cam_cx, self.cam_cy, self.cam_fx, self.cam_fy = get_intrinsic_info(color_frame)
        self.ycb_dataset.update_cam_info([self.cam_cx, self.cam_cy, self.cam_fx, self.cam_fy])

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return None, None

        # obtain rgb and depth image from the buffer
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())  # in bgr format
        return color_image, depth_image

    def segment_objects(self, rgb, color_image=None, bbox_obj_id=None):
        """
        Segment all objects from the list_obj of the dataset
        :param rgb: rgb image
        :param color_image: original color_image from the camera, which is used to draw the bbox
        :param bbox_obj_id: index of object(s) to draw the bbox
        :return: [n_obj, h, w] array of all object mask (after removed small portion), dictt of bboxes of all object,
        [1, h, w] image of all object masks
        """
        if bbox_obj_id is not None:
            bbox_obj_id = [bbox_obj_id] if not isinstance(bbox_obj_id, collections.Iterable) else bbox_obj_id
        rgb_ = self.rgb_norm(torch.from_numpy(rgb.astype(np.float32)))  # normalize and convert to torch tensor
        rgb_ = Variable(rgb_.unsqueeze(0)).cuda()  # expand dimension as fake batch size
        semantic = self.segmenter(rgb_)
        semantic = semantic[0].cpu().detach().numpy()  # array of (nb_objs, h, w)
        semantic_img = np.argmax(semantic, axis = 0)  # convert each pixel as the index of object with highest score

        segmented_bboxes = {}
        obj_masks = {}
        obj_masks_all = np.zeros_like(semantic_img)
        # for i in range(1, len(list_obj)):
        for i in range(1, len(self.list_obj)):
            mask_idx = ma.masked_equal(semantic_img, i)
            color = self.rgb_colors[i]
            # calculate bbox and draw them
            # obj_mask is equivalent to the mask_label in dataset
            img = color_image if i in bbox_obj_id else None
            # r_min, r_max, c_min, c_max, obj_mask = bbox_obj_mask(mask_idx, list_obj[i], None, color)
            r_min, r_max, c_min, c_max, obj_mask = bbox_obj_mask(mask_idx, self.list_obj[i], img, color)
            segmented_bboxes[i] = [r_min, r_max, c_min, c_max]
            obj_masks[i] = obj_mask
            obj_masks_all += obj_mask // 255 * i

        return obj_masks, segmented_bboxes, obj_masks_all

    def object_pose_estimate(self, obj_idx, obj_points, rgb, depth_image, obj_masks, segmented_bboxes, color_image = None):
        r_min, r_max, c_min, c_max = segmented_bboxes[obj_idx]
        obj_mask = obj_masks[obj_idx]
        # choose = obj_mask[r_min:r_max, c_min:c_max].flatten().nonzero()[0]
        choose = np.flatnonzero(obj_mask[r_min:r_max, c_min:c_max])
        if len(choose) > 0:
            if len(choose) > self.num_points:
                c_mask = np.zeros(len(choose), dtype = int)
                c_mask[:self.num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')

            depth_masked = depth_image[r_min:r_max, c_min:c_max].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = xmap[r_min:r_max, c_min:c_max].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = ymap[r_min:r_max, c_min:c_max].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            # pt2 = depth_masked / depth_scale    # cam_scale
            pt2 = depth_masked * self.depth_scale  # cam_scale
            pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
            pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis = 1)

            # img_masked = np.array(img)[:, :, :3]
            # img_masked = np.transpose(img_masked, (2, 0, 1))
            img_masked = rgb[:, r_min:r_max, c_min:c_max]

            cloud = torch.from_numpy(cloud.astype(np.float32))
            choose = torch.LongTensor(choose.astype(np.int32))
            img_masked = self.rgb_norm(torch.from_numpy(img_masked.astype(np.float32)))
            index = torch.LongTensor([obj_idx - 1])

            cloud = Variable(cloud).cuda()
            choose = Variable(choose).cuda()
            img_masked = Variable(img_masked).cuda()
            index = Variable(index).cuda()

            cloud = cloud.view(1, self.num_points, 3)
            img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

            pred_r, pred_t, pred_c, emb = self.estimator(img_masked, cloud, choose, index)
            pred_r = pred_r / torch.norm(pred_r, dim = 2).view(1, self.num_points, 1)
            pred_c = pred_c.view(1, self.num_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(1 * self.num_points, 1, 3)

            # get the rotation and translation of the most confident predicted point from the cloud set
            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (cloud.view(1 * self.num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            my_pred = np.append(my_r, my_t)
            # print('check', cloud, emb, index, my_r, my_t)
            _, my_r, my_t = iterative_points_refine(self.refiner, cloud, emb, index, 4, my_r, my_t, 1,
                                                    self.num_points)
            self.ycb_dataset.update_transformation(my_r, my_t)

            # randomly sample points from object mesh and transform it with my_r, my_t
            list_points = [i for i in range(0, len(obj_points))]
            list_points = random.sample(list_points, self.num_points)
            transformed_model_points = self.ycb_dataset.transform_points(obj_points[list_points])
            target_pxl = self.ycb_dataset.project_point_pxl(transformed_model_points)
            # ycb_dataset.visualize_item(index, target_pxl)
            self.ycb_dataset.visualize_img(color_image, obj_idx, target_pxl, cv_show = False)
            return my_r, my_t

    def render(self, depth_image, color_image, obj_masks_all):
        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha = 0.03), cv2.COLORMAP_JET)
        # obj_semantic_colormap = cv2.applyColorMap(obj_semantic.astype(np.uint8), cv2.COLORMAP_JET)
        obj_mask_colormap = cv2.applyColorMap(cv2.convertScaleAbs(obj_masks_all * 10), cv2.COLORMAP_JET)
        # obj_mask_colormap = cv2.applyColorMap(obj_mask, cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap, obj_mask_colormap))
        # images = np.hstack((depth_image, obj_semantic))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)

    def stream_and_process(self):
        # color_image, depth_image = next(self.streamer)
        # while True:
        #     color_image, depth_image = self.stream_rgbd_()
        for rgbd in self.streamer:  # streaming with a generator
        #     color_image, depth_image = next(self.streamer)
            color_image, depth_image = rgbd[0], rgbd[1]

            rgb = np.transpose(color_image[:, :, ::-1], (2, 0, 1))    # convert to rgb and channel first
            obj_masks, segmented_bboxes, obj_masks_all = self.segment_objects(rgb, color_image, self.obj_idx)
            # start = time.time()
            for idx, obj_points in zip(self.obj_idx, self.model_points):
                try:
                    pred_r, pred_t = self.object_pose_estimate(idx, obj_points, rgb, depth_image, obj_masks, segmented_bboxes,
                                                               color_image)
                except Exception as e:
                    print(e)
                    pass
            # end = time.time()
            # print('computation time {}s'.format(end - start))
            self.render(depth_image, color_image, obj_masks_all)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break


def main():
    visualize_with_o3d = False
    num_points = 500

    # ycb_dataset = PoseDataset_ycb('test', num_points, False, opt.dataset_root, 0.0, True)
    ycb_dataset = PoseYCBDataset_visualize('test', num_points, False, opt.dataset_root, 0.0, True)
    list_obj = ycb_dataset.list_obj
    num_objects = len(list_obj)

    # Create models (segmentation and pose estimation) and load trained checkpoints
    segmenter = segnet()
    segmenter = segmenter.cuda()

    estimator = PoseNet(num_points=num_points, num_obj=num_objects)
    estimator.cuda()
    refiner = PoseRefineNet(num_points=num_points, num_obj=num_objects)
    refiner.cuda()
    estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.pose_model_save_path, opt.pose_model)))
    refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.pose_model_save_path, opt.refine_model)))
    estimator.eval()
    refiner.eval()

    # rgb_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    list_obj[0] = 'background' # SegNet output has dimension of num_objects + background
    rgb_norm = ycb_dataset.norm

    if opt.segnet_model != '':
        checkpoint = torch.load('{0}/{1}'.format(opt.segnet_model_save_path, opt.segnet_model))
        segmenter.load_state_dict(checkpoint)
    segmenter.eval()

    obj_name = 'mug'
    obj_idx = find_idx_with_name(list_obj, obj_name)

    print('object idx {}'.format(obj_idx))

    model_points = ycb_dataset.cld[obj_idx]

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, opt.img_w, opt.img_h, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, opt.img_w, opt.img_h, rs.format.bgr8, 30)     # color image streamed in bgr format

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 2 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # point cloud and points
    pc = rs.pointcloud()
    # points = rs.points()

    if visualize_with_o3d:
        # open3d visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        pcd = o3d.geometry.PointCloud()
        frame_count = 0

    rgb_colors = generate_colors(len(list_obj))
    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            cam_cx, cam_cy, cam_fx, cam_fy = get_intrinsic_info(color_frame)
            ycb_dataset.update_cam_info([cam_cx, cam_cy, cam_fx, cam_fy])

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            # points = pc.calculate(aligned_depth_frame)

            # obtain rgb and depth image from the buffer
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())     # in bgr format
            rgb = np.transpose(color_image[:,:,::-1], (2, 0, 1))    # convert to rgb and channel first
            rgb_ = rgb_norm(torch.from_numpy(rgb.astype(np.float32)))    # normalize and convert to torch tensor
            rgb_ = Variable(rgb_.unsqueeze(0)).cuda()     # expand dimension as fake batch size
            semantic = segmenter(rgb_)
            semantic = semantic[0].cpu().detach().numpy()   # array of (nb_objs, h, w)
            semantic_img = np.argmax(semantic, axis=0)  # convert each pixel as the index of object with highest score

            segmented_bboxes = {}
            obj_masks = {}
            obj_masks_new = np.zeros_like(semantic_img)
            # for i in range(1, len(list_obj)):
            for i in range(1, len(list_obj)):
                mask_idx = ma.masked_equal(semantic_img, i)
                color = rgb_colors[i]
                # calculate bbox and draw them
                # obj_mask is equivalent to the mask_label in dataset
                # r_min, r_max, c_min, c_max, obj_mask = bbox_obj_mask(mask_idx, list_obj[i], None, color)
                r_min, r_max, c_min, c_max, obj_mask = bbox_obj_mask(mask_idx, list_obj[i], color_image, color)
                segmented_bboxes[i] = [r_min, r_max, c_min, c_max]
                obj_masks[i] = obj_mask
                obj_masks_new += obj_mask//255*i

            obj_semantic = semantic[obj_idx]

            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image)) # depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
            # bg_removed = np.where((depth_image_3d > clipping_distance), grey_color, color_image)

            r_min, r_max, c_min, c_max = segmented_bboxes[obj_idx]
            cropped_depth_image = depth_image[r_min:r_max, c_min:c_max]
            cropped_color_image = color_image[r_min:r_max, c_min:c_max]
            obj_mask = obj_masks[obj_idx]
            choose = obj_mask[r_min:r_max, c_min:c_max].flatten().nonzero()[0]
            if len(choose) > 0:
                if len(choose) > num_points:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:num_points] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                else:
                    choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

                depth_masked = depth_image[r_min:r_max, c_min:c_max].flatten()[choose][:, np.newaxis].astype(np.float32)
                xmap_masked = xmap[r_min:r_max, c_min:c_max].flatten()[choose][:, np.newaxis].astype(np.float32)
                ymap_masked = ymap[r_min:r_max, c_min:c_max].flatten()[choose][:, np.newaxis].astype(np.float32)
                choose = np.array([choose])

                # pt2 = depth_masked / depth_scale    # cam_scale
                pt2 = depth_masked * depth_scale  # cam_scale
                pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
                pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
                cloud = np.concatenate((pt0, pt1, pt2), axis=1)

                # img_masked = np.array(img)[:, :, :3]
                # img_masked = np.transpose(img_masked, (2, 0, 1))
                img_masked = rgb[:, r_min:r_max, c_min:c_max]

                cloud = torch.from_numpy(cloud.astype(np.float32))
                choose = torch.LongTensor(choose.astype(np.int32))
                img_masked = rgb_norm(torch.from_numpy(img_masked.astype(np.float32)))
                index = torch.LongTensor([obj_idx - 1])

                cloud = Variable(cloud).cuda()
                choose = Variable(choose).cuda()
                img_masked = Variable(img_masked).cuda()
                index = Variable(index).cuda()

                cloud = cloud.view(1, num_points, 3)
                img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

                pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
                pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
                pred_c = pred_c.view(1, num_points)
                how_max, which_max = torch.max(pred_c, 1)
                pred_t = pred_t.view(1 * num_points, 1, 3)

                # get the rotation and translation of the most confident predicted point from the cloud set
                my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                my_t = (cloud.view(1 * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                my_pred = np.append(my_r, my_t)

                _, my_r, my_t = iterative_points_refine(refiner, cloud, emb, index, 4, my_r, my_t, 1,
                                                        num_points)
                ycb_dataset.update_transformation(my_r, my_t)

                # randomly sample points from object mesh and transform it with my_r, my_t
                list_points = [i for i in range(0, len(model_points))]
                list_points = random.sample(list_points, num_points)
                transformed_model_points = ycb_dataset.transform_points(model_points[list_points])
                target_pxl = ycb_dataset.project_point_pxl(transformed_model_points)
                # ycb_dataset.visualize_item(index, target_pxl)
                ycb_dataset.visualize_img(color_image, obj_idx, target_pxl, cv_show=False)

            if visualize_with_o3d:
                # obtain point cloud with open3d and visualize it
                temp = pc_with_o3d(color_frame, depth_image, clipping_distance_in_meters, depth_scale)
                pcd.points = temp.points
                pcd.colors = temp.colors

                if frame_count == 0:
                    vis.add_geometry(pcd)

                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                frame_count += 1

            # Render images
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            obj_semantic_colormap = cv2.applyColorMap(obj_semantic.astype(np.uint8), cv2.COLORMAP_JET)
            obj_mask_colormap = cv2.applyColorMap(cv2.convertScaleAbs(obj_masks_new*10), cv2.COLORMAP_JET)
            # obj_mask_colormap = cv2.applyColorMap(obj_mask, cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap, obj_mask_colormap))
            # images = np.hstack((depth_image, obj_semantic))
            cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


if __name__ == '__main__':
    # main()
    estimator = ObjectPoseEstimate()
    try:
        estimator.stream_and_process()
    finally:
        estimator.pipeline.stop()