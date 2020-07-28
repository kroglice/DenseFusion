## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import open3d as o3d

img_w, img_h = 640, 480
cropped_w, cropped_h = 160, 160
# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, img_w, img_h, rs.format.z16, 30)
config.enable_stream(rs.stream.color, img_w, img_h, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# point cloud and points
pc = rs.pointcloud()
points = rs.points()

vis = o3d.visualization.Visualizer ()
vis.create_window ()

pcd = o3d.geometry.PointCloud ()
flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
frame_count = 0

def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = o3d.camera.PinholeCameraIntrinsic(img_w, img_h, intrinsics.fx,
                                            intrinsics.fy, intrinsics.ppx,
                                            intrinsics.ppy)
    return out

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

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        points = pc.calculate(aligned_depth_frame)

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # 2D bbox
        rmin, rmax, cmin, cmax = img_h//2-cropped_h//2, img_h//2+cropped_h//2, img_w//2-cropped_w//2, img_w//2+cropped_w//2

        cropped_depth_image = depth_image[rmin:rmax, cmin:cmax]
        cropped_bg_removed = bg_removed[rmin:rmax, cmin:cmax]

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
        pcd.points = temp.points
        pcd.colors = temp.colors

        if frame_count == 0:
            vis.add_geometry(pcd)
        #
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        frame_count += 1

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(cropped_depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((cropped_bg_removed, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()