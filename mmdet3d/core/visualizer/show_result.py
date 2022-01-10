# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import trimesh
from os import path as osp
from pyquaternion import Quaternion
from mmdet3d.core import Box3DMode
from .image_vis import (draw_camera_bbox3d_on_img, draw_depth_bbox3d_on_img,
                        draw_lidar_bbox3d_on_img)
import cv2


def velo2img(point, cam2velo, cam2img, distort=False, dist_coef=None):
    """
    Project a 3D point in Lidar coordinate on 2D camera image.
    """
    # velo -> cam
    Tx, Ty, Tz, Rx, Ry, Rz, Rw = cam2velo
    point = Quaternion(a=Rw, b=Rx, c=Ry, d=Rz).inverse.rotate(point - np.asarray([Tx, Ty, Tz]))
    point = point / point[-1]
    if distort:
        # distort
        x_p, y_p = point[:2]
        k1, k2, p1, p2, k3 = dist_coef
        r_sq = x_p**2 + y_p**2
        x_p = x_p * (1+k1*r_sq+k2*r_sq**2+k3*r_sq**3) + 2*p1*x_p*y_p + p2*(r_sq+2*x_p**2)
        y_p = y_p * (1+k1*r_sq+k2*r_sq**2+k3*r_sq**3) + p1*(r_sq+2*y_p**2) + 2*p2*x_p*y_p
        point = np.asarray([x_p, y_p, 1])
    # cam -> image
    point = cam2img.reshape(3, 3) @ point
    return point[:2]


def draw_3dbox(image, coords, color=(0, 0, 255), thickness=1):
    image = cv2.imread(image)
    im_h, im_w, _ = image.shape
    for obj in coords:
        rear_bottom_left, rear_up_left, rear_up_right, rear_bottom_right, \
        front_bottom_left, front_up_left, front_up_right, front_bottom_right = obj
        if 0 <= rear_up_left[0] < im_w and 0 <= rear_bottom_right[0] < im_w \
                and 0 <= rear_up_left[1] < im_h and 0 <= rear_bottom_right[1] < im_h\
                and 0 <= front_up_left[0] < im_w and 0 <= front_bottom_right[0] < im_w \
                and 0 <= front_up_left[1] < im_h and 0 <= front_bottom_right[1] < im_h:
            cv2.line(image, rear_bottom_left, front_bottom_left, color=color, thickness=thickness)
            cv2.line(image, rear_up_left, front_up_left, color=color, thickness=thickness)
            cv2.line(image, rear_up_right, front_up_right, color=color, thickness=thickness)
            cv2.line(image, rear_bottom_right, front_bottom_right, color=color, thickness=thickness)

            cv2.line(image, rear_bottom_left, rear_up_left, color=color, thickness=thickness)
            cv2.line(image, rear_up_left, rear_up_right, color=color, thickness=thickness)
            cv2.line(image, rear_up_right, rear_bottom_right, color=color, thickness=thickness)
            cv2.line(image, rear_bottom_right, rear_bottom_left, color=color, thickness=thickness)

            cv2.line(image, front_bottom_left, front_up_left, color=color, thickness=thickness)
            cv2.line(image, front_up_left, front_up_right, color=color, thickness=thickness)
            cv2.line(image, front_up_right, front_bottom_right, color=color, thickness=thickness)
            cv2.line(image, front_bottom_right, front_bottom_left, color=color, thickness=thickness)
    return image


def _write_obj(points, out_filename):
    """Write points into ``obj`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                'v %f %f %f %d %d %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

        else:
            fout.write('v %f %f %f\n' %
                       (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()


def _write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes.

    Args:
        scene_bbox(list[ndarray] or ndarray): xyz pos of center and
            3 lengths (dx,dy,dz) and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename(str): Filename.
    """

    def heading2rotmat(heading_angle):
        rotmat = np.zeros((3, 3))
        rotmat[2, 2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    if len(scene_bbox) == 0:
        scene_bbox = np.zeros((1, 7))
    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to obj file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='obj')

    return


def show_result(points,
                gt_bboxes,
                pred_bboxes,
                out_dir,
                filename,
                show=False,
                snapshot=False,
                pred_labels=None,
                img_filename=None,
                image_shapes=None,
                image_scales=None,
                image_calibs=None,
                corners=None
                ):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_bboxes (np.ndarray): Ground truth boxes.
        pred_bboxes (np.ndarray): Predicted boxes.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        show (bool, optional): Visualize the results online. Defaults to False.
        snapshot (bool, optional): Whether to save the online results.
            Defaults to False.
        pred_labels (np.ndarray, optional): Predicted labels of boxes.
            Defaults to None.
    """
    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)
    cam_images = {}
    gt_corners = gt_bboxes.corners.cpu().numpy()
    if img_filename is not None:
        cams = ['narrow', 'obstacle', 'wide', 'left-fisheye', 'right-fisheye',
                'spherical-left-backward', 'spherical-right-backward']
        for cam_type, image, shape, scale, calib in zip(cams, img_filename, image_shapes, image_scales, image_calibs):
            cam2velo = calib['cam2velo']
            if 'dist_coef' in calib:
                distort = True
                dist_coef = calib['dist_coef']
            else:
                distort = False
                dist_coef = None
            cam2img = calib['cam2img']
            im_coords = np.apply_along_axis(velo2img, 2, corners, cam2velo, cam2img, distort, dist_coef)
            gt_im_coords = np.apply_along_axis(velo2img, 2, gt_corners, cam2velo, cam2img, distort, dist_coef)
            im = draw_3dbox(image, np.rint(im_coords).astype(int))
            im = draw_3dbox(im, np.rint(gt_im_coords).astype(int), color=(0, 255, 0))
            cam_images[cam_type] = im
            if show:
                cv2.imshow(cam_type, im)


    if show:
        from .open3d_vis import Visualizer

        vis = Visualizer(points)
        if pred_bboxes is not None:
            if pred_labels is None:
                vis.add_bboxes(bbox3d=pred_bboxes)
            else:
                palette = np.random.randint(
                    0, 255, size=(pred_labels.max() + 1, 3)) / 255
                labelDict = {}
                for j in range(len(pred_labels)):
                    i = int(pred_labels[j].numpy())
                    if labelDict.get(i) is None:
                        labelDict[i] = []
                    labelDict[i].append(pred_bboxes[j])
                for i in labelDict:
                    vis.add_bboxes(
                        bbox3d=np.array(labelDict[i]),
                        bbox_color=palette[i],
                        points_in_box_color=palette[i])

        if gt_bboxes is not None:
            vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 1))
        show_path = osp.join(result_path,
                             f'{filename}_online.png') if snapshot else None
        vis.show(show_path)

    if points is not None:
        _write_obj(points, osp.join(result_path, f'{filename}_points.obj'))

    if gt_bboxes is not None:
        # bottom center to gravity center
        gt_bboxes[..., 2] += gt_bboxes[..., 5] / 2
        # the positive direction for yaw in meshlab is clockwise
        gt_bboxes[:, 6] *= -1
        _write_oriented_bbox(gt_bboxes,
                             osp.join(result_path, f'{filename}_gt.obj'))

    if pred_bboxes is not None:
        # bottom center to gravity center
        pred_bboxes[..., 2] += pred_bboxes[..., 5] / 2
        # the positive direction for yaw in meshlab is clockwise
        pred_bboxes[:, 6] *= -1
        _write_oriented_bbox(pred_bboxes,
                             osp.join(result_path, f'{filename}_pred.obj'))

    if cam_images:
        for cam_type, im in cam_images.items():
            cv2.imwrite(osp.join(result_path, f'{filename}_{cam_type}.jpg'), im)


def show_seg_result(points,
                    gt_seg,
                    pred_seg,
                    out_dir,
                    filename,
                    palette,
                    ignore_index=None,
                    show=True,
                    snapshot=False):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_seg (np.ndarray): Ground truth segmentation mask.
        pred_seg (np.ndarray): Predicted segmentation mask.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        palette (np.ndarray): Mapping between class labels and colors.
        ignore_index (int, optional): The label index to be ignored, e.g. \
            unannotated points. Defaults to None.
        show (bool, optional): Visualize the results online. Defaults to False.
        snapshot (bool, optional): Whether to save the online results. \
            Defaults to False.
    """
    # we need 3D coordinates to visualize segmentation mask
    if gt_seg is not None or pred_seg is not None:
        assert points is not None, \
            '3D coordinates are required for segmentation visualization'

    # filter out ignored points
    if gt_seg is not None and ignore_index is not None:
        if points is not None:
            points = points[gt_seg != ignore_index]
        if pred_seg is not None:
            pred_seg = pred_seg[gt_seg != ignore_index]
        gt_seg = gt_seg[gt_seg != ignore_index]

    if gt_seg is not None:
        gt_seg_color = palette[gt_seg]
        gt_seg_color = np.concatenate([points[:, :3], gt_seg_color], axis=1)
    if pred_seg is not None:
        pred_seg_color = palette[pred_seg]
        pred_seg_color = np.concatenate([points[:, :3], pred_seg_color],
                                        axis=1)

    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    # online visualization of segmentation mask
    # we show three masks in a row, scene_points, gt_mask, pred_mask
    if show:
        from .open3d_vis import Visualizer
        mode = 'xyzrgb' if points.shape[1] == 6 else 'xyz'
        vis = Visualizer(points, mode=mode)
        if gt_seg is not None:
            vis.add_seg_mask(gt_seg_color)
        if pred_seg is not None:
            vis.add_seg_mask(pred_seg_color)
        show_path = osp.join(result_path,
                             f'{filename}_online.png') if snapshot else None
        vis.show(show_path)

    if points is not None:
        _write_obj(points, osp.join(result_path, f'{filename}_points.obj'))

    if gt_seg is not None:
        _write_obj(gt_seg_color, osp.join(result_path, f'{filename}_gt.obj'))

    if pred_seg is not None:
        _write_obj(pred_seg_color, osp.join(result_path,
                                            f'{filename}_pred.obj'))


def show_multi_modality_result(img,
                               gt_bboxes,
                               pred_bboxes,
                               proj_mat,
                               out_dir,
                               filename,
                               box_mode='lidar',
                               img_metas=None,
                               show=True,
                               gt_bbox_color=(61, 102, 255),
                               pred_bbox_color=(241, 101, 72)):
    """Convert multi-modality detection results into 2D results.

    Project the predicted 3D bbox to 2D image plane and visualize them.

    Args:
        img (np.ndarray): The numpy array of image in cv2 fashion.
        gt_bboxes (:obj:`BaseInstance3DBoxes`): Ground truth boxes.
        pred_bboxes (:obj:`BaseInstance3DBoxes`): Predicted boxes.
        proj_mat (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        out_dir (str): Path of output directory.
        filename (str): Filename of the current frame.
        box_mode (str): Coordinate system the boxes are in. Should be one of
           'depth', 'lidar' and 'camera'. Defaults to 'lidar'.
        img_metas (dict): Used in projecting depth bbox.
        show (bool): Visualize the results online. Defaults to False.
        gt_bbox_color (str or tuple(int)): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
        pred_bbox_color (str or tuple(int)): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
    """
    if box_mode == 'depth':
        draw_bbox = draw_depth_bbox3d_on_img
    elif box_mode == 'lidar':
        draw_bbox = draw_lidar_bbox3d_on_img
    elif box_mode == 'camera':
        draw_bbox = draw_camera_bbox3d_on_img
    else:
        raise NotImplementedError(f'unsupported box mode {box_mode}')

    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    if show:
        show_img = img.copy()
        if gt_bboxes is not None:
            show_img = draw_bbox(
                gt_bboxes, show_img, proj_mat, img_metas, color=gt_bbox_color)
        if pred_bboxes is not None:
            show_img = draw_bbox(
                pred_bboxes,
                show_img,
                proj_mat,
                img_metas,
                color=pred_bbox_color)
        mmcv.imshow(show_img, win_name='project_bbox3d_img', wait_time=0)

    if img is not None:
        mmcv.imwrite(img, osp.join(result_path, f'{filename}_img.png'))

    if gt_bboxes is not None:
        gt_img = draw_bbox(
            gt_bboxes, img, proj_mat, img_metas, color=gt_bbox_color)
        mmcv.imwrite(gt_img, osp.join(result_path, f'{filename}_gt.png'))

    if pred_bboxes is not None:
        pred_img = draw_bbox(
            pred_bboxes, img, proj_mat, img_metas, color=pred_bbox_color)
        mmcv.imwrite(pred_img, osp.join(result_path, f'{filename}_pred.png'))
