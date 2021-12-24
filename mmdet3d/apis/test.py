# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmcv.image import tensor2imgs
from mmcv.parallel import DataContainer as DC
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d)
import open3d as o3d
from open3d import geometry
from os import path as osp
import numpy as np

from open3d.visualization.tensorboard_plugin import summary
# Utility function to convert Open3D geometry to a dictionary format
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from torch.utils.tensorboard import SummaryWriter

from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    log_dir = './o3d_log'
    writer = SummaryWriter(log_dir)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for step, data in enumerate(data_loader):
        if step == 2:
            import sys
            sys.exit(1)
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        # if show:
        #     for batch_id in range(len(result)):
        #         vertex_positions = []
        #         vertex_colors = []
        #         bboxes = []
        #         if isinstance(data['points'][0], DC):
        #             points = data['points'][0]._data[0][batch_id].numpy()
        #         elif mmcv.is_list_of(data['points'][0], torch.Tensor):
        #             points = data['points'][0][batch_id].cpu().numpy()
        #         else:
        #             ValueError(f"Unsupported data type {type(data['points'][0])} "
        #                        f'for visualization!')
        #         if isinstance(data['img_metas'][0], DC):
        #             pts_filename = data['img_metas'][0]._data[0][batch_id][
        #                 'pts_filename']
        #             box_mode_3d = data['img_metas'][0]._data[0][batch_id][
        #                 'box_mode_3d']
        #         elif mmcv.is_list_of(data['img_metas'][0], dict):
        #             pts_filename = data['img_metas'][0][batch_id]['pts_filename']
        #             box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
        #         else:
        #             ValueError(
        #                 f"Unsupported data type {type(data['img_metas'][0])} "
        #                 f'for visualization!')
        #         file_name = osp.split(pts_filename)[-1].split('.')[0]
        #
        #         assert out_dir is not None, 'Expect out_dir, got none.'
        #         inds = result[batch_id]['pts_bbox']['scores_3d'] > 0.1
        #         pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d'][inds]
        #         pred_labels = result[batch_id]['pts_bbox']['labels_3d'][inds]
        #
        #         # for now we convert points and bbox into depth mode
        #         if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
        #                                               == Box3DMode.LIDAR):
        #             points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
        #                                                Coord3DMode.DEPTH)
        #             pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
        #                                             Box3DMode.DEPTH)
        #         elif box_mode_3d != Box3DMode.DEPTH:
        #             ValueError(
        #                 f'Unsupported box_mode_3d {box_mode_3d} for convertion!')
        #
        #         pred_bboxes = pred_bboxes.tensor.cpu().numpy()
        #
        #         # mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
        #         #     size=1, origin=[0, 0, 0])  # create coordinate frame
        #
        #         point_color = (0.5, 0.5, 0.5)
        #         points = points.copy()
        #         vertex_positions.append(points[:, :3])
        #
        #         points_colors = np.tile(
        #             np.array(point_color), (points.shape[0], 1))
        #
        #         palette = np.random.randint(
        #             0, 255, size=(pred_labels.max() + 1, 3)) / 256
        #         lut = o3d.ml.vis.LabelLUT()
        #         labelDict = {}
        #         for j in range(len(pred_labels)):
        #             label = int(pred_labels[j].numpy())
        #             if labelDict.get(label) is None:
        #                 labelDict[label] = []
        #             labelDict[label].append(pred_bboxes[j])
        #         for k in labelDict:
        #             bbox3d = np.array(labelDict[k])
        #             bbox_color = palette[k]
        #             lut.add_label(dataset.CLASSES[k], k, bbox_color)
        #             points_in_box_color = palette[k]
        #             bbox3d = bbox3d.copy()
        #             in_box_color = np.array(points_in_box_color)
        #             for b in range(len(bbox3d)):
        #                 center = bbox3d[b, 0:3]
        #                 size = bbox3d[b, 3:6]
        #                 yaw = np.zeros(3)
        #                 rot_axis = 2
        #                 yaw[rot_axis] = -bbox3d[b, 6]
        #                 rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
        #                 front = rot_mat[:, 1]
        #                 up = rot_mat[:, 2]
        #                 left = -rot_mat[:, 0]
        #
        #                 center[rot_axis] += size[rot_axis] / \
        #                                     2  # bottom center to gravity center
        #
        #                 box3d = geometry.OrientedBoundingBox(
        #                     center, rot_mat, size)
        #
        #                 # line_set = geometry.LineSet.create_from_oriented_bounding_box(
        #                 #     box3d)
        #                 # line_set.paint_uniform_color(bbox_color)
        #
        #                 # change the color of points which are in box
        #                 indices = box3d.get_point_indices_within_bounding_box(
        #                     o3d.utility.Vector3dVector(points[:, :3]))
        #                 points_colors[indices] = in_box_color
        #
        #                 o3box = o3d.ml.vis.BoundingBox3D(
        #                     center=center,
        #                     front=front,
        #                     up=up,
        #                     left=left,
        #                     size=size[[0, 2, 1]],
        #                     confidence=1.,
        #                     label_class=k
        #                 )
        #                 bboxes.append(o3box)
        #
        #         vertex_colors.append(points_colors)
        #         log_dict = {
        #             'vertex_positions': np.array(vertex_positions)[0],
        #             'vertex_colors': np.array(vertex_colors)[0],
        #         }
        #
        #         writer.add_3d('input_point_cloud', log_dict, step=step)
        #         o3d.ml.vis.BoundingBox3D.create_lines(bboxes, lut=lut)
        #         writer.add_3d(
        #             'prediction', {'bboxes': np.array(bboxes)}, step=step)

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(
                    data,
                    result,
                    out_dir=out_dir,
                    show=show,
                    score_thr=show_score_thr)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
