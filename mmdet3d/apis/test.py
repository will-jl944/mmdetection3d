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
from torch.utils.tensorboard import SummaryWriter


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    tb_show=False,
                    out_dir=None,
                    show_score_thr=0.15,
                    with2d=False):
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
    writer = SummaryWriter(out_dir)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for step, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if tb_show:
            for batch_id in range(len(result)):
                vertex_positions = []
                vertex_colors = []
                bboxes = []
                if isinstance(data['points'][0], DC):
                    points = data['points'][0]._data[0][batch_id].numpy()
                elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                    points = data['points'][0][batch_id].cpu().numpy()
                else:
                    ValueError(f"Unsupported data type {type(data['points'][0])} "
                               f'for visualization!')
                if isinstance(data['img_metas'][0], DC):
                    pts_filename = data['img_metas'][0]._data[0][batch_id][
                        'pts_filename']
                    box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                        'box_mode_3d']
                elif mmcv.is_list_of(data['img_metas'][0], dict):
                    pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                    box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
                else:
                    ValueError(
                        f"Unsupported data type {type(data['img_metas'][0])} "
                        f'for visualization!')
                file_name = osp.split(pts_filename)[-1].split('.')[0]
                inds = result[batch_id]['pts_bbox']['scores_3d'] > show_score_thr
                pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d'][inds]
                pred_labels = result[batch_id]['pts_bbox']['labels_3d'][inds]
                pred_scores = result[batch_id]['pts_bbox']['scores_3d'][inds]
                pred_bboxes = torch.cat((pred_bboxes, pred_scores), dim=1)

                # for now we convert points and bbox into depth mode
                if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                      == Box3DMode.LIDAR):
                    points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                       Coord3DMode.DEPTH)
                    pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                    Box3DMode.DEPTH)
                elif box_mode_3d != Box3DMode.DEPTH:
                    ValueError(
                        f'Unsupported box_mode_3d {box_mode_3d} for convertion!')

                pred_bboxes = pred_bboxes.tensor.cpu().numpy()

                point_color = (0.5, 0.5, 0.5)
                points = points.copy()
                vertex_positions.append(points[:, :3])

                points_colors = np.tile(
                    np.array(point_color), (points.shape[0], 1))

                palette = np.random.randint(
                    0, 255, size=(pred_labels.max() + 1, 3))
                lut = o3d.ml.vis.LabelLUT()
                labelDict = {}
                for j in range(len(pred_labels)):
                    label = int(pred_labels[j].numpy())
                    score = pred_scores[j].numpy()
                    if labelDict.get(label) is None:
                        labelDict[label] = []
                    labelDict[label].append(pred_bboxes[j])
                for k in labelDict:
                    bbox3d = np.array(labelDict[k])
                    bbox_color = palette[k]
                    # lut.add_label(dataset.CLASSES[k], k, bbox_color/255)
                    lut.add_label(dataset.CLASSES[k], k)
                    points_in_box_color = palette[k]
                    bbox3d = bbox3d.copy()
                    in_box_color = np.asarray(points_in_box_color)
                    for b in range(len(bbox3d)):
                        score = bbox3d[b, -1]
                        center = bbox3d[b, 0:3]
                        size = bbox3d[b, 3:6]
                        yaw = np.zeros(3)
                        rot_axis = 2
                        yaw[rot_axis] = -bbox3d[b, 6] + np.pi / 2
                        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
                        front = rot_mat[:, 0]
                        up = rot_mat[:, 2]
                        left = rot_mat[:, 1]

                        center[rot_axis] += size[rot_axis] / 2  # bottom center to gravity center

                        box3d = geometry.OrientedBoundingBox(
                            center, rot_mat, size)

                        # change the color of points which are in box
                        indices = box3d.get_point_indices_within_bounding_box(
                            o3d.utility.Vector3dVector(points[:, :3]))
                        points_colors[indices] = in_box_color

                        o3dbox = o3d.ml.vis.BoundingBox3D(
                            center=center,
                            front=front,
                            up=up,
                            left=left,
                            size=size[[0, 2, 1]],
                            confidence=score,
                            label_class=k
                        )
                        bboxes.append(o3dbox)

                vertex_colors.append(points_colors)
                log_dict = {
                    'vertex_positions': np.array(vertex_positions)[0],
                    'vertex_colors': np.array(vertex_colors)[0],
                }

                writer.add_3d('input_point_cloud', log_dict, step=step)
                o3d.ml.vis.BoundingBox3D.create_lines(bboxes, lut=lut)
                writer.add_3d(
                    'prediction', {'bboxes': np.array(bboxes)}, step=step)

        elif show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            model.module.show_results(
                data,
                result,
                out_dir=out_dir,
                show=show,
                score_thr=show_score_thr,
                with2d=with2d)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
