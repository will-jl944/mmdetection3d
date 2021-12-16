import numpy as np
from os import path as osp
import tempfile
from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet.datasets import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose


@DATASETS.register_module()
class AcgDataset(Custom3DDataset):
    """
    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list): The range of point cloud used to filter
            invalid predicted boxes. Default: [0, -40, -3, 70.4, 40, 0.0].
    """
    CLASSES = ('barrier', 'bus', 'car', 'emergencyvehicle',
               'trafficcone', 'trailer', 'truck', 'van1', 'van2')

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 pcd_limit_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        self.split = split
        self.root_split = osp.join(self.data_root, split)
        assert self.modality is not None
        self.pcd_limit_range = pcd_limit_range
        self.pts_prefix = pts_prefix

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - img_prefix (str | None): Prefix of image files.
                - img_info (dict): Image info.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']

        pts_filename = info['point_cloud']['velodyne_path']
        input_dict = dict(sample_idx=sample_idx, pts_filename=pts_filename)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]

        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']
        yaws = annos['yaw']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate(
            [loc, dims, yaws[..., np.newaxis]], axis=1).astype(np.float32)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        # gt_bboxes = annos['bbox']

        selected = self.drop_arrays_by_name(gt_names, ['ghost'])
        # gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]

        gt_labels_3d = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d).astype(np.int64)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names)
        return anns_results

    def drop_arrays_by_name(self, gt_names, used_classes):
        """Drop irrelevant ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be dropped.
        """
        inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def keep_arrays_by_name(self, gt_names, used_classes):
        """Keep useful ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be keeped.
        """
        inds = [i for i, x in enumerate(gt_names) if x in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def remove_dontcare(self, ann_info):
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos. The ``'DontCare'``
                annotations will be removed according to ann_file['name'].

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(ann_info['name']) if x != 'ghost'
        ]
        for key in ann_info.keys():
            img_filtered_annotations[key] = (
                ann_info[key][relevant_annotation_indices])
        return img_filtered_annotations

    # def format_results(self, results, jsonfile_prefix=None):
    #     """Format the results to json (standard format for COCO evaluation).
    #
    #     Args:
    #         results (list[dict]): Testing results of the dataset.
    #         jsonfile_prefix (str | None): The prefix of json files. It includes
    #             the file path and the prefix of filename, e.g., "a/b/prefix".
    #             If not specified, a temp file will be created. Default: None.
    #
    #     Returns:
    #         tuple: Returns (result_files, tmp_dir), where `result_files` is a \
    #             dict containing the json filepaths, `tmp_dir` is the temporal \
    #             directory created for saving json files when \
    #             `jsonfile_prefix` is not specified.
    #     """
    #     assert isinstance(results, list), 'results must be a list'
    #     assert len(results) == len(self), (
    #         'The length of results is not equal to the dataset len: {} != {}'.format(len(results), len(self)))
    #
    #     if jsonfile_prefix is None:
    #         tmp_dir = tempfile.TemporaryDirectory()
    #         jsonfile_prefix = osp.join(tmp_dir.name, 'results')
    #     else:
    #         tmp_dir = None
    #
    #     # currently the output prediction results could be in two formats
    #     # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
    #     # 2. list of dict('pts_bbox' or 'img_bbox':
    #     #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
    #     # this is a workaround to enable evaluation of both formats on nuScenes
    #     # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
    #     if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
    #         result_files = self._format_bbox(results, jsonfile_prefix)
    #     else:
    #         # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
    #         result_files = dict()
    #         for name in results[0]:
    #             print(f'\nFormating bboxes of {name}')
    #             results_ = [out[name] for out in results]
    #             tmp_file_ = osp.join(jsonfile_prefix, name)
    #             result_files.update(
    #                 {name: self._format_bbox(results_, tmp_file_)})
    #     return result_files, tmp_dir

    # def evaluate(self,
    #              results,
    #              metric='bbox',
    #              logger=None,
    #              jsonfile_prefix=None,
    #              result_names=['pts_bbox'],
    #              show=False,
    #              out_dir=None,
    #              pipeline=None):
    #     """Evaluation in nuScenes protocol.
    #
    #     Args:
    #         results (list[dict]): Testing results of the dataset.
    #         metric (str | list[str]): Metrics to be evaluated.
    #         logger (logging.Logger | str | None): Logger used for printing
    #             related information during evaluation. Default: None.
    #         jsonfile_prefix (str | None): The prefix of json files. It includes
    #             the file path and the prefix of filename, e.g., "a/b/prefix".
    #             If not specified, a temp file will be created. Default: None.
    #         show (bool): Whether to visualize.
    #             Default: False.
    #         out_dir (str): Path to save the visualization results.
    #             Default: None.
    #         pipeline (list[dict], optional): raw data loading for showing.
    #             Default: None.
    #
    #     Returns:
    #         dict[str, float]: Results of each evaluation metric.
    #     """
    #     result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
    #
    #     if isinstance(result_files, dict):
    #         results_dict = dict()
    #         for name in result_names:
    #             print('Evaluating bboxes of {}'.format(name))
    #             ret_dict = self._evaluate_single(result_files[name])
    #         results_dict.update(ret_dict)
    #     elif isinstance(result_files, str):
    #         results_dict = self._evaluate_single(result_files)
    #
    #     if tmp_dir is not None:
    #         tmp_dir.cleanup()
    #
    #     if show:
    #         self.show(results, out_dir, pipeline=pipeline)
    #     return results_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=3,
                use_dim=3,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['point_cloud']['velodyne_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, img_metas, img = self._extract_data(
                i, pipeline, ['points', 'img_metas', 'img'])
            points = points.numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)
