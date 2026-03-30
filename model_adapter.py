import os
import subprocess
import logging
import torch
import dtlpy as dl
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from mmdet.apis import init_detector, inference_detector
from mmengine.registry import DefaultScope
import numpy as np
from PIL import Image

logger = logging.getLogger("MMDetection")


class MMDetection(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        model_name = self.model_entity.configuration.get("pose_model_name", "td-hm_hrnet-w48_8xb32-210e_coco-256x192")
        config_file = self.model_entity.configuration.get(
            "pose_config_file", "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
        )
        checkpoint_file = self.model_entity.configuration.get(
            "pose_checkpoint_file", "td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth"
        )

        if not os.path.exists(config_file) or not os.path.exists(checkpoint_file):
            logger.info("Downloading mmdet artifacts")
            download_status = subprocess.Popen(
                f"mim download mmpose --config {model_name} --dest .",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
            )
            download_status.wait()
            (out, err) = download_status.communicate()
            if download_status.returncode != 0:
                raise Exception(f"Failed to download mmdet artifacts: {err}")
            logger.info(f"MMDetection artifacts downloaded successfully, Loading Model {out}")

        logger.info("MMDetection artifacts downloaded successfully, Loading Model")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model on device {device}")
        self.pose_model = init_pose_estimator(config_file, checkpoint_file, device=device)  # or device='cuda:0'

        model_name_detector = self.model_entity.configuration.get("mmdet_model_name", "rtmdet_tiny_8xb32-300e_coco")
        config_file_detector = self.model_entity.configuration.get(
            "mmdet_config_file", "rtmdet_tiny_8xb32-300e_coco.py"
        )
        checkpoint_file_detector = self.model_entity.configuration.get(
            "mmdet_checkpoint_file", "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
        )

        if not os.path.exists(config_file_detector) or not os.path.exists(checkpoint_file_detector):
            logger.info("Downloading mmdet artifacts")
            download_status = subprocess.Popen(
                f"mim download mmdet --config {model_name_detector} --dest .",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
            )
            download_status.wait()
            (out, err) = download_status.communicate()
            if download_status.returncode != 0:
                raise Exception(f"Failed to download mmdet artifacts: {err}")
            logger.info(f"MMDetection artifacts downloaded successfully, Loading Model {out}")

        logger.info("MMDetection artifacts downloaded successfully, Loading Model")
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model on device {device}")
        self.model = init_detector(config_file_detector, checkpoint_file_detector, device=device)
        logger.info("Model Loaded Successfully")

    def prepare_item_func(self, item):
        buffer = item.download(save_locally=False)
        image = np.asarray(Image.open(buffer))
        return item, image

    def predict(self, batch, **kwargs):
        logger.info(f"Predicting on batch of {len(batch)} images")
        try:
            item, image = batch[0]
            recipe: dl.Recipe = item.dataset.recipes.list()[0]
            template_id = recipe.get_annotation_template_id(template_name="mmpose-human")
            logger.info(f"Found annotation template 'mmpose-human' with id: {template_id}")
        except Exception as e:
            logger.warning(f"Annotation template 'mmpose-human' not found, skipping pose parent annotations: {e}")
            template_id = None
        batch_annotations = list()
        for item, image in batch:
            image_annotations = dl.AnnotationCollection()
            scope = DefaultScope.get_current_instance()
            if scope is not None:
                scope._scope_name = 'mmdet'
            det_result = inference_detector(self.model, image)
            pred_instance = det_result.pred_instances.cpu().numpy()
            bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.25)]
            bboxes = bboxes[nms(bboxes, 0), :4]
            pose_results = inference_topdown(self.pose_model, image, bboxes)
            data_samples = merge_data_samples(pose_results)
            pose_annotations = data_samples.pred_instances.keypoints
            labels = data_samples.metainfo.get("flip_indices", list())
            confidence_samples = data_samples.pred_instances.keypoint_scores
            for pose_annotation, confidence_sample in zip(pose_annotations, confidence_samples):
                if template_id is not None:
                    dl_pose_annotation = dl.Pose(label="mmpose-model", template_id=template_id)
                    parent_annotation = item.annotations.upload(
                        dl.Annotation.new(annotation_definition=dl_pose_annotation)
                    )[0]
                    logger.debug(f"Uploaded pose parent annotation: {parent_annotation.id}")
                else:
                    logger.debug("No template_id available, skipping pose parent annotation")
                    parent_annotation = None
                for point, label_id, confidence in zip(pose_annotation, labels, confidence_sample):
                    parent_id = parent_annotation.id if parent_annotation is not None else None
                    image_annotations.add(
                        parent_id=parent_id,
                        annotation_definition=dl.Point(
                            x=point[0], y=point[1], label=self.pose_model.dataset_meta.get("keypoint_id2name")[label_id]
                        ),
                        model_info={
                            "name": self.model_entity.name,
                            "model_id": self.model_entity.id,
                            "confidence": confidence,
                        },
                    )
            batch_annotations.append(image_annotations)
        return batch_annotations


if __name__ == "__main__":
    model_entity = dl.models.get(model_id="")
    model_adapter = MMDetection(model_entity=model_entity)
    _item = dl.items.get(item_id="")
    model_adapter.predict_items(items=[_item])
