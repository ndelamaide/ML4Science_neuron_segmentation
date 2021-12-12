# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Youngwan Lee (ETRI), 2020. All Rights Reserved.
import logging
import os
from collections import OrderedDict
import torch
import pdb

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    # CityscapesInstanceEvaluator,
    # CityscapesSemSegEvaluator,
    # COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from centermask.evaluation import (
    COCOEvaluator,
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from centermask.config import get_cfg


class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader` method.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

IMAGE_DIR = "/home/james/LIVECell/images/sliced/unlabelled/all/200 2_0106_trans_0"
ANNOTATIONS_DIR = "/home/james/LIVECell/images/sliced/labelled/all/200 2_0106_trans_0/annotations.json"

def main(args):
    cfg = setup(args)
    from detectron2.data.datasets import register_coco_instances
    #from detectron2.data.catalog import DatasetCatalog
    #register_coco_instances("TEST", {}, "/home/james/LIVECell/images/livecell_test_images/test_a.json", "/home/james/LIVECell/images/livecell_test_images")
    #register_coco_instances("TEST", {}, "/home/james/LIVECell/images/data/empty/empty.json", "/home/james/LIVECell/images/unlabelled_data/empty")
    #ann = os.path.join(cfg.ANNOT_DIR, "annotations.json")
    #image = cfg.IMAGE_DIR
    register_coco_instances("TEST", {}, ANNOTATIONS_DIR, IMAGE_DIR)
    register_coco_instances("TRAIN", {}, ANNOTATIONS_DIR, IMAGE_DIR)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        # here requires_grad == true
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        #predictor = DefaultPredictor(self.cfg)
        #outputs = predictor(im)
        #v = Visualizer(im[:, :, ::-1],
                #metadata=train_metadata,
                #scale=0.8
                 #)
        #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #cv2_imshow(out.get_image()[:, :, ::-1])
        #pred = predictor()
        res = Trainer.test(cfg, model) #Test on all images
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
        #DatasetCatalog.remove("TEST")

        """
        If you'd like to do anything fancier than the standard training logic,
        consider writing your own training loop or subclassing the trainer.
        """

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    #trainer.model.parameters() are all s'Non-existent config key: IMAGE_DIR'et to TRUE => modify only that last layer has true grad
    #for param in trainer.model.parameters() :
        #print(param, param.requires_grad)
    #    param.requires_grad = False
    #number = 0
    #for children in trainer.model.children() :
    #    number+=1
    #    print(children)
        #if number==3 :
        #    print(children)
        #    for grandchildren in children.children():
        #        print("DONC")
        #        print(grandchildren)
    #print("NUMBER", number)
    #print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    #print(trainer.model.backbone)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
