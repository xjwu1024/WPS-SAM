import os
import time

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from box import Box
from dataset import load_datasets
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger

from model_reproductive import Model
from torch.utils.data import DataLoader
from utils import AverageMeter

from config_eval import cfg
import logging  
import cv2
from scheduler import WarmupCosineSchedule

torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')

def validate(fabric: L.Fabric, model: Model, val_dataloader: DataLoader, epoch: int = 0, best_ious=0):
    model.eval()
    ious = AverageMeter()
    accs = AverageMeter()
    f1_scores = AverageMeter()
    null_imgs = 0
    failed_parts = 0
    ok_parts = 0
    
    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
        
            images, bboxes, batch_gt_masks, batch_category_ids = data
            batch_pred_masks, _, batch_teacher_masks, _, batch_logits, batch_pred_indices = model(images, batch_category_ids, bboxes)

            for logits, pred_masks, pred_indices, category_ids, gt_masks in zip(batch_logits, batch_pred_masks, batch_pred_indices, batch_category_ids, batch_gt_masks):
          
                if len(pred_masks)==0:
                    null_imgs = null_imgs + 1
                    iou = 0
                    f1 = 0
                    acc = 0
                    accs.update(acc, n=1)
                    ious.update(iou, n=1)
                    f1_scores.update(f1, n=1)
                    continue

                category_masks_dict = {}
                for category, mask in zip(category_ids, gt_masks):
                    if category.item() not in category_masks_dict:
                        category_masks_dict[category.item()] = mask
                    else:
                        category_masks_dict[category.item()] = torch.add(category_masks_dict[category.item()], mask)
              
                category_masks_dict_pred = {}
                _, out_class = logits.max(-1)
                pred_category = out_class[pred_indices]
                for category, mask in zip(pred_category, pred_masks):
                    if category.item() not in category_masks_dict_pred:
                        category_masks_dict_pred[category.item()] = mask
                    else:
                        category_masks_dict_pred[category.item()] = torch.add(category_masks_dict_pred[category.item()], mask)

                for category in category_masks_dict:

                    if category in category_masks_dict_pred:
                        ok_parts = ok_parts + 1
                        stats = smp.metrics.get_stats(
                            category_masks_dict_pred[category],
                            category_masks_dict[category].int(),
                            mode='binary',
                            threshold=0.5,
                        )
                        iou = smp.metrics.iou_score(*stats, reduction="micro-imagewise")
                        acc = smp.metrics.accuracy(*stats, reduction="micro-imagewise")
                        f1 = smp.metrics.f1_score(*stats, reduction="micro-imagewise")
                        ious.update(iou, n=1)
                        accs.update(acc, n=1)
                        f1_scores.update(f1, n=1)
                    else:  
                        failed_parts = failed_parts + 1
                        iou = 0
                        f1 = 0
                        acc = 0
                        accs.update(acc, n=1)
                        ious.update(iou, n=1)
                        f1_scores.update(f1, n=1)

            fabric.print(f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean ACC: [{accs.avg:.4f}] -- Mean F1: {f1_scores.avg:.4f}]')
            logging.info(f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean ACC: [{accs.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')
    
    fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean ACC: [{accs.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- null imgs: [{null_imgs}] -- failed parts: [{failed_parts}] -- seg imgs: [{ok_parts}]')
    logging.info(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean ACC: [{accs.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- null imgs: [{null_imgs}] -- failed parts: [{failed_parts}] -- seg imgs: [{ok_parts}]')
    
    logging.info('==============================next epoch=============================================')
    return best_ious

def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        return 1.0
    
    def get_parameters():
        params = []
        for name, param in model.named_parameters():
            if not name.startswith('SAM_mode'):
                params.append(param)
        return params
    
    optimizer = torch.optim.Adam(get_parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def main(cfg: Box) -> None:

    print('ready!')
    if not os.path.exists(cfg.out_dir):
        os.mkdir(cfg.out_dir)
    
    os.system('cp config_eval.py '+ cfg.out_dir)
    
    log_file = os.path.join(cfg.out_dir, "log.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO)

    fabric = L.Fabric(accelerator="cuda",
                      devices=cfg.num_devices,
                      strategy="ddp",
                      loggers=[TensorBoardLogger(cfg.out_dir, name="lightning-sam")])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    model = Model(cfg)

    train_data, val_data = load_datasets(cfg, model.SAM_model.image_encoder.img_size)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    validate(fabric, model, val_data, epoch=0, best_ious=0)
    logging.shutdown()

if __name__ == "__main__":
    main(cfg)

