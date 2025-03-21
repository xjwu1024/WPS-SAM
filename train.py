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

from config_reproductive import cfg
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
    cls_acc = AverageMeter()
    
    null_imgs = 0
    failed_imgs = 0
    matched_imgs = 0

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
                        matched_imgs = matched_imgs + 1
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
                        failed_imgs = failed_imgs + 1
                        iou = 0
                        f1 = 0
                        acc = 0
                        accs.update(acc, n=1)
                        ious.update(iou, n=1)
                        f1_scores.update(f1, n=1)

            fabric.print(f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean ACC: [{accs.avg:.4f}] -- Mean F1: {f1_scores.avg:.4f}]')
            logging.info(f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean ACC: [{accs.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean ACC: [{accs.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- matched imgs: [{matched_imgs}] -- null imgs: [{null_imgs}] -- failed imgs: [{failed_imgs}]')
    logging.info(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean ACC: [{accs.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- matched imgs: [{matched_imgs}] -- null imgs: [{null_imgs}] -- failed imgs: [{failed_imgs}]')
  
    fabric.print(f"Saving last checkpoint to {cfg.out_dir}")
    state_dict = model.state_dict()
    if fabric.global_rank == 0:
        torch.save(state_dict, os.path.join(cfg.out_dir, f"last_ckpt.pth"))
    model.train()
    
    if ious.avg > best_ious:
        fabric.print(f"Cool! Saving checkpoint to {cfg.out_dir}")
        state_dict = model.state_dict()
        if fabric.global_rank == 0:
            torch.save(state_dict, os.path.join(cfg.out_dir, f"best_ckpt.pth"))
        best_ious = ious.avg
        fabric.print(f'best performance: Mean IoU: [{ious.avg:.4f}] -- Mean ACC: [{accs.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')
        logging.info(f'best performance: Mean IoU: [{ious.avg:.4f}] -- Mean ACC: [{accs.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')
    
    logging.info('==============================next epoch=============================================')
    return best_ious

def prompter_kd(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """The prompter kd loop."""
    
    best_ious = 0
    for epoch in range(1, cfg.num_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        cls_losses = AverageMeter()
        embedding_losses = AverageMeter()
        Acc = AverageMeter()
        Recall = AverageMeter()
        end = time.time()

        for iter, data in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            images, bboxes, _, batch_category_ids = data
            batch_size = images.size(0)
            
            _, batch_teacher_embeddings, _, batch_student_embeddings, batch_logits, batch_pred_indices = model(images, batch_category_ids, bboxes)

            loss_ce = torch.tensor(0., device=fabric.device)
            loss_embedding = torch.tensor(0., device=fabric.device)
            acc = torch.tensor(0., device=fabric.device)
            recall = torch.tensor(0., device=fabric.device)
            
            for teacher_embeddings, student_embeddings, logits, pred_indices, category_ids in zip(batch_teacher_embeddings, batch_student_embeddings, batch_logits, batch_pred_indices, batch_category_ids):
                loss_embedding += F.smooth_l1_loss(teacher_embeddings, student_embeddings[pred_indices]) # L1_smoth_loss
                expended_labels = torch.full((logits.size(0), ), cfg.num_catgories , dtype=torch.int64, device=category_ids.device)
                expended_labels[pred_indices] = category_ids 
                loss_ce += F.cross_entropy(logits, expended_labels)
                _, pred_class = logits.max(1)
                acc += sum((pred_class==expended_labels).int())/logits.size(0)
                recall += sum(pred_class[pred_indices]==category_ids.int())/category_ids.size(0)

            loss_ce = loss_ce / batch_size
            loss_embedding = loss_embedding / batch_size
            acc = acc / batch_size
            recall = recall / batch_size
            
            loss_matcher = cfg.weight_adjust.loss_cls_weight * loss_ce + cfg.weight_adjust.loss_embedding_weight * loss_embedding
            
            optimizer.zero_grad()
            fabric.backward(loss_matcher)
            optimizer.step()
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

            cls_losses.update(loss_ce.item(), batch_size)
            embedding_losses.update(loss_embedding.item(), batch_size)
            Acc.update(acc.item(), batch_size)
            Recall.update(recall.item(), batch_size)

            fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | embedding Loss [{embedding_losses.val:.4f} ({embedding_losses.avg:.4f})]'
                         f' | cls Loss [{cls_losses.val:.4f} ({cls_losses.avg:.4f})]'
                         f' | matcher Acc [{Acc.val:.4f} ({Acc.avg:.4f})]'
                         f' | matcher recall [{Recall.val:.4f} ({Recall.avg:.4f})]'
                         )
            logging.info(f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | embedding Loss [{embedding_losses.val:.4f} ({embedding_losses.avg:.4f})]'
                         f' | cls Loss [{cls_losses.val:.4f} ({cls_losses.avg:.4f})]'
                         f' | matcher Acc [{Acc.val:.4f} ({Acc.avg:.4f})]'
                         f' | matcher recall [{Recall.val:.4f} ({Recall.avg:.4f})]'
                         )

        if epoch % cfg.eval_interval == 0:
            best_ious = validate(fabric, model, val_dataloader, epoch, best_ious) 
        fabric.print(f'best Mean IoU: [{best_ious:.4f}]')


def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)
        # return 1.0
    
    def get_parameters():
        params = []
        for name, param in model.named_parameters():
            if not name.startswith('SAM_mode'):
                params.append(param)
        return params
    
    optimizer = torch.optim.Adam(get_parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfg.opt.warmup_steps, t_total=38370)

    return optimizer, scheduler


def main(cfg: Box) -> None:

    print('ready!')
    if not os.path.exists(cfg.out_dir):
        os.mkdir(cfg.out_dir)
    
    os.system('cp config_reproductive.py '+ cfg.out_dir)
    
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
    # breakpoint()
    # total = sum([param.nelement() for param in model.parameters() if param.requires_grad]) 
    torch.backends.cudnn.benchmark = True
    prompter_kd(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    logging.shutdown()

if __name__ == "__main__":
    main(cfg)

