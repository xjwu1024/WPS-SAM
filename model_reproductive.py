import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
import torch
from scipy.optimize import linear_sum_assignment
import numpy as np

from transformer import build_transformer

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def matcher(source_label, source_embedding, logits, target_embedding, cost_cls_weight, cost_embedding_weight):

    source_embedding = source_embedding.reshape(source_embedding.shape[0], -1)
    target_embedding = target_embedding.detach().reshape(target_embedding.shape[0], -1)

    # Compute the classification cost
    out_prob = logits.softmax(-1)
    cost_class = -out_prob[:, source_label] 
    
    # Compute the embedding cost
    cost_embedding = torch.cdist(target_embedding, source_embedding, p=2)  

    # Final cost matrix [M, N]
    C = (cost_cls_weight * cost_class + cost_embedding_weight * cost_embedding).cpu()

    pred_indices, gt_indices = linear_sum_assignment(C.detach()) 
    sorted_gt_indices = np.argsort(gt_indices)
    sorted_pred_indices = pred_indices[sorted_gt_indices]

    return sorted_pred_indices

class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.SAM_model = sam_model_registry[self.cfg.SAM_model.type](checkpoint=self.cfg.SAM_model.checkpoint)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # shape: 64*64→32*32
            LayerNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # shape: 32*32→16*16
            LayerNorm2d(256),
            nn.GELU()          
        )
        self.prompter = build_transformer(cfg.transformer)
        self.num_queries = cfg.num_proposals
        self.dim_embedding = 256
        self.num_classes = cfg.num_catgories
        self.pos_embed = nn.Parameter(torch.zeros(1, self.dim_embedding, 16, 16))
        self.query_embed = nn.Embedding(self.num_queries, self.dim_embedding)
        self.class_embed = nn.Linear(self.dim_embedding, self.num_classes + 1)
        self.coords_embed = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.GELU(),
            nn.Linear(in_features=512, out_features=512),
        )
        
        if self.cfg.Full_checkpoint is not None:
            with open(self.cfg.Full_checkpoint, "rb") as f:
                state_dict = torch.load(f)
            self.load_state_dict(state_dict, strict=True)

        for param in self.SAM_model.image_encoder.parameters():
            param.requires_grad = False
        for param in self.SAM_model.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.SAM_model.mask_decoder.parameters():
            param.requires_grad = False        

    def forward(self, images, batch_category_ids, batch_bboxes):
        bs, _, H, W = images.shape
        with torch.no_grad():
            batch_image_embeddings = self.SAM_model.image_encoder(images) 
        
        batch_feature_maps = self.conv_layers(batch_image_embeddings)
        hs = self.prompter(src=batch_feature_maps, mask=None, query_embed=self.query_embed.weight, pos_embed=self.pos_embed)[0][-1] 
        batch_logits = self.class_embed(hs) 
        batch_student_embeddings = self.coords_embed(hs).reshape(bs, self.num_queries, 2, 256) 
        batch_teacher_embeddings = []
        batch_pred_indices = []
        batch_pred_masks = []
        batch_teacher_masks = []
        
        for image_embeddings, category_ids, bboxes, student_embeddings, logits in zip(batch_image_embeddings, batch_category_ids, batch_bboxes, batch_student_embeddings, batch_logits):
            
            # bbox_supervised
            with torch.no_grad():
                teacher_embeddings, dense_embeddings = self.SAM_model.prompt_encoder(
                    points=None,
                    boxes=bboxes,
                    masks=None,
                ) 

            # point_supervised
            # coords = torch.cat((((bboxes[:, 0]+bboxes[:, 2])/2).unsqueeze(1), ((bboxes[:, 1]+bboxes[:, 3])/2).unsqueeze(1)), dim=1).unsqueeze(1)
            # labels = torch.ones(coords.shape[0], 1, device=coords.device, dtype=int)
            # points = [coords, labels]
            # with torch.no_grad():
            #     teacher_embeddings, dense_embeddings = self.SAM_model.prompt_encoder(
            #         points=points,
            #         boxes=None,
            #         masks=None,
            #     ) 
    
            if self.training:    
                batch_teacher_embeddings.append(teacher_embeddings)
                pred_indices = matcher(source_label=category_ids, source_embedding=teacher_embeddings, logits=logits, target_embedding=student_embeddings, cost_cls_weight=self.cfg.weight_adjust.cost_cls_weight, cost_embedding_weight=self.cfg.weight_adjust.cost_embedding_weight)
                batch_pred_indices.append(pred_indices)
                
            else:  
                _, pred_class = logits.max(-1)
                pred_indices = torch.nonzero(pred_class!=self.num_classes).squeeze(-1)
                
                if pred_indices.shape[0]>0:
                    with torch.no_grad():
                        low_res_masks, _ = self.SAM_model.mask_decoder(
                            image_embeddings=image_embeddings.unsqueeze(0),
                            image_pe=self.SAM_model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=student_embeddings[pred_indices],
                            dense_prompt_embeddings=dense_embeddings[0].unsqueeze(0).repeat(pred_indices.shape[0], 1, 1, 1),
                            multimask_output=False,
                        )
                    pred_masks = F.interpolate(
                        low_res_masks,
                        (H, W),
                        mode="bilinear",
                        align_corners=False,
                    )
                    pred_masks = torch.sigmoid(pred_masks) 
                    batch_pred_masks.append(pred_masks.squeeze(1))
                    batch_pred_indices.append(pred_indices)
            
        return batch_pred_masks, batch_teacher_embeddings, batch_teacher_masks, batch_student_embeddings, batch_logits, batch_pred_indices
