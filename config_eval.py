from box import Box

config = {
    "num_devices": 4,
    "batch_size": 2,
    "num_workers": 4,
    "num_epochs": 150,
    "eval_interval": 1,
    "out_dir": "/path/to/out_dir/eval",
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [50000, 100000],
        "warmup_steps": 250,
    },
    "transformer":{
        "hidden_dim": 256,
        "dropout": 0.0,
        "nheads": 8,
        "dim_feedforward": 1024,
        "enc_layers": 6,
        "dec_layers": 6,
        "pre_norm": False,
    },
    "SAM_model": {   
        "type": 'vit_b',
        "checkpoint": "/path/to/sam_vit_b_01ec64.pth",
    }, 
    "Full_checkpoint": "/path/to/out_dir/last_ckpt.pth",
    "dataset": {
        "train": {
            "root_dir": '/path/to/PartImageNet/images/train/',
            "annotation_file": '/path/to/PartImageNet/annotations/train/train.json'
        },
        "val": {
            "root_dir": '/path/to/PartImageNet/images/val/',
            "annotation_file": "/path/to/PartImageNet/annotations/val/val.json"
        }
    },
    "weight_adjust": {
        "loss_cls_weight": 5,
        "loss_embedding_weight": 20,
        "cost_cls_weight": 10,
        "cost_embedding_weight": 1,
    },
    "num_proposals": 25,
    "num_catgories": 40,
}

cfg = Box(config)
