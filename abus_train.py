"""
SegMamba Training Script for ABUS 3D Ultrasound Binary Segmentation.

Key adaptations from the BraTS original:
    - in_chans  = 1   (single-channel ultrasound, not 4-modality MRI)
    - out_chans = 2   (background + tumour, not 4-class BraTS)
    - Loss      = DiceLoss + CrossEntropyLoss  (handles class imbalance)
    - No label conversion  (binary mask, not TC/WT/ET)
    - Pre-split data dirs  (train / val / test are separate)

Usage:
    python abus_train.py

Prerequisites:
    Run  abus_preprocessing.py  first to create
        ./data/abus/{train,val,test}/*.npz  +  *.pkl
"""

import os
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from monai.losses.dice import DiceLoss
from monai.utils import set_determinism

from light_training.dataloading.dataset import get_train_val_test_loader_seperate
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last

set_determinism(123)

# ---------------------------------------------------------------------------
# Paths & hyper-parameters
# ---------------------------------------------------------------------------
data_dir_train = "./data/abus/train"
data_dir_val   = "./data/abus/val"

logdir          = "./logs/segmamba_abus"
model_save_path = os.path.join(logdir, "model")

augmentation = True          # full augmentation (spatial + intensity + mirror)
env          = "pytorch"     # single-GPU
max_epoch    = 1000
batch_size   = 2
val_every    = 2
num_gpus     = 1
device       = "cuda:0"
roi_size     = [128, 128, 128]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ABUSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu",
                 val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750,
                 training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every,
                         num_gpus, logdir, master_ip, master_port,
                         training_script)

        # --- sliding-window inferer for validation ---
        self.window_infer = SlidingWindowInferer(
            roi_size=roi_size, sw_batch_size=1, overlap=0.5)

        self.augmentation = augmentation

        # --- model ---
        from model_segmamba.segmamba import SegMamba
        self.model = SegMamba(
            in_chans=1,                         # single-channel US
            out_chans=2,                        # background + tumour
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
        )

        self.patch_size = roi_size
        self.best_mean_dice = 0.0

        # --- loss ---
        self.ce = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,           # focus on tumour class
        )

        # --- optimiser ---
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=1e-2,
            weight_decay=3e-5,
            momentum=0.99,
            nesterov=True,
        )
        self.scheduler_type = "poly"

        # data-loader workers (reduce if OOM)
        self.train_process = 8

    # -----------------------------------------------------------------------
    # data helpers
    # -----------------------------------------------------------------------
    def get_input(self, batch):
        image = batch["data"]                   # (B, 1, D, H, W)
        label = batch["seg"]                    # (B, 1, D, H, W)
        label = label[:, 0].long()              # (B, D, H, W)  values {0, 1}
        return image, label

    # -----------------------------------------------------------------------
    # training
    # -----------------------------------------------------------------------
    def training_step(self, batch):
        image, label = self.get_input(batch)
        pred = self.model(image)                # (B, 2, D, H, W)

        ce_loss   = self.ce(pred, label)
        dice_loss = self.dice_loss(pred, label.unsqueeze(1))
        loss = ce_loss + dice_loss

        self.log("training_loss", loss, step=self.global_step)
        self.log("ce_loss",   ce_loss,   step=self.global_step)
        self.log("dice_loss", dice_loss, step=self.global_step)
        return loss

    # -----------------------------------------------------------------------
    # validation
    # -----------------------------------------------------------------------
    @staticmethod
    def cal_metric(gt, pred):
        """Binary Dice between numpy arrays."""
        if pred.sum() > 0 and gt.sum() > 0:
            return dice(pred, gt)
        elif gt.sum() == 0 and pred.sum() == 0:
            return 1.0
        else:
            return 0.0

    def validation_step(self, batch):
        image, label = self.get_input(batch)
        output = self.model(image)              # (B, 2, D, H, W)
        output = output.argmax(dim=1)           # (B, D, H, W)

        output_np = output.cpu().numpy()
        target_np = label.cpu().numpy()

        d = self.cal_metric(target_np, output_np)
        return d

    def validation_end(self, val_outputs):
        dices = val_outputs
        mean_dice = dices.mean().item() if hasattr(dices, 'mean') else float(
            np.mean(dices))

        self.log("mean_dice", mean_dice, step=self.epoch)
        print(f"  [epoch {self.epoch}]  mean_dice = {mean_dice:.4f}")

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(
                self.model,
                os.path.join(model_save_path,
                             f"best_model_{mean_dice:.4f}.pt"),
                delete_symbol="best_model")

        save_new_model_and_delete_last(
            self.model,
            os.path.join(model_save_path,
                         f"final_model_{mean_dice:.4f}.pt"),
            delete_symbol="final_model")

        if (self.epoch + 1) % 100 == 0:
            torch.save(
                self.model.state_dict(),
                os.path.join(model_save_path,
                             f"tmp_model_ep{self.epoch}_{mean_dice:.4f}.pt"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trainer = ABUSTrainer(
        env_type=env,
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        logdir=logdir,
        val_every=val_every,
        num_gpus=num_gpus,
        master_port=17760,
        training_script=__file__,
    )

    train_ds, val_ds, _ = get_train_val_test_loader_seperate(
        data_dir_train, data_dir_val)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
