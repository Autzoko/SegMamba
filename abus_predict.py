"""
SegMamba Prediction Script for ABUS 3D Ultrasound Dataset.

Runs sliding-window inference with test-time augmentation (mirroring)
on preprocessed ABUS test data and saves binary segmentation masks
as NIfTI (.nii.gz) files.

Usage:
    python abus_predict.py

Prerequisites:
    1.  Run  abus_preprocessing.py   (creates ./data/abus/test/*.npz)
    2.  Run  abus_train.py           (creates model checkpoint)
    3.  Update  model_path  below to point to the trained checkpoint.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism

from light_training.dataloading.dataset import MedicalDataset
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.prediction import Predictor

set_determinism(123)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
data_dir_test = "./data/abus/test"

# >>> UPDATE THIS PATH to your trained model checkpoint <<<
model_path = "./logs/segmamba_abus/model/best_model_0.0000.pt"

save_path = "./prediction_results/segmamba_abus"

env        = "pytorch"
max_epoch  = 1
batch_size = 1
val_every  = 1
num_gpus   = 1
device     = "cuda:0"
patch_size = [128, 128, 128]


# ---------------------------------------------------------------------------
# Trainer subclass for prediction
# ---------------------------------------------------------------------------

class ABUSPredictor(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu",
                 val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750,
                 training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every,
                         num_gpus, logdir, master_ip, master_port,
                         training_script)
        self.patch_size = patch_size
        self.augmentation = False

    # -----------------------------------------------------------------------
    # model + predictor setup
    # -----------------------------------------------------------------------
    def define_model(self):
        from model_segmamba.segmamba import SegMamba
        model = SegMamba(
            in_chans=1,
            out_chans=2,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
        )

        new_sd = self.filte_state_dict(
            torch.load(model_path, map_location="cpu"))
        model.load_state_dict(new_sd)
        model.eval()

        window_infer = SlidingWindowInferer(
            roi_size=patch_size,
            sw_batch_size=1,
            overlap=0.5,
            progress=True,
            mode="gaussian",
        )

        predictor = Predictor(
            window_infer=window_infer,
            mirror_axes=[0, 1, 2],
        )

        os.makedirs(save_path, exist_ok=True)
        return model, predictor

    # -----------------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------------
    def get_input(self, batch):
        image = batch["data"]                   # (B, 1, D, H, W)
        label = batch["seg"]                    # (B, 1, D, H, W)
        label = label[:, 0].long()              # (B, D, H, W)
        properties = batch["properties"]
        return image, label, properties

    @staticmethod
    def filte_state_dict(sd):
        if "module" in sd:
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k
            new_sd[new_k] = v
        del sd
        return new_sd

    # -----------------------------------------------------------------------
    # inference per sample
    # -----------------------------------------------------------------------
    def validation_step(self, batch):
        image, label, properties = self.get_input(batch)
        model, predictor = self.define_model()

        # --- sliding-window + TTA ---
        model_output = predictor.maybe_mirror_and_predict(
            image, model, device=device)

        # --- resample to pre-resample shape ---
        model_output = predictor.predict_raw_probability(
            model_output, properties=properties)

        # --- argmax -> binary mask ---
        model_output = model_output.argmax(dim=0)[None]    # (1, D, H, W)
        pred_binary = (model_output > 0).float()           # tumour = 1

        # --- Dice on cropped volume ---
        label_np = label[0].cpu().numpy()
        pred_np  = pred_binary[0].cpu().numpy()

        if pred_np.sum() > 0 and label_np.sum() > 0:
            d = dice(pred_np, label_np)
        elif pred_np.sum() == 0 and label_np.sum() == 0:
            d = 1.0
        else:
            d = 0.0
        print(f"  Dice (cropped) = {d:.4f}")

        # --- restore full (non-cropped) volume ---
        full_output = predictor.predict_noncrop_probability(
            pred_binary, properties)

        # --- save as NIfTI ---
        case_name = properties['name']
        if isinstance(case_name, list):
            case_name = case_name[0]

        raw_spacing = properties['spacing']
        if isinstance(raw_spacing[0], torch.Tensor):
            raw_spacing = [s.item() for s in raw_spacing]
        elif isinstance(raw_spacing, tuple):
            raw_spacing = list(raw_spacing)

        predictor.save_to_nii(
            full_output,
            raw_spacing=raw_spacing,
            case_name=case_name,
            save_dir=save_path,
        )
        return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- build test dataset (with seg for evaluation) ---
    test_datalist = sorted(glob.glob(f"{data_dir_test}/*.npz"))
    if len(test_datalist) == 0:
        print(f"No test data found in {data_dir_test}. "
              "Run abus_preprocessing.py first.")
        exit(1)

    print(f"Test data: {len(test_datalist)} cases")
    test_ds = MedicalDataset(test_datalist, test=False)

    # --- predictor ---
    predictor_trainer = ABUSPredictor(
        env_type=env,
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        logdir="",
        val_every=val_every,
        num_gpus=num_gpus,
        master_port=17761,
        training_script=__file__,
    )

    v_mean, val_outputs = predictor_trainer.validation_single_gpu(test_ds)
    print(f"\n{'='*60}")
    print(f"  Mean Dice on test set = {v_mean}")
    print(f"  Predictions saved to  {save_path}/")
    print(f"{'='*60}")
