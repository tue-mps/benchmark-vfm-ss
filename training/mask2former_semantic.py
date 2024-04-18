import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR

from training.mask2former_loss import Mask2formerLoss
from training.lightning_module import LightningModule


class Mask2formerSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_metrics: int,
        num_classes: int,
        ignore_idx: int,
        lr: float = 1e-4,
        lr_multiplier_encoder: float = 0.1,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        freeze_encoder: bool = False,
        poly_lr_decay_power: float = 0.9,
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
    ):
        super().__init__(
            img_size=img_size,
            freeze_encoder=freeze_encoder,
            network=network,
            weight_decay=weight_decay,
            lr=lr,
            lr_multiplier_encoder=lr_multiplier_encoder,
        )

        self.save_hyperparameters()

        self.ignore_idx = ignore_idx
        self.poly_lr_decay_power = poly_lr_decay_power

        self.criterion = Mask2formerLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
        )

        self.init_metrics_semantic(num_classes, ignore_idx, num_metrics)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        mask_logits_per_layer, class_logits_per_layer = self(imgs)

        losses_all_layers = {}
        for i, (mask_logits, class_logits) in enumerate(
            zip(mask_logits_per_layer, class_logits_per_layer)
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )
            losses = {f"{key}_{i}": value for key, value in losses.items()}
            losses_all_layers |= losses

        return self.criterion.loss_total(losses_all_layers, self.log)

    def eval_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=None,
        log_prefix=None,
        is_notebook=False,
    ):
        imgs, targets = batch

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        mask_logits, class_logits = self(crops)
        mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
        crop_logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)
        logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

        if is_notebook:
            return logits

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        self.update_metrics(logits, targets, dataloader_idx)

        if batch_idx == 0:
            name = f"{log_prefix}_{dataloader_idx}_pred_{batch_idx}"
            plot = self.plot_semantic(
                imgs[0],
                targets[0],
                logits=logits[0],
            )
            self.trainer.logger.experiment.log({name: [wandb.Image(plot)]})  # type: ignore

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()

        lr_scheduler = {
            "scheduler": PolynomialLR(
                optimizer,
                int(self.trainer.estimated_stepping_batches),
                self.poly_lr_decay_power,
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
