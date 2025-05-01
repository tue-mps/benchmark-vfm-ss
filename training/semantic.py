import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import PolynomialLR, SequentialLR, LambdaLR
from torchmetrics.classification import MulticlassJaccardIndex
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import io
import numpy as np
from torchvision.transforms.v2.functional import resize
import math
from training.lightning_module import LightningModule


class Semantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        num_classes: int,
        ignore_idx: int,
        img_size: tuple[int, int],
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        poly_lr_decay_power: float = 0.9,
        lr_multiplier_encoder: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__(
            img_size,
            freeze_encoder,
            network,
            weight_decay,
            lr,
            lr_multiplier_encoder,
        )
        self.save_hyperparameters()
        self.ignore_idx = ignore_idx
        self.poly_lr_decay_power = poly_lr_decay_power
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
        self.metric = MulticlassJaccardIndex(
            num_classes=num_classes,
            validate_args=False,
            ignore_index=ignore_idx,
            average=None,
        )
        self._train_core_compiled = None

    def _train_core(self, imgs: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        logits = self.forward(imgs)
        logits = F.interpolate(logits, self.img_size, mode="bilinear")
        return self.criterion(logits, tgt)

    def setup(self, stage: str):
        if stage == "fit" and self._train_core_compiled is None:
            self._train_core_compiled = self.compile_train_core(self._train_core)

    def to_per_pixel(self, targets: list[dict]) -> list[torch.Tensor]:
        per_pixel = []
        for t in targets:
            mask = torch.full(
                t["masks"].shape[-2:],
                self.ignore_idx,
                dtype=t["labels"].dtype,
                device=t["labels"].device,
            )
            for i, m in enumerate(t["masks"]):
                mask[m] = t["labels"][i]
            per_pixel.append(mask)
        return per_pixel

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        imgs = torch.stack(imgs) if isinstance(imgs, list) else imgs
        tgt = torch.stack(self.to_per_pixel(targets)).long()
        loss = self._train_core_compiled(imgs, tgt)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def plot(
        self,
        img,
        target,
        logits,
        cmap="tab20",
    ):
        fig, axes = plt.subplots(1, 3, figsize=[15, 5], sharex=True, sharey=True)
        axes[0].imshow(img.cpu().numpy().transpose(1, 2, 0))
        axes[0].axis("off")
        target = target.cpu().numpy()
        unique_classes = np.unique(target)
        preds = torch.argmax(logits, dim=0).cpu().numpy()
        unique_classes = np.unique(np.concatenate((unique_classes, np.unique(preds))))
        num_classes = len(unique_classes)
        colors = plt.get_cmap(cmap, num_classes)(np.linspace(0, 1, num_classes))  # type: ignore
        if self.ignore_idx in unique_classes:
            colors[unique_classes == self.ignore_idx] = [0, 0, 0, 1]  # type: ignore
        custom_cmap = mcolors.ListedColormap(colors)  # type: ignore
        norm = mcolors.Normalize(0, num_classes - 1)
        axes[1].imshow(
            np.digitize(target, unique_classes) - 1,
            cmap=custom_cmap,
            norm=norm,
            interpolation="nearest",
        )
        axes[1].axis("off")
        axes[2].imshow(
            np.digitize(preds, unique_classes, right=True),
            cmap=custom_cmap,
            norm=norm,
            interpolation="nearest",
        )
        axes[2].axis("off")
        patches = [
            Line2D([0], [0], color=colors[i], lw=4, label=str(unique_classes[i]))
            for i in range(num_classes)
        ]
        fig.legend(handles=patches, loc="upper left")
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, facecolor="black")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def scale_img_size(self, size: tuple[int, int]) -> list[int]:
        factor = max(self.img_size[0] / size[0], self.img_size[1] / size[1])
        return [round(s * factor) for s in size]

    def window_imgs(self, imgs: list[torch.Tensor]) -> tuple[torch.Tensor, list, list]:
        sizes = [img.shape[-2:] for img in imgs]
        crops, origins = [], []
        for idx, img in enumerate(imgs):
            resized = resize(img, self.scale_img_size(sizes[idx]))
            num = math.ceil(max(resized.shape[-2:]) / min(self.img_size))
            overlap = num * min(self.img_size) - max(resized.shape[-2:])
            step = (
                (min(self.img_size) - overlap / (num - 1))
                if overlap > 0
                else min(self.img_size)
            )
            for j in range(num):
                st = int(j * step)
                en = st + min(self.img_size)
                crop = (
                    resized[:, st:en, :]
                    if resized.shape[-2] > resized.shape[-1]
                    else resized[:, :, st:en]
                )
                crops.append(crop)
                origins.append((idx, st, en))
        return torch.stack(crops), origins, sizes

    def revert_window_logits(
        self, crop_logits: torch.Tensor, origins: list, sizes: list
    ) -> list[torch.Tensor]:
        sums, counts = [], []
        for size in sizes:
            h, w = self.scale_img_size(size)
            sums.append(
                torch.zeros(crop_logits.shape[1], h, w, device=crop_logits.device)
            )
            counts.append(
                torch.zeros(crop_logits.shape[1], h, w, device=crop_logits.device)
            )
        for i, (img_i, st, en) in enumerate(origins):
            if sizes[img_i][0] > sizes[img_i][1]:
                sums[img_i][:, st:en, :] += crop_logits[i]
                counts[img_i][:, st:en, :] += 1
            else:
                sums[img_i][:, :, st:en] += crop_logits[i]
                counts[img_i][:, :, st:en] += 1
        return [
            F.interpolate((s / c)[None], sizes[i], mode="bilinear")[0]
            for i, (s, c) in enumerate(zip(sums, counts))
        ]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, targets = batch
        crops, origins, sizes = self.window_imgs(imgs)
        logits_crops = self.forward(crops)
        logits_crops = F.interpolate(logits_crops, self.img_size, mode="bilinear")
        logits = self.revert_window_logits(logits_crops, origins, sizes)

        per_pixel = self.to_per_pixel(targets)
        for logit, mask in zip(logits, per_pixel):
            pred = logit.argmax(dim=0, keepdim=False)
            self.metric.update(pred[None], mask[None])

        if batch_idx == 0:
            plot = self.plot(imgs[0], per_pixel[0], logits=logits[0])
            self.trainer.logger.experiment.log(
                {f"val_pred_{batch_idx}": [wandb.Image(plot)]}
            )

    def on_validation_epoch_end(self):
        iou = self.metric.compute()
        self.metric.reset()
        self.log("val_miou", float(iou.mean()), sync_dist=True)

    def configure_optimizers(self):
        opt = super().configure_optimizers()
        total = int(self.trainer.estimated_stepping_batches)
        warmup_steps = 1500
        warmup = LambdaLR(
            opt,
            lr_lambda=lambda step: step / warmup_steps if step < warmup_steps else 1.0,
        )
        decay = PolynomialLR(
            opt, total_iters=total - warmup_steps, power=self.poly_lr_decay_power
        )
        sched = SequentialLR(opt, schedulers=[warmup, decay], milestones=[warmup_steps])
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
