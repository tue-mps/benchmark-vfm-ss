import math
import lightning
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics.classification import MulticlassJaccardIndex
from PIL import Image
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import io
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import interpolate
from torchvision.transforms.v2.functional import resize


class LightningModule(lightning.LightningModule):
    def __init__(
        self,
        img_size: tuple[int, int],
        freeze_encoder: bool,
        network: nn.Module,
        weight_decay: float,
        lr: float,
        lr_multiplier_encoder: float,
    ):
        super().__init__()

        self.img_size = img_size
        self.network = network
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_multiplier_encoder = lr_multiplier_encoder

        for param in self.network.encoder.parameters():
            param.requires_grad = not freeze_encoder

        self.log = torch.compiler.disable(self.log)  # type: ignore

    def init_metrics_semantic(self, num_classes, ignore_idx, num_metrics):
        self.metrics = nn.ModuleList(
            [
                MulticlassJaccardIndex(
                    num_classes=num_classes,
                    validate_args=False,
                    ignore_index=ignore_idx,
                    average=None,
                )
                for _ in range(num_metrics)
            ]
        )

    @torch.compiler.disable
    def update_metrics(
        self, preds: list[torch.Tensor], targets: list[torch.Tensor], dataloader_idx
    ):
        for i in range(len(preds)):
            self.metrics[dataloader_idx].update(
                preds[i][None, ...], targets[i][None, ...]
            )

    def forward(self, imgs):
        x = imgs / 255.0

        output = self.network(x)

        if not self.training and isinstance(output, tuple):
            return (y[-1] for y in self.network(x))

        return output

    def on_train_batch_start(self, _, batch_idx):
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(
                f"group_{i}_lr_{len(param_group['params'])}",
                param_group["lr"],
                on_step=True,
            )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {
            k.replace("._orig_mod", ""): v for k, v in checkpoint["state_dict"].items()
        }

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self.eval_step(batch, batch_idx, dataloader_idx, "val")

    def _on_eval_epoch_end_semantic(self, log_prefix):
        miou_per_dataset = []
        iou_per_dataset_per_class = []
        for metric_idx, metric in enumerate(self.metrics):
            iou_per_dataset_per_class.append(metric.compute())
            metric.reset()

            for iou_idx, iou in enumerate(iou_per_dataset_per_class[-1]):
                self.log(
                    f"{log_prefix}_{metric_idx}_iou_{iou_idx}", iou, sync_dist=True
                )

            miou_per_dataset.append(float(iou_per_dataset_per_class[-1].mean()))
            self.log(
                f"{log_prefix}_{metric_idx}_miou", miou_per_dataset[-1], sync_dist=True
            )

    def configure_optimizers(self):
        encoder_param_names = {
            name for name, _ in self.network.encoder.named_parameters()
        }
        base_params = []
        encoder_params = []

        for name, param in self.named_parameters():
            if name.replace("network.encoder.", "") in encoder_param_names:
                encoder_params.append(param)
            else:
                base_params.append(param)

        return AdamW(
            [
                {"params": base_params, "lr": self.lr},
                {
                    "params": encoder_params,
                    "lr": self.lr * self.lr_multiplier_encoder,
                },
            ],
            weight_decay=self.weight_decay,
        )

    @torch.compiler.disable
    def plot_semantic(
        self,
        img,
        target,
        logits=None,
        cmap="tab20",
    ):
        fig, axes = plt.subplots(1, 3, figsize=[15, 5], sharex=True, sharey=True)

        axes[0].imshow(img.cpu().numpy().transpose(1, 2, 0))
        axes[0].axis("off")

        target = target.cpu().numpy()
        unique_classes = np.unique(target)

        preds = None
        if logits is not None:
            preds = torch.argmax(logits, dim=0).cpu().numpy()
            unique_classes = np.unique(
                np.concatenate((unique_classes, np.unique(preds)))
            )

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

        if preds is not None:
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

    def scale_img_size_semantic(self, size: tuple[int, int]):
        factor = max(
            self.img_size[0] / size[0],
            self.img_size[1] / size[1],
        )

        return [round(s * factor) for s in size]

    def window_imgs_semantic(self, imgs):
        img_sizes = [img.shape[-2:] for img in imgs]

        crops, origins = [], []
        for i in range(len((imgs))):
            new_img = resize(
                imgs[i],
                self.scale_img_size_semantic(img_sizes[i]),
            )

            num_crops = math.ceil(max(new_img.shape[-2:]) / min(self.img_size))
            overlap = num_crops * min(self.img_size) - max(new_img.shape[-2:])
            overlap_per_crop = (overlap / (num_crops - 1)) if overlap > 0 else 0

            for j in range(num_crops):
                start = int(j * (min(self.img_size) - overlap_per_crop))
                end = start + min(self.img_size)
                crop = (
                    new_img[:, start:end, :]
                    if new_img.shape[-2] > new_img.shape[-1]
                    else new_img[:, :, start:end]
                )

                crops.append(crop)
                origins.append((i, start, end))

        return torch.stack(crops), origins, [img.shape[-2:] for img in imgs]

    def revert_window_logits_semantic(self, crop_logits, origins, img_sizes):
        logit_sums, logit_counts = [], []
        for size in img_sizes:
            h, w = self.scale_img_size_semantic(size)
            logit_sums.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )
            logit_counts.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )

        for crop_i, (img_i, start, end) in enumerate(origins):
            if img_sizes[img_i][0] > img_sizes[img_i][1]:
                logit_sums[img_i][:, start:end, :] += crop_logits[crop_i]
                logit_counts[img_i][:, start:end, :] += 1
            else:
                logit_sums[img_i][:, :, start:end] += crop_logits[crop_i]
                logit_counts[img_i][:, :, start:end] += 1

        return [
            interpolate((sums / counts)[None, ...], img_sizes[i], mode="bilinear")[0]
            for i, (sums, counts) in enumerate(zip(logit_sums, logit_counts))
        ]

    @staticmethod
    def to_per_pixel_logits_semantic(
        mask_logits: torch.Tensor, class_logits: torch.Tensor
    ):
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )

    @staticmethod
    @torch.compiler.disable
    def to_per_pixel_targets_semantic(
        targets: list[dict],
        ignore_idx,
    ):
        per_pixel_targets = []
        for target in targets:
            per_pixel_target = torch.full(
                target["masks"].shape[-2:],
                ignore_idx,
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                per_pixel_target[mask] = target["labels"][i]

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets
