import torch
import lightning as L
import torch.nn as nn
from torch.optim import AdamW


class LightningModule(L.LightningModule):
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
        self.net_eager = network
        for p in network.encoder.parameters():
            p.requires_grad = not freeze_encoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_multiplier_encoder = lr_multiplier_encoder
        self._train_core_compiled = None

    def forward(self, imgs):
        if isinstance(imgs, list):
            imgs = torch.stack(imgs)
        return self.net_eager(imgs / 255.0)

    def compile_train_core(self, train_core_fn):
        if self._train_core_compiled is None:
            self._train_core_compiled = torch.compile(
                train_core_fn, mode="max-autotune"
            )
        return self._train_core_compiled

    def configure_optimizers(self):
        enc_names = {n for n, _ in self.net_eager.encoder.named_parameters()}
        base_params, enc_params = [], []
        for name, param in self.named_parameters():
            key = name.replace("net_eager.encoder.", "", 1)
            (enc_params if key in enc_names else base_params).append(param)
        return AdamW(
            [
                {"params": base_params, "lr": self.lr},
                {"params": enc_params, "lr": self.lr * self.lr_multiplier_encoder},
            ],
            weight_decay=self.weight_decay,
        )
