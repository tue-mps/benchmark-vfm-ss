from torch.utils.data import DataLoader

from datasets.urban_scenes import UrbanScenes


class Cityscapes(UrbanScenes):
    def __init__(
        self,
        root,
        devices,
        num_workers: int,
        img_size: tuple[int, int] = (1024, 1024),
        batch_size: int = 1,
        num_classes: int = 19,
        num_metrics: int = 1,
        ignore_idx: int = 255,
    ) -> None:
        super().__init__(
            root=root,
            devices=devices,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            num_metrics=num_metrics,
            ignore_idx=ignore_idx,
            img_size=img_size,
        )

        self.save_hyperparameters()

    def train_dataloader(self):
        return DataLoader(
            self.cityscapes_train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )
