from pathlib import Path
from typing import Union
import zipfile
from torch.utils.data import DataLoader, Subset
from os.path import basename, splitext

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms


class PascalVOC(LightningDataModule):
    def __init__(
        self,
        root,
        devices,
        num_workers: int,
        img_size: tuple[int, int] = (512, 512),
        batch_size: int = 1,
        num_classes: int = 21,
        num_metrics: int = 1,
        scale_range=(0.5, 2.0),
        ignore_idx: int = 255,
        zip_path: str = "VOCtrainval_11-May-2012.zip",
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
        self.zip_path = Path(root) / zip_path
        self.save_hyperparameters()

        self.transforms = Transforms(img_size=img_size, scale_range=scale_range)

    def _read_split_file(self, file_path_in_zip):
        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            with zip_ref.open(file_path_in_zip) as f:
                return [line.decode("utf-8").strip() for line in f.readlines()]

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        full_dataset_kwargs = {
            "zip_path": self.zip_path,
            "img_folder_path_in_zip": Path("./VOCdevkit/VOC2012/JPEGImages"),
            "target_folder_path_in_zip": Path("./VOCdevkit/VOC2012/SegmentationClass"),
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "ignore_idx": self.ignore_idx,
        }

        train_names = self._read_split_file(
            "VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
        )
        val_names = self._read_split_file(
            "VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
        )

        full_dataset = Dataset(**full_dataset_kwargs, transforms=self.transforms)
        train_indices = [
            i
            for i, img_name in enumerate(full_dataset.imgs)
            if splitext(basename(img_name))[0] in train_names
        ]
        val_dataset = Dataset(**full_dataset_kwargs)
        val_indices = [
            i
            for i, img_name in enumerate(val_dataset.imgs)
            if splitext(basename(img_name))[0] in val_names
        ]

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(val_dataset, val_indices)

        return self

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
