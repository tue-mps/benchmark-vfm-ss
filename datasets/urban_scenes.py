from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader, ConcatDataset

from datasets.lightning_data_module import LightningDataModule
from datasets.mappings import get_cityscapes_mapping
from datasets.dataset import Dataset
from datasets.transforms import Transforms


class UrbanScenes(LightningDataModule):
    def __init__(
        self,
        root,
        devices,
        batch_size: int,
        img_size: tuple[int, int],
        num_workers: int,
        num_classes: int,
        num_metrics: int,
        ignore_idx: int,
        scale_range=(0.5, 2.0),
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

        self.transforms = Transforms(self.img_size, scale_range)

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        gta5_train_datasets = [
            Dataset(
                zip_path=Path(self.root, f"{i:02}_images.zip"),
                target_zip_path=Path(self.root, f"{i:02}_labels.zip"),
                img_folder_path_in_zip=Path("./images"),
                target_folder_path_in_zip=Path("./labels"),
                img_suffix=".png",
                target_suffix=".png",
                class_mapping=get_cityscapes_mapping(),
                ignore_idx=self.ignore_idx,
                transforms=self.transforms,
            )
            for i in range(1, 11)
        ]
        self.gta5_train_dataset = ConcatDataset(gta5_train_datasets)

        cityscapes_dataset_kwargs = {
            "img_suffix": ".png",
            "target_suffix": ".png",
            "img_stem_suffix": "leftImg8bit",
            "target_stem_suffix": "gtFine_labelIds",
            "zip_path": Path(self.root, "leftImg8bit_trainvaltest.zip"),
            "target_zip_path": Path(self.root, "gtFine_trainvaltest.zip"),
            "class_mapping": get_cityscapes_mapping(),
        }
        self.cityscapes_train_dataset = Dataset(
            transforms=self.transforms,
            img_folder_path_in_zip=Path("./leftImg8bit/train"),
            target_folder_path_in_zip=Path("./gtFine/train"),
            ignore_idx=self.ignore_idx,
            **cityscapes_dataset_kwargs,
        )
        self.cityscapes_val_dataset = Dataset(
            img_folder_path_in_zip=Path("./leftImg8bit/val"),
            target_folder_path_in_zip=Path("./gtFine/val"),
            ignore_idx=self.ignore_idx,
            **cityscapes_dataset_kwargs,
        )

        return self

    def val_dataloader(self):
        return DataLoader(
            self.cityscapes_val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
