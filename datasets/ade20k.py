from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.mappings import get_ade20k_mapping
from datasets.transforms import Transforms


class ADE20K(LightningDataModule):
    def __init__(
        self,
        root,
        num_workers: int,
        img_size: tuple[int, int] = (224, 224),
        batch_size: int = 4,
        num_classes: int = 150,
        num_metrics: int = 1,
        scale_range=(0.5, 2.0),
        ignore_idx: int = 255,
    ) -> None:
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            num_metrics=num_metrics,
            ignore_idx=ignore_idx,
            img_size=img_size,
            scale_range=scale_range,
        )
        self.save_hyperparameters()

        self.transforms = Transforms(img_size=img_size, scale_range=scale_range)

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "zip_path": Path(self.root, "ADEChallengeData2016.zip"),
            "target_zip_path": Path(self.root, "ADEChallengeData2016.zip"),
            "class_mapping": get_ade20k_mapping(),
            "ignore_idx": self.ignore_idx,
            "img_size": self.img_size,
            "scale_range": self.scale_range,
        }
        self.train_dataset = Dataset(
            img_folder_path_in_zip=Path("./ADEChallengeData2016/images/training"),
            target_folder_path_in_zip=Path(
                "./ADEChallengeData2016/annotations/training"
            ),
            transforms=self.transforms,
            check_empty_targets=True,
            **dataset_kwargs,
        )
        self.val_dataset = Dataset(
            img_folder_path_in_zip=Path("./ADEChallengeData2016/images/validation"),
            target_folder_path_in_zip=Path(
                "./ADEChallengeData2016/annotations/validation"
            ),
            check_empty_targets=False,
            **dataset_kwargs,
        )

        return self

    def train_dataloader(self):
        dataset = self.train_dataset

        return DataLoader(
            dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
