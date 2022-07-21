class RZDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset=RZDDataset,
        all_images: List[Path] = ALL_IMAGES,
        all_masks: List[Path] = ALL_MASKS,
        train_size_coef: int = 0.8,
        batch_size: int = 8,
        num_workers: int = 2,
        input_shape: Tuple[int, int] = (512, 512),
    ):
        super().__init__()

        self.dataset = dataset
        self.all_images = all_images
        self.all_masks = all_masks
        self.save_hyperparameters()

        self.train_transforms, self.val_transforms = self._init_transforms()

    def _init_transforms(self) -> Tuple[Callable, Callable]:
        train_transforms = [
            A.Resize(*self.hparams.input_shape),
            A.OneOf(
                [
                    A.RandomFog(),
                    A.RandomSunFlare(src_radius=100),
                    A.NoOp(),
                    A.RandomShadow(),
                ]
            ),
            A.HorizontalFlip(p=0.5),
            A.OpticalDistortion(),
            A.RandomBrightnessContrast(),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

        val_transforms = [
            A.Resize((512, 512)),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

        return A.Compose(train_transforms), A.Compose(val_transforms)

    def setup(self, stage=None):
        images_train, images_val, masks_train, masks_val = train_test_split(
            self.all_images, self.all_masks, train_size=self.hparams.train_size_coef
        )
        self.train_dataset = self.dataset(
            images_train, masks_train, self.train_transforms
        )
        self.val_dataset = self.dataset(images_val, masks_val, self.val_transforms)

    def train_dataloader(self):
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def _dataloader(self, dataset: RZDDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
