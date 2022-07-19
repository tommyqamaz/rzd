class RZDDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path] = ALL_IMAGES,
        mask_paths: List[Path] = ALL_MASKS,
        transforms: Callable = None,
    ):
        self.image_paths = image_paths

        self.mask_paths = mask_paths

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        if self.transforms is not None:
            data = self.transforms(image=image, mask=mask)
            image, mask = data["image"], data["mask"]

        return image, mask

    @staticmethod
    def _load_image(image_path: Path) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_mask(mask_path: Path) -> np.ndarray:
        transorm_mask = np.vectorize(lambda x: MAP_MASKS[x])
        return transorm_mask(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE))


class TestRZD(RZDDataset):
    def __init__(self, image_paths: List[Path] = TEST_IMAGES):
        super().__init__(image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]
        image_name = image_path.split("/")[-1]
        image = self._load_image(image_path)
        return image, image_name
