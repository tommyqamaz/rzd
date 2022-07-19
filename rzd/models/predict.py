def predict(model, dataset, savedir: Path = SUBMISSION_PATH):
    os.makedirs(SUBMISSION_PATH)
    test_transform = {
        "out": A.Resize(2160, 3840),
        "in": A.Compose(
            [
                A.Resize(*(512, 512)),
                A.augmentations.transforms.Normalize(
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
                ),
                ToTensorV2(),
            ]
        ),
    }
    transform_submit = np.vectorize(lambda x: MAP_SUBMIT[x])

    for pair in dataset:
        image, image_name = pair
        image = test_transform["in"](image=image)["image"]
        mask = model(image.reshape(1, *image.shape))

        mask_np = (
            mask.argmax(dim=1).numpy().reshape(512, 512)
        )  # size depends on your model
        mask_qhd = test_transform["out"](image=mask_np.astype(np.float64))[
            "image"
        ].astype(int)
        cv2.imwrite(
            os.path.join(SUBMISSION_PATH, image_name), transform_submit(mask_qhd)
        )
