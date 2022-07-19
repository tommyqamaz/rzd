def show_batch(nrows, ncols, datamodule):
    nrows = 3
    ncols = 3
    batch_size = nrows * ncols
    data_module = datamodule(batch_size=batch_size)
    data_module.setup()
    data_loader = data_module.train_dataloader()

    images, masks = next(iter(data_loader))

    fig, _ = plt.subplots(figsize=(10, 10))
    for i, (image, mask) in enumerate(zip(images, masks)):
        plt.subplot(nrows, ncols, i + 1)
        plt.tight_layout()
        plt.axis("off")

        image = image.permute(1, 2, 0).numpy()
        mask = mask.numpy()

        print(image.shape, image.min(), image.max(), image.mean(), image.std())
        print(mask.shape, mask.min(), mask.max(), mask.mean(), mask.std())

        plt.imshow(image)
        plt.imshow(mask, alpha=0.2)
