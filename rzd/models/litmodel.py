class RZDModel(pl.LightningModule):
    def __init__(
        self,
        loss: str,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        scheduler: str,
        T_max: int,
        T_0: int,
        min_lr: int,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = self._init_model()

        self.loss_fn = self._init_loss_fn()

    #         self.metrics = self._init_metrics()

    def _init_model(self) -> nn.Module:
        return smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_depth=5,
            encoder_weights=None,
            decoder_channels=(512, 256, 128, 64, 16),
            in_channels=3,
            classes=4,
            activation=None,
        )

    def _init_loss_fn(self) -> Callable:
        loss = self.hparams.loss
        assert loss in LOSS_FNS, "Choose from exstisting!"
        return LOSS_FNS[loss]

    #     def _init_metrics(self) -> nn.ModuleDict:
    #         train_metrics = MetricCollection({"train_dice": Dice()})
    #         val_metrics = MetricCollection({"val_dice": Dice()})

    #         return nn.ModuleDict(
    #             {
    #                 "train_metrics": train_metrics,
    #                 "val_metrics": val_metrics,
    #             }
    #         )

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer_kwargs = dict(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(**optimizer_kwargs)
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(**optimizer_kwargs)
        elif self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(**optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

        if self.hparams.scheduler is not None:
            if self.hparams.scheduler == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.hparams.T_max, eta_min=self.hparams.min_lr
                )
            elif self.hparams.scheduler == "CosineAnnealingWarmRestarts":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=self.hparams.T_0, eta_min=self.hparams.min_lr
                )
            else:
                raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        else:
            return {"optimizer": optimizer}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.shared_step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        self.shared_step(batch, "val")

    def shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> torch.Tensor:
        images, masks = batch
        y_pred = self(images)

        loss = self.loss_fn(y_pred, masks.type(torch.int64))  # error here
        #         metrics = self.metrics[f"{stage}_metrics"](y_pred, masks)

        self._log(loss, metrics={}, stage=stage)

        return loss

    def _log(self, loss: torch.Tensor, metrics: dict, stage: str):
        on_step = True if stage == "train" else False
        self.log(
            f"{stage}_loss", loss
        )  # , on_step=on_step, on_epoch=True, prog_bar=not on_step)

    #         self.log_dict(metrics, on_step=False, on_epoch=True)

    @classmethod
    def load_eval_checkpoint(cls, checkpoint_path: Path, device: str) -> nn.Module:
        module = cls.load_from_checkpoint(checkpoint_path=checkpoint_path).to(device)
        module.eval()

        return module
