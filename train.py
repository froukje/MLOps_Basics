import hydra
import omegaconf
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import logging
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import wandb
from data import DataModule
from model import ColaModel

logger = logging.getLogger(__name__)

class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        print("Callback on_validation_end")
        sentences = []
        for s in pl_module.validation_step_sentences:
            sentences.extend(s)
        preds = torch.cat(pl_module.validation_step_preds)
        labels = torch.cat(pl_module.validation_step_labels)
        logits = torch.cat(pl_module.validation_step_logits)

        df = pd.DataFrame(
            {
                "Sentence": sentences,
                "Label": labels.cpu().numpy(),
                "Predicted": preds.cpu().numpy(),
            }
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )

        data = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
        df_cm = pd.DataFrame(
            data, columns=np.unique(labels.cpu()), index=np.unique(labels.cpu())
        )
        df_cm.index.name = "Actual"
        df_cm.columns.name = "Predicted"
        plt.figure(figsize=(7, 4))
        plot = sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})
        trainer.logger.experiment.log(
            {
                "Confusion Matrix": wandb.Image(plot),
                "global_step": trainer.global_step,
            }
        )

        trainer.logger.experiment.log(
            {
                "roc": wandb.plot.roc_curve(labels.cpu().numpy(), logits.cpu().numpy()),
                "global_step": trainer.global_step,
            }
        )

        # pl_module.validation_step_preds.clear()


@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    #wandb.config = omegaconf.OmegaConf.to_container(
    #    cfg, resolve=True, throw_on_missing=True
    #)
    #wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)

    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.name}")
    logger.info(f"Using the tokenizer: {cfg.model.tokenizer}")
    cola_data = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    cola_model = ColaModel(cfg.model.name)

    root_dir = hydra.utils.get_original_cwd()
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{root_dir}/models", monitor="valid/loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    torch.set_float32_matmul_precision("medium")
    wandb_logger = WandbLogger(project="MLOps Basics")
    trainer = pl.Trainer(
        default_root_dir="logs",
        accelerator="auto",
        max_epochs=cfg.training.max_epochs,
        fast_dev_run=False,
        logger=wandb_logger,
        # logger=pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
        callbacks=[
            checkpoint_callback,
            SamplesVisualisationLogger(cola_data),
            early_stopping_callback,
        ],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
    )
    trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    main()
