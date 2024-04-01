# import numpy as np
# import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
from transformers import AutoModelForSequenceClassification

import wandb


class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=1e-2):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()
        self.validation_step_logits = []
        self.validation_step_labels = []
        self.validation_step_preds = []
        self.validation_step_sentences = []

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.W = nn.Linear(self.bert.config.hidden_size, 2)
        self.num_classes = 2
        self.train_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.val_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.f1_metric = torchmetrics.F1Score(
            task="binary", num_classes=self.num_classes
        )
        self.precision_macro_metric = torchmetrics.Precision(
            task="binary", average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            task="binary", average="macro", num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(
            task="binary", average="micro"
        )
        self.recall_micro_metric = torchmetrics.Recall(task="binary", average="micro")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        # loss = F.cross_entropy(logits, batch["label"])
        preds = torch.argmax(outputs.logits, dim=1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):

        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        # loss = F.cross_entropy(logits, batch["label"])
        preds = torch.argmax(outputs.logits, dim=1)
        self.validation_step_preds.append(preds)
        self.validation_step_labels.append(batch["label"])
        self.validation_step_logits.append(outputs.logits)
        self.validation_step_sentences.append(batch["sentence"])
        # Metrics
        valid_acc = self.val_accuracy_metric(preds, batch["label"])
        precision_macro = self.precision_macro_metric(preds, batch["label"])
        recall_macro = self.recall_macro_metric(preds, batch["label"])
        precision_micro = self.precision_micro_metric(preds, batch["label"])
        recall_micro = self.recall_micro_metric(preds, batch["label"])
        f1 = self.f1_metric(preds, batch["label"])
        # Logging metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        return {"labels": batch["label"], "logits": outputs.logits}

    # def on_validation_epoch_end(self):
    #    preds = torch.cat(self.validation_step_preds)
    #    labels = torch.cat(self.validation_step_labels)
    #    logits = torch.cat(self.validation_step_logits)

    # 1. Confusion matrix plotting using inbuilt W&B method
    # self.logger.experiment.log(
    #    {
    #        "conf": wandb.plot.confusion_matrix(
    #            probs=logits.cpu().numpy(), y_true=labels.cpu().numpy(),
    #        )
    #    }
    # )
    # 2. Confusion Matrix plotting using scikit-learn method
    # wandb.log(
    #    {
    #        "cm": wandb.sklearn.plot_confusion_matrix(
    #            labels.cpu().numpy(), preds.cpu().numpy()
    #        )
    #    }
    # )
    # 3. Confusion Matric plotting using Seaborn
    # print(labels.shape)
    # print(preds.shape)
    #    data = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
    #    df_cm = pd.DataFrame(
    #        data, columns=np.unique(labels.cpu()), index=np.unique(labels.cpu())
    #    )
    #    df_cm.index.name = "Actual"
    #    df_cm.columns.name = "Predicted"
    #    plt.figure(figsize=(7, 4))
    #    plot = sns.heatmap(
    #        df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}
    #    )  # font size
    #    self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

    #   self.logger.experiment.log(
    #        {"roc": wandb.plot.roc_curve(labels.cpu().numpy(), logits.cpu().numpy())}
    #    )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
