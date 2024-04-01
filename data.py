import datasets
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
)  # This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library when created with the AutoTokenizer.from_pretrained() class method.


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name="google/bert_uncased_L-2_H-128_A-2",
        batch_size=64,
        max_length=128,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"], truncation=True, padding="max_length", max_length=512
        )

    def setup(self, stage=None):
        #  we set up only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
            # The fastest way to tokenize your entire dataset is to use the map() function.
            # This function speeds up tokenization by applying the tokenizer to batches of examples
            # instead of individual examples. Set the batched parameter to True:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            # Set the format of your dataset to be compatible with your machine learning framework
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"],
                output_all_columns=True,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=8
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, drop_last=True
        )


if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)
