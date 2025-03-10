import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from transformers import AutoFeatureExtractor
from torch.utils.data import DataLoader
import torch

class AudioDataset(L.LightningDataModule):
    def __init__(self, model_name, batch_size, dataset):
        super().__init__()
        self.train = None
        self.test = None
        self.batch_size = batch_size
        self.dataset = dataset
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    def prepare_dataset(self, batch):
        audio = [x['array'] for x in batch['audio']]
        inputs = self.feature_extractor(
            audio, sampling_rate=self.feature_extractor.sampling_rate
        )
        return inputs

    def get_max_len(self, inputs):
        max_len=0
        for i in inputs:
            if torch.numel(i['input_values'][0]) > max_len:
                max_len = len(i['input_values'][0])
        return max_len

    def collate(self, inputs):
        max_dur = self.get_max_len(inputs)
        input_features = [{"input_values": i['input_values'][0]} for i in inputs]
        labels = [i["label"] for i in inputs]
        batch = self.feature_extractor.pad(
            input_features,
            padding=True,
            truncation=True,
            max_length=16000*max_dur,
        )

        batch["labels"] = torch.tensor(labels).long()
        return batch

    def setup(self, stage: str =None):
        train_data = self.dataset["train"].with_format("torch").shuffle(seed=42)
        k = round(len(train_data)/5)
        end = len(train_data)
        self.val_dataset = train_data.select((range(0, k)))
        self.val_dataset = self.val_dataset.map(self.prepare_dataset, remove_columns='audio',
                                                batch_size=1, writer_batch_size=100, batched=True)
        self.train = train_data.select((range(k, end)))
        del train_data
        self.train = self.train.map(self.prepare_dataset, remove_columns='audio',
                                              batch_size=1, writer_batch_size=100, batched=True)
        self.test = self.dataset["test"].with_format("torch")
        self.test = self.test.map(self.prepare_dataset, remove_columns='audio',
                                  batch_size=1, writer_batch_size=100, batched=True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train, batch_size=self.batch_size,
                          collate_fn=self.collate, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, batch_size=1, collate_fn=self.collate)