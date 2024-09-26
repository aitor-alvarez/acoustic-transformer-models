import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from transformers import AutoFeatureExtractor, BatchFeature
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
            audio, sampling_rate=self.feature_extractor.sampling_rate, max_length=16000, padding=True, truncation=True
        )
        return inputs


    def collate(self, inputs):
        max_dur = max([len(x['input_values'][0]) for x in inputs])
        input_features = [{"input_values": i['input_values'][0]} for i in inputs]
        labels = [i["label"] for i in inputs]
        d_type = torch.long if isinstance(labels[0], int) else torch.float

        batch = self.feature_extractor.pad(
            input_features,
            padding=True,
            truncation=True,
            max_length=16000*max_dur,
        )

        batch["labels"] = torch.tensor(labels).long()
        return batch

    def setup(self, stage: str =None):
        self.train = self.dataset["train"].with_format("torch")
        self.encoded_dataset = self.train.map(self.prepare_dataset, remove_columns='audio', batch_size=1, batched=True)
        #self.test = self.dataset["test"].with_format("torch")
       #self.test = self.test.map(self.preprocess_fun, batch_size=10, batched=True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.encoded_dataset, batch_size=self.batch_size, collate_fn=self.collate)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, batch_size=1)