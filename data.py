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
        new_batch={}
        audio = batch["audio"]
        new_batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"])['input_values']
        new_batch["labels"] = batch['label']
        return new_batch

    def collate_function(self, data):
        input_features = BatchFeature([{"input_features": feature["input_features"]} for feature in data])
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")
        labels = [x["labels"] for x in data]
        d_type = torch.long if isinstance(labels[0], int) else torch.float
        batch['labels'] = torch.tensor(labels, dtype=d_type)
        print(batch)
        return batch

    def setup(self, stage: str =None):
        self.train = self.dataset["train"]
        self.train = self.train.map(self.prepare_dataset, remove_columns=self.train.column_names)
        #self.test = self.dataset["test"].with_format("torch")
       #self.test = self.test.map(self.preprocess_fun, batch_size=10, batched=True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.collate_function)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, batch_size=1)