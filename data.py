import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from transformers import AutoFeatureExtractor
from torch.utils.data import DataLoader

class AudioDataset(L.LightningDataModule):
    def __init__(self, model_name, batch_size, dataset):
        super().__init__()
        self.train = None
        self.test = None
        self.batch_size = batch_size
        self.dataset = dataset
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    def preprocess_function(self, examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = self.feature_extractor(
            audio_arrays, sampling_rate=self.feature_extractor.sampling_rate, max_length=16000,
            padding=True, truncation=True
        )
        return inputs

    def setup(self, stage: str):
        self.train = self.dataset["train"]
        self.test = self.dataset["test"]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.preprocess_function)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, batch_size=1, collate_fn=self.preprocess_function)

