import lightning as L

class AudioDataset(L.LightningDataModule):
    def __init__(self, batch_size, dir_path):
        super().__init__()
        self.train = None
        self.test = None
        self.batch_size = batch_size
        self.dir = dir_path

    def setup(self, stage: str):
