import lightning as L
import torch
import torch.nn as nn
from transformers.models.hubert import HubertForSequenceClassification
from transformers.models.wav2vec2 import Wav2Vec2ForSequenceClassification
from transformers.models.wav2vec2_conformer import Wav2Vec2ConformerForSequenceClassification
from torchmetrics import Accuracy, Recall, F1Score

class AcousticTransformer(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.train_acc = Accuracy(task="multiclass", num_classes=config.num_labels)
        self.train_f1 = F1Score(task="multiclass", average='macro', num_classes=config.num_labels)
        self.train_rec = Recall(task="multiclass", average='macro', num_classes=config.num_labels)
        self.config = config
        self.num_labels = self.config.num_labels
        if 'hubert' in self.config._name_or_path:
            self.model = HubertForSequenceClassification.from_pretrained(config._name_or_path, config=config)
            self.model.freeze_feature_encoder()

        elif 'wav2vec2-conformer' in self.config._name_or_path:
            self.model = Wav2Vec2ConformerForSequenceClassification.from_pretrained(config._name_or_path, config=config)
            self.model.freeze_feature_encoder()

        elif 'wav2vec' in self.config._name_or_path:
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(config._name_or_path, config=config)
            self.model.freeze_feature_extractor()
        self.model.train()

    def compute_metrics(self, preds, targets, name):
        accuracy = self.train_acc(preds, targets)
        recall = self.train_rec(preds, targets)
        f1 = self.train_f1(preds, targets)
        self.log(f'{name} Accuracy', accuracy, prog_bar=True)
        self.log(f'{name} Recall', recall, prog_bar=True)
        self.log(f'{name} F1 Score', f1, prog_bar=True)
        return None

    def training_step(self, batch):
        assert self.model.training
        x = batch['input_values']
        y = batch['labels']
        outputs = self.model(x, labels=y)
        loss = outputs[0]
        self.log('Training Loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x = batch['input_values']
        y = batch['labels']
        outputs = self.model(x, labels=y)
        loss = outputs[0]
        logits = outputs[1]
        preds = logits.view(-1, self.num_labels)
        targets = y.view(-1)
        self.log('Validation Loss', loss, prog_bar=True)
        self.compute_metrics(preds, targets, 'Validation')

    def test_step(self, batch):
        x = batch['input_features']
        y = batch['labels']
        logits = self.forward(x)
        celoss = nn.CrossEntropyLoss()
        preds = logits.view(-1, self.num_labels)
        targets = y.view(-1)
        loss = celoss(preds, targets)
        self.log('Test Loss', loss, prog_bar=True)
        self.compute_metrics(preds, targets, 'Test')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]