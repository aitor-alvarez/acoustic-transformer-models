from typing import Any

import lightning as L
import torch
import torch.nn as nn
from transformers.models.hubert import HubertModel
from transformers.models.wav2vec2 import Wav2Vec2Model
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
            self.model = HubertModel(config)
            self.model = self.model.from_pretrained(config._name_or_path, config=config)
            self.model.feature_extractor._freeze_parameters()
            self.model.init_weights()
        elif 'wav2vec' in self.config._name_or_path:
            self.model = Wav2Vec2Model(config)
            self.model = self.model.from_pretrained(config._name_or_path, config=config)
            self.model.freeze_feature_extractor()
            self.model.init_weights()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self,
        input_values,
        attention_mask = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict =  None):

        outputs = self.model(input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
        hidden_states = outputs[0]
        x = torch.mean(hidden_states, dim=1)
        x = self.dense(x)
        x = self.linear(x)
        return x

    def compute_metrics(self, preds, targets, name):
        accuracy = self.train_acc(preds, targets)
        recall = self.train_rec(preds, targets)
        f1 = self.train_f1(preds, targets)
        self.log(f'{name} Accuracy', accuracy, prog_bar=True)
        self.log(f'{name} Recall', recall, prog_bar=True)
        self.log(f'{name} F1 Score', f1, prog_bar=True)
        return None

    def training_step(self, batch):
        x = batch['input_values']
        y = batch['labels']
        logits = self.forward(x)
        celoss = nn.CrossEntropyLoss()
        preds = logits.view(-1, self.num_labels)
        targets = y.view(-1)
        loss = celoss(preds, targets)
        self.log('Training Loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x = batch['input_values']
        y = batch['labels']
        logits = self.forward(x)
        celoss = nn.CrossEntropyLoss()
        preds = logits.view(-1, self.num_labels)
        targets = y.view(-1)
        loss = celoss(preds, targets)
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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)
        # update learning rate
        self.lr_schedulers().step()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]