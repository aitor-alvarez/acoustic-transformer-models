import lightning as L
import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers.models.hubert import HubertModel, HubertPreTrainedModel
from transformers.models.wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from torchmetrics import Accuracy, Recall, F1Score
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple


@dataclass
class AcousticTransformerOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    preds: torch.LongTensor = None
    targets: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ClassifierModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dense(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class Wav2VecAcousticClassifier(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.w2v = Wav2Vec2Model(config)
        self.classifier = ClassifierModule(config)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.w2v.feature_extractor._freeze_parameters()

    def forward(self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
                labels=None):

        outputs = self.w2v(input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        hidden_states = outputs[0]
        x = torch.mean(hidden_states, dim=1)
        logits = self.classifier(x)
        celoss = nn.CrossEntropyLoss()
        preds = logits.view(-1, self.num_labels)
        targets = labels.view(-1)
        loss = celoss(preds, targets)
        return AcousticTransformerOutput(loss=loss, logits=logits, preds=preds, targets=targets)


class HubertAcousticClassifier(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = self.config.num_labels
        self.hubert = HubertModel(config)
        self.classifier = ClassifierModule(config)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.hubert.feature_extractor._freeze_parameters()
    def forward(self,
        input_values,
        attention_mask = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict =  None,
                labels = None):
        outputs = self.hubert(input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        hidden_states = outputs[0]
        x = torch.mean(hidden_states, dim=1)
        logits = self.classifier(x)
        celoss = nn.CrossEntropyLoss()
        preds = logits.view(-1, self.num_labels)
        targets = labels.view(-1)
        loss = celoss(preds, targets)
        return AcousticTransformerOutput(loss=loss, logits=logits, preds=preds, targets=targets)

class AcousticTransformer(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.train_acc = Accuracy(task="multiclass", num_classes=config.num_labels)
        self.train_f1 = F1Score(task="multiclass", average='macro', num_classes=config.num_labels)
        self.train_rec = Recall(task="multiclass", average='macro', num_classes=config.num_labels)
        self.config = config
        if 'hubert' in config._name_or_path:
            self.model = HubertAcousticClassifier.from_pretrained(config._name_or_path, config=config)
        elif 'wav2vec' in config._name_or_path:
            self.model = Wav2VecAcousticClassifier.from_pretrained(config._name_or_path, config=config)
        self.model.freeze_feature_extractor()

    def compute_metrics(self, output, name):
        preds = output[2]
        targets = output[3]
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
        output = self.model(x, labels=y)
        loss = output[0]
        self.log('Training Loss', loss, prog_bar=True)
        self.compute_metrics(output, 'Training')
        return loss

    def test_step(self, batch):
        x = batch['input_features']
        y = batch['labels']
        output = self.model(x, labels=y)
        loss = output[0]
        self.log('Test Loss', loss, prog_bar=True)
        self.compute_metrics(output, 'Test')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]