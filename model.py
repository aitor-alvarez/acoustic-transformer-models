import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.hubert import HubertModel, HubertPreTrainedModel
from transformers.models.wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")
recall = evaluate.load('recall')
precision = evaluate.load('precision')

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
        return nn.Softmax(x)

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
            return_dict=None):

        outputs = self.w2v(input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        hidden_states = outputs[0]
        x = torch.mean(hidden_states, dim=1)
        x = self.classifier(x)
        return x


class HubertAcousticClassifier(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
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
        return_dict =  None):
        outputs = self.hubert(input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        hidden_states = outputs[0]
        x = torch.mean(hidden_states, dim=1)
        x = self.classifier(x)
        return x

class AcousticTransformer(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        print(config)
        if 'hubert' in config._name_or_path:
            self.model = HubertAcousticClassifier.from_pretrained(config._name_or_path, config=config)
        elif 'wav2vec' in config._name_or_path:
            self.model = Wav2VecAcousticClassifier.from_pretrained(config._name_or_path, config=config)
        self.model.freeze_feature_extractor()

    def compute_metrics(self, eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        acc = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
        rec_w = recall.compute(predictions=predictions, references=eval_pred.label_ids, average='weighted')
        rec_u = recall.compute(predictions=predictions, references=eval_pred.label_ids, average=None)
        return acc, rec_w, rec_u


    def training_step(self, batch):
        x = batch['input_values']
        y = batch['label']
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        self.log('Training loss', loss)
        return loss

    def test_step(self, batch):
        x = batch['input_features']
        y = batch['labels']
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]