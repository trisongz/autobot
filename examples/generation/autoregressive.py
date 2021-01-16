import pandas as pd
import auto
import torch
import torch.nn as nn
import transformers
from sklearn import metrics, model_selection, preprocessing
from transformers import AdamW, get_linear_schedule_with_warmup, AutoModelForCausalLM, AutoModelForPreTraining, AutoTokenizer


class GenerationModel(auto.Model):
    def __init__(self, num_train_steps, model_name='gpt2-base'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if 'gpt' in model_name:
            self.tokenizer.add_special_tokens({'pad_token': '<|padding|>'})
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self._loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.num_train_steps = num_train_steps
        self.step_scheduler_after = "batch"

    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_parameters, lr=3e-5)
        return optimizer

    def fetch_scheduler(self):
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        )
        return scheduler

    def loss(self, outputs, labels):
        if labels is None:
            return None
        batch_logits = outputs.logits[..., :-1, :].contiguous()
        target_labels = labels[..., 1:].contiguous()
        loss = self._loss(
            batch_logits.view(-1, batch_logits.size(-1)), target_labels.view(-1)
        )
        return loss

    def forward(self, **inputs):
        outputs = self.genmodel(**inputs, return_dict=True)
        loss = outputs.loss
        loss = self.loss(outputs, inputs['labels'])
        return outputs.logits, loss, {}