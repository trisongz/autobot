# autobot: a simple pytorch trainer

Tis a fork from tez

### Training using Autobot:

- To train a model, define a dataset and model. The dataset class is the same old class you would write when writing pytorch models.

- Create your model class. Instead of inheriting from `nn.Module`, import auto and inherit from `auto.Model` as shown in the following example.


```python
class MyModel(auto.Model):
    def __init__(self):
        super().__init__()
        .
        .
        # tell when to step the scheduler
        self.step_scheduler_after="batch"

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def fetch_scheduler(self):
        # create your own scheduler

    def fetch_optimizer(self):
        # create your own optimizer

    def forward(self, ids, mask, token_type_ids, targets=None):
        _, o_2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        b_o = self.bert_drop(o_2)
        output = self.out(b_o)

        # calculate loss here
        loss = nn.BCEWithLogitsLoss()(output, targets)

        # calculate the metric dictionary here
        metric_dict = self.monitor_metrics(output, targets)
        return output, loss, metric_dict
```

Everything is super-intuitive!

- Now you can train your model!

```python
# init datasets
train_dataset = SomeTrainDataset()
valid_dataset = SomeValidDataset()

# init model
model = MyModel()


# init callbacks, you can also write your own callback
tb_logger = auto.callbacks.TensorBoardLogger(log_dir=".logs/")
es = auto.callbacks.EarlyStopping(monitor="valid_loss", model_path="model.bin")

# train model. a familiar api!
model.fit(
    train_dataset,
    valid_dataset=valid_dataset,
    train_bs=32,
    device="cuda",
    epochs=50,
    callbacks=[tb_logger, es],
    fp16=True,
)

# save model (with optimizer and scheduler for future!)
model.save("model.bin")
```

You can checkout examples in `examples/`
