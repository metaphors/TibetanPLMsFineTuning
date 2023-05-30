from transformers import XLMRobertaForSequenceClassification
from transformers import TrainingArguments
from datasets import load_dataset
from transformers import XLMRobertaTokenizer
from evaluate import combine
from numpy import argmax
from transformers import Trainer

model = XLMRobertaForSequenceClassification.from_pretrained("../model/cino-base-v2", num_labels=2)

training_args = TrainingArguments(
    output_dir="../saved_model/cino-base-v2_tusa",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=40,
    learning_rate=5e-5,
    warmup_ratio=0.1
)

dataset = load_dataset("../dataset_loader/tusa.py")

tokenizer = XLMRobertaTokenizer.from_pretrained("../model/cino-base-v2")


def tokenize(examples):
    return tokenizer(examples["text"], max_length=512, padding="max_length", truncation=True)


tokenized_dataset = dataset.map(tokenize, batched=True)

clf_metrics = combine(["accuracy", "precision", "recall", "f1"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = argmax(logits, axis=-1)
    return clf_metrics.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].shuffle(),
    eval_dataset=tokenized_dataset["validation"].shuffle(),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
