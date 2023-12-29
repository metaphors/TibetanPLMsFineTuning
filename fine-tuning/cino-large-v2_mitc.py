from transformers import XLMRobertaForSequenceClassification
from transformers import TrainingArguments
from datasets import load_dataset
from transformers import XLMRobertaTokenizer
from transformers import DataCollatorWithPadding
from evaluate import load
from numpy import argmax
from transformers import Trainer

id2label = {0: "国内新闻", 1: "国际新闻", 2: "常识", 3: "教育", 4: "文化", 5: "文学", 6: "旅行", 7: "民俗", 8: "法律", 9: "经济", 10: "藏区新闻"}
label2id = {"国内新闻": 0, "国际新闻": 1, "常识": 2, "教育": 3, "文化": 4, "文学": 5, "旅行": 6, "民俗": 7, "法律": 8, "经济": 9, "藏区新闻": 10}

model = XLMRobertaForSequenceClassification.from_pretrained("../model/cino-large-v2", num_labels=11, id2label=id2label, label2id=label2id)

training_args = TrainingArguments(
    output_dir="../saved_model/cino-large-v2_mitc",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_safetensors=False,
    save_only_model=True,
    load_best_model_at_end=True,
    metric_for_best_model="macro-f1",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=40,
    learning_rate=3e-5,
    warmup_ratio=0.1
)

dataset = load_dataset("../dataset_loader/mitc.py")

tokenizer = XLMRobertaTokenizer.from_pretrained("../model/cino-large-v2")


def tokenize(examples):
    return tokenizer(examples["text"], max_length=512, padding="max_length", truncation=True)


tokenized_dataset = dataset.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy_metric = load("accuracy")
precision_metric = load("precision")
recall_metric = load("recall")
f1_metric = load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = argmax(logits, axis=-1)
    accuracy_metric_results = accuracy_metric.compute(predictions=predictions, references=labels)
    macro_precision_metric_results = precision_metric.compute(predictions=predictions, references=labels, average="macro")
    macro_precision_metric_results["macro-precision"] = macro_precision_metric_results.pop("precision")
    macro_recall_metric_results = recall_metric.compute(predictions=predictions, references=labels, average="macro")
    macro_recall_metric_results["macro-recall"] = macro_recall_metric_results.pop("recall")
    macro_f1_metric_results = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    macro_f1_metric_results["macro-f1"] = macro_f1_metric_results.pop("f1")
    weighted_precision_metric_results = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
    weighted_precision_metric_results["weighted-precision"] = weighted_precision_metric_results.pop("precision")
    weighted_recall_metric_results = recall_metric.compute(predictions=predictions, references=labels, average="weighted")
    weighted_recall_metric_results["weighted-recall"] = weighted_recall_metric_results.pop("recall")
    weighted_f1_metric_results = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    weighted_f1_metric_results["weighted-f1"] = weighted_f1_metric_results.pop("f1")
    clf_metrics_results = {**accuracy_metric_results,
                           **macro_precision_metric_results, **macro_recall_metric_results, **macro_f1_metric_results,
                           **weighted_precision_metric_results, **weighted_recall_metric_results, **weighted_f1_metric_results}
    return clf_metrics_results


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].shuffle(),
    eval_dataset=tokenized_dataset["validation"].shuffle(),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
