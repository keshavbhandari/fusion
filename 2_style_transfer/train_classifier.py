from torch.cuda import is_available as cuda_available, is_bf16_supported
from torch.backends.mps import is_available as mps_available
import yaml
import json
import pickle
import os
import random
import torch
from torch import Tensor, argmax
from transformers import AutoModelForSequenceClassification, BertConfig, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from evaluate import load as load_metric
from data_loader import Genre_Classifier_Dataset
import sys
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath("configs/configs_style_transfer.yaml"),
                    help="Path to the config file")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configs = yaml.safe_load(f)

batch_size = configs['training']['classifier']['batch_size']
learning_rate = configs['training']['classifier']['learning_rate']
epochs = configs['training']['classifier']['epochs']

# Artifact folder
artifact_folder = configs['raw_data']['artifact_folder']
# Load encoder tokenizer json file dictionary
tokenizer_filepath = os.path.join(artifact_folder, "style_transfer", "vocab_corrupted.pkl")
# Load the tokenizer dictionary
with open(tokenizer_filepath, "rb") as f:
    tokenizer = pickle.load(f)

    
# Open the train, validation, and test sets files
with open(os.path.join(artifact_folder, "style_transfer", "fine_tuning_train.pkl"), "rb") as f:
    train_sequences = pickle.load(f)
with open(os.path.join(artifact_folder, "style_transfer", "fine_tuning_valid.pkl"), "rb") as f:
    valid_sequences = pickle.load(f)

# Print length of train, validation, and test sets
print("Length of train set: ", len(train_sequences))
print("Length of validation set: ", len(valid_sequences))

# Load the dataset
train_dataset = Genre_Classifier_Dataset(configs, train_sequences, mode="train", shuffle=True)
valid_dataset = Genre_Classifier_Dataset(configs, valid_sequences, mode="eval", shuffle=False)

# Get the vocab size
vocab_size = len(tokenizer)+1
# Get the data length
train_length = len(train_dataset.data_list)

# Create the encoder-decoder model
config_encoder = BertConfig()
config_encoder.vocab_size = vocab_size
config_encoder.max_position_embeddings = configs['classifier_model']['encoder_max_sequence_length']
config_encoder.max_length = configs['classifier_model']['encoder_max_sequence_length']
config_encoder.pad_token_id = 0
config_encoder.bos_token_id = tokenizer["<S>"]
config_encoder.eos_token_id = tokenizer["<E>"]
config_encoder.num_hidden_layers = configs['classifier_model']['encoder_num_layers']
config_encoder.num_attention_heads = configs['classifier_model']['encoder_num_heads']
config_encoder.hidden_size = configs['classifier_model']['encoder_hidden_size']
config_encoder.intermediate_size = configs['classifier_model']['encoder_intermediate_size']
config_encoder.hidden_dropout_prob = configs['classifier_model']['encoder_dropout']
config_encoder.attention_probs_dropout_prob = configs['classifier_model']['encoder_dropout']
config_encoder.num_labels = 3

model = AutoModelForSequenceClassification.from_config(config_encoder)
print("Created new model")

# Print the number of parameters in the model
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params}")

# Create config for the Trainer
USE_CUDA = cuda_available()
print(f"USE_CUDA: {USE_CUDA}")
if not cuda_available():
    FP16 = FP16_EVAL = BF16 = BF16_EVAL = False
elif is_bf16_supported():
    BF16 = BF16_EVAL = True
    FP16 = FP16_EVAL = False
else:
    BF16 = BF16_EVAL = False
    FP16 = FP16_EVAL = True
USE_MPS = not USE_CUDA and mps_available()

metrics = {metric: load_metric(metric) for metric in ["accuracy"]}

# Define the accuracy metric function
def compute_metrics(p):
    preds = torch.argmax(torch.from_numpy(p.predictions), axis=1)
    return {
        'accuracy': accuracy_score(p.label_ids, preds),
        # 'precision': precision_recall_fscore_support(p.label_ids, preds, average='binary')[0],
        # 'recall': precision_recall_fscore_support(p.label_ids, preds, average='binary')[1],
        # 'f1': precision_recall_fscore_support(p.label_ids, preds, average='binary')[2],
    }

run_name = configs['training']['classifier']['run_name']
model_dir = os.path.join(artifact_folder, "style_transfer", run_name)
log_dir = os.path.join(model_dir, "logs")
# Clear the logs directory before training
os.system(f"rm -rf {log_dir}")

# Define the training arguments
training_args = TrainingArguments(
    output_dir=model_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_strategy="epoch",  # "steps" or "epoch"
    save_total_limit=1,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio=0.3,
    max_grad_norm=3.0,
    weight_decay= configs['training']['classifier']['weight_decay'],
    num_train_epochs=epochs,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=configs['training']['classifier']['gradient_accumulation_steps'],
    gradient_checkpointing=True,
    optim="adafactor",
    seed=444,
    logging_strategy="steps",
    logging_steps=10,
    logging_dir=log_dir,
    no_cuda=not USE_CUDA,
    fp16=FP16,
    fp16_full_eval=FP16_EVAL,
    bf16=BF16,
    bf16_full_eval=BF16_EVAL,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to="tensorboard",
    run_name=run_name,
    push_to_hub=False,
    dataloader_num_workers=5
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=30)]
)

# Train and save the model
train_result = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()
