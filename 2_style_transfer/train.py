from torch.cuda import is_available as cuda_available, is_bf16_supported
from torch.backends.mps import is_available as mps_available
import yaml
import json
import pickle
import os
import random
import torch
from torch import Tensor, argmax
from transformers import EncoderDecoderModel, EncoderDecoderConfig, BertConfig, Trainer, TrainingArguments, EarlyStoppingCallback
from evaluate import load as load_metric
from data_loader import Fusion_Dataset
import sys
import argparse
from accelerate import Accelerator

accelerator = Accelerator()

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath("configs/configs_style_transfer.yaml"),
                    help="Path to the config file")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configs = yaml.safe_load(f)

# Artifact folder
artifact_folder = configs['raw_data']['artifact_folder']
# Load encoder tokenizer json file dictionary
tokenizer_filepath = os.path.join(artifact_folder, "style_transfer", "vocab_corrupted.pkl")
# Load the tokenizer dictionary
with open(tokenizer_filepath, "rb") as f:
    tokenizer = pickle.load(f)

run_pretraining = configs['training']['pretraining']['run_pretraining']  
if run_pretraining:
    # Open the train, validation, and test sets files
    with open(os.path.join(artifact_folder, "style_transfer", "pre_training_train.pkl"), "rb") as f:
        train_sequences = pickle.load(f)
    with open(os.path.join(artifact_folder, "style_transfer", "pre_training_valid.pkl"), "rb") as f:
        valid_sequences = pickle.load(f)
else:
    # Open the train, validation, and test sets files
    with open(os.path.join(artifact_folder, "style_transfer", "fine_tuning_train.pkl"), "rb") as f:
        train_sequences = pickle.load(f)
    with open(os.path.join(artifact_folder, "style_transfer", "fine_tuning_valid.pkl"), "rb") as f:
        valid_sequences = pickle.load(f)

# Print length of train, validation, and test sets
print("Length of train set: ", len(train_sequences))
print("Length of validation set: ", len(valid_sequences))

# Load the dataset
train_dataset = Fusion_Dataset(configs, train_sequences, mode="train")
valid_dataset = Fusion_Dataset(configs, valid_sequences, mode="eval")

# Get the vocab size
vocab_size = len(tokenizer)+1
print(f"Vocab size: {vocab_size}")
# Get the data length
train_length = len(train_dataset.data_list)

# Create the encoder-decoder model
config_encoder = BertConfig()
config_encoder.vocab_size = vocab_size
config_encoder.max_position_embeddings = configs['model']['encoder_max_sequence_length']
config_encoder.max_length = configs['model']['encoder_max_sequence_length']
config_encoder.pad_token_id = 0
config_encoder.bos_token_id = tokenizer["<S>"]
config_encoder.eos_token_id = tokenizer["<E>"]
config_encoder.num_hidden_layers = configs['model']['encoder_num_layers']
config_encoder.num_attention_heads = configs['model']['encoder_num_heads']
config_encoder.hidden_size = configs['model']['encoder_hidden_size']
config_encoder.intermediate_size = configs['model']['encoder_intermediate_size']

config_decoder = BertConfig()
config_decoder.vocab_size = vocab_size
config_decoder.max_position_embeddings = configs['model']['decoder_max_sequence_length']
config_decoder.max_length = configs['model']['decoder_max_sequence_length']
config_decoder.bos_token_id = tokenizer["<S>"]
config_decoder.eos_token_id = tokenizer["<E>"]
config_decoder.pad_token_id = 0
config_decoder.num_hidden_layers = configs['model']['decoder_num_layers']
config_decoder.num_attention_heads = configs['model']['decoder_num_heads']
config_decoder.hidden_size = configs['model']['decoder_hidden_size']
config_decoder.intermediate_size = configs['model']['decoder_intermediate_size']

# set decoder config to causal lm
config_decoder.is_decoder = True
config_decoder.add_cross_attention = True
config_decoder.tie_encoder_decoder = False
config_decoder.tie_word_embeddings = False

# Use pretrained model if it exists in the artifact folder to continue training
if not run_pretraining:
    model_path = os.path.join(artifact_folder, "style_transfer", "pre_trained_model")
    model = EncoderDecoderModel.from_pretrained(model_path)    
    print("Loaded model from pre-trained model")
else:
    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
    model = EncoderDecoderModel(config=config)
    config.decoder_start_token_id = tokenizer["<S>"]
    config.pad_token_id = 0
    print("Created new model")

# model.gradient_checkpointing_enable()

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

def compute_metrics(eval_pred):
    """
    Compute metrics for pretraining.

    Must use preprocess_logits function that converts logits to predictions (argmax or sampling).

    :param eval_pred: EvalPrediction containing predictions and labels
    :return: metrics
    """
    predictions, labels = eval_pred
    not_pad_mask = labels != 0
    labels, predictions = labels[not_pad_mask], predictions[not_pad_mask]
    return metrics["accuracy"].compute(predictions=predictions.flatten(), references=labels.flatten())

def preprocess_logits(logits: Tensor, _: Tensor) -> Tensor:
    """
    Preprocess the logits before accumulating them during evaluation.

    This allows to significantly reduce the memory usage and make the training tractable.
    """
    pred_ids = argmax(logits[0], dim=-1)  # long dtype
    return pred_ids

if run_pretraining:
    run_name = configs['training']['pretraining']['run_name']
    weight_decay = configs['training']['pretraining']['weight_decay']
    gradient_accumulation_steps = configs['training']['pretraining']['gradient_accumulation_steps']
    batch_size = configs['training']['pretraining']['batch_size']
    learning_rate = configs['training']['pretraining']['learning_rate']
    epochs = configs['training']['pretraining']['epochs']
    warmup_ratio = configs['training']['pretraining']['warmup_ratio']
else:
    run_name = configs['training']['fine_tuning']['run_name']
    weight_decay = configs['training']['fine_tuning']['weight_decay']
    gradient_accumulation_steps = configs['training']['fine_tuning']['gradient_accumulation_steps']
    batch_size = configs['training']['fine_tuning']['batch_size']
    learning_rate = configs['training']['fine_tuning']['learning_rate']
    epochs = configs['training']['fine_tuning']['epochs']
    warmup_ratio = configs['training']['fine_tuning']['warmup_ratio']

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
    save_total_limit=2,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio=warmup_ratio,
    max_grad_norm=3.0,
    weight_decay= weight_decay,
    num_train_epochs=epochs,
    eval_strategy="epoch",
    gradient_accumulation_steps=gradient_accumulation_steps,
    # gradient_checkpointing=True,
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
    # metric_for_best_model="loss",
    greater_is_better=False,
    report_to="tensorboard",
    run_name=run_name,
    push_to_hub=False,
    dataloader_num_workers=5,
    ddp_find_unused_parameters=True
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits,
    # data_collator=collate_fn,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=30)]
)

# Train and save the model
print("Training the model")
train_result = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()
