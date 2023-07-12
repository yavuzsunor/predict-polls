import os
import sys
import random
import time
from dataclasses import dataclass

import pandas as pd
import numpy as np
from transformers import  AutoModelForSequenceClassification, AdamW, AutoConfig, get_linear_schedule_with_warmup

import tensorflow as tf
import torch

from src.exception import CustomException
from src.logger import logging
from src.utils import find_device, flat_accuracy, format_time

# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
config = AutoConfig.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=5)
model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", config=config)
# model.cuda() # when GPU is accessable

# assign GPU as device if available 
device = find_device()

@dataclass
class ModelTrainerConfig:
    raw_data_path = 'artifacts/data/tweets_multiple_label'
    model_data_path = 'artifacts/model/BERT_multi_classifier/bert-classifier-turkish-sentiment'

@dataclass
class ClassifierModelFinetune: 
    def train_model(self, train_dataloader, validation_dataloader):
        try: 
            # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
            # 'W' stands for 'Weight Decay fix"
            optimizer = AdamW(model.parameters(),
                              lr = 2e-5, # args.learning_rate - default is 5e-5
                              betas=[0.9,0.999],
                              eps = 1e-6 # args.adam_epsilon  - default is 1e-8.
                            )

            # Number of training epochs
            epochs = 5

            # Total number of training steps is number of batches * number of epochs.
            total_steps = len(train_dataloader) * epochs

            # Create the learning rate scheduler.
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

            
            # Training loop
            # This training code is based on the `run_glue.py` script here:
            # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128


            # Set the seed value all over the place to make this reproducible.
            seed_val = 42

            random.seed(seed_val)
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)
            torch.cuda.manual_seed_all(seed_val)

            # Store the average loss after each epoch so we can plot them.
            loss_values = []

            # For each epoch...
            logging.info("Starting model training...")
            for epoch_i in range(0, epochs):

                # ========================================
                #               Training
                # ========================================

                # Perform one full pass over the training set.

                print("")
                print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
                print('Training...')

                # Measure how long the training epoch takes.
                t0 = time.time()

                # Reset the total loss for this epoch.
                total_loss = 0

                # Put the model into training mode. Don't be mislead--the call to
                # `train` just changes the *mode*, it doesn't *perform* the training.
                # `dropout` and `batchnorm` layers behave differently during training
                # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
                model.train()

                # For each batch of training data...
                for step, batch in enumerate(train_dataloader):

                    # Progress update every 30 batches.
                    if step % 30 == 0 and not step == 0:
                        # Calculate elapsed time in minutes.
                        elapsed = format_time(time.time() - t0)

                        # Report progress.
                        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                    # Unpack this training batch from our dataloader.
                    #
                    # As we unpack the batch, we'll also copy each tensor to the GPU using the
                    # `to` method.
                    #
                    # `batch` contains three pytorch tensors:
                    #   [0]: input ids
                    #   [1]: attention masks
                    #   [2]: labels
                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)
                    b_labels = batch[2].to(device)

                    # Always clear any previously calculated gradients before performing a
                    # backward pass. PyTorch doesn't do this automatically because
                    # accumulating the gradients is "convenient while training RNNs".
                    # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                    model.zero_grad()

                    # Perform a forward pass (evaluate the model on this training batch).
                    # This will return the loss (rather than the model output) because we
                    # have provided the `labels`.
                    # The documentation for this `model` function is here:
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    outputs = model(b_input_ids,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)

                    # The call to `model` always returns a tuple, so we need to pull the
                    # loss value out of the tuple.
                    loss = outputs[0]

                    # Accumulate the training loss over all of the batches so that we can
                    # calculate the average loss at the end. `loss` is a Tensor containing a
                    # single value; the `.item()` function just returns the Python value
                    # from the tensor.
                    total_loss += loss.item()

                    # Perform a backward pass to calculate the gradients.
                    loss.backward()

                    # Clip the norm of the gradients to 1.0.
                    # This is to help prevent the "exploding gradients" problem.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # Update parameters and take a step using the computed gradient.
                    # The optimizer dictates the "update rule"--how the parameters are
                    # modified based on their gradients, the learning rate, etc.
                    optimizer.step()

                    # Update the learning rate.
                    scheduler.step()

                # Calculate the average loss over the training data.
                avg_train_loss = total_loss / len(train_dataloader)

                # Store the loss value for plotting the learning curve.
                loss_values.append(avg_train_loss)

                print("")
                print("  Average training loss: {0:.2f}".format(avg_train_loss))
                print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

                # ========================================
                #               Validation
                # ========================================
                # After the completion of each training epoch, measure our performance on
                # our validation set.

                print("")
                print("Running Validation...")

                t0 = time.time()

                # Put the model in evaluation mode--the dropout layers behave differently
                # during evaluation.
                model.eval()

                # Tracking variables
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0

                # Evaluate data for one epoch
                for batch in validation_dataloader:

                    # Add batch to GPU
                    batch = tuple(t.to(device) for t in batch)

                    # Unpack the inputs from our dataloader
                    b_input_ids, b_input_mask, b_labels = batch

                    # Telling the model not to compute or store gradients, saving memory and
                    # speeding up validation
                    with torch.no_grad():

                        # Forward pass, calculate logit predictions.
                        # This will return the logits rather than the loss because we have
                        # not provided labels.
                        # token_type_ids is the same as the "segment ids", which
                        # differentiates sentence 1 and 2 in 2-sentence tasks.
                        # The documentation for this `model` function is here:
                        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                        outputs = model(b_input_ids,
                                        attention_mask=b_input_mask,
                                        )

                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    logits = outputs[0]

                    # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()

                    # Calculate the accuracy for this batch of test sentences.
                    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

                    # Accumulate the total accuracy.
                    eval_accuracy += tmp_eval_accuracy

                    # Track the number of batches
                    nb_eval_steps += 1

                # Report the final accuracy for this validation run.
                print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
                print("  Validation took: {:}".format(format_time(time.time() - t0)))

            print("")
            print("Training complete!")
            logging.info("Finished model training")
            return model

        except Exception as e:
            raise CustomException(e, sys)