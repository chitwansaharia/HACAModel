import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from model import *
from utils import *
from data_iterator import *
from tensorboardX import SummaryWriter
import argparse
import time
import datetime
from evaluate import calculate_metrics

# Vocabulary (Used while testing)
vocab_path = "./meta_data/vocab.pkl"
vocab = pickle.load(open(vocab_path,'rb'))
id_to_word = {v: k for k, v in vocab.items()}

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=20,
                    help="batch size for training (default: 20)")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate for training (default: 1e-3)")
parser.add_argument("--model", required=True,
                    help="model name to save (required)")
parser.add_argument("--epochs", type=int, default=20,
                    help="number of epochs to be trained (default: 20)")
parser.add_argument("--log-interval", type=int, default=10,
                    help="log interval (default: 10)")
parser.add_argument("--ss", type=float, default=None,
                    help="starting epsilon for scheduled sampling (default: None)")
parser.add_argument("--max-grad-norm", type=float, default=10,
                    help="maximum gradient clipping norm")
parser.add_argument("--captions-per-vid", type=int, default=20,
                    help="number of captions per video for training")
parser.add_argument("--optimizer", default="adadelta",
                    help="type of optimizer to use for training")
parser.add_argument("--max-videos", default=-1, type=int,
                    help="maximum number of videos in the training set (default: -1 -> use all the videos)")

args = parser.parse_args()

# general parameters
batch_size = args.batch_size
max_caption_length = 15
lr = args.lr
# Used for Adam Optimizer
optim_eps = 1e-5

model_name = args.model

# Scheduled sampling parameter (decides whether to teacher force, or auto regress) (None or 0 means complete teacher forcing)
ss_epsilon = None

# All configurations used for training
config = {
    "audio_input_size" : 128,
    "visual_input_size" : 2048,
    "chunk_size" : {"audio" : 4, "visual" : 10}, # Granularity of high level encoder
    "audio_hidden_size" : {"low" : 128, "high" : 64},
    "visual_hidden_size" : {"low" : 512, "high" : 256},
    "local_input_size" : 1024,
    "global_input_size" : 1024,
    "global_hidden_size" : 256,
    "local_hidden_size" : 1024,
    "vocab_size" : 10004,
    "embedding_size" : 512
    }

# Tensorboard writer to write the summary
writer = SummaryWriter("logs/{}".format(model_name))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Cross entropy loss
criterion = nn.CrossEntropyLoss(reduce=False)


''' Runs one epoch '''
''' Params :
        epoch : The epoch number
        mode : train/valid (dataset to use)
        log_number : used for logging
'''
def run_epoch(epoch, mode, log_number):
    # Loads the iterator, if in training mode, and max_video != -1, only that much videos will be loaded for training.
    # However, for validation, all the videos will be loaded.
    if mode == "train":
        iterator = SSIterator(batch_size, max_caption_length, args.captions_per_vid, mode, device, args.max_videos)
    else:
        iterator = SSIterator(batch_size, max_caption_length, args.captions_per_vid, mode, device)

    if mode == "train":
        print("Training")
    else:
        haca_model.eval()
        print("Validating")

    iterator.start()
    total_loss, num_batch, total_grad_norm = 0,0,0
    batch = iterator.next()
    ss_epsilon = None
    # The scheduled sampling parameter decays after each epoch. ss_t = ss_(t-1)*ss_0
    if args.ss is not None:
        ss_epsilon = args.ss**epoch
    # If in validation mode, shifting to teacher forcing
    if ss_epsilon is not None and mode != "train":
        ss_epsilon = 1

    while batch != None:
        # output_captions obtained from haca_model
        output_caption = haca_model(batch, ss_epsilon)
        output_caption = torch.cat(output_caption, dim=1)

        loss = criterion(output_caption.view(-1,config["vocab_size"]), batch["caption_y"].view(-1))
        # Loss multiplied with mask
        loss = loss*batch["mask"].view(-1)
        numele = torch.sum(batch["mask"]).item()
        loss = torch.sum(loss)/(numele)

        # Doing backprop, and computing grad norm, clipping gradients
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            grad_norm = sum(p.grad.data.norm(2) ** 2 for p in haca_model.parameters() if p.grad is not None) ** 0.5
            total_grad_norm += grad_norm.item()
            torch.nn.utils.clip_grad_norm_(haca_model.parameters(), args.max_grad_norm)
            optimizer.step()

        total_loss += loss.item()
        num_batch += 1

        # Logging
        if num_batch % args.log_interval == 0:
            total_ellapsed_time = int(time.time() - total_start_time)
            duration = datetime.timedelta(seconds=total_ellapsed_time)
            print("D {} | E {} | B {} | Loss {} | G {}".format(duration, epoch, num_batch, total_loss/num_batch, total_grad_norm/num_batch))
            if mode == "train":
                writer.add_scalar("Training Loss", total_loss/num_batch, log_number)
                writer.add_scalar("Gradient Mag", total_grad_norm/num_batch, log_number)
                writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], log_number)
                log_number += 1

        batch = iterator.next()
    # Putting the model back on train mode in case it was on eval mode
    haca_model.train()
    # Logging the epoch parameters
    if mode == "train":
        writer.add_scalar("Epoch Training Loss", total_loss/num_batch, epoch)
    else:
        writer.add_scalar("Epoch Validation Loss", total_loss/num_batch, epoch)
    return log_number, total_loss/num_batch

''' Tests the model '''
''' Params :
        model : The model to test
        mode : train/valid/test (dataset to use)
'''
def test_model(model, mode):
    print("Testing the model")
    # Loads the iterator, if in training mode, and max_video != -1, only that much videos will be loaded for training.
    # However, for validation/testing, all the videos will be loaded.
    if mode == "train":
        iterator = SSIterator(batch_size, max_caption_length, 1, mode, device, args.max_videos)
    else:
        iterator = SSIterator(batch_size, max_caption_length, 1, mode, device)
    # Loading the caption ground truths to test the model
    with open('./meta_data/cider_jsons/gts_{}.pkl'.format(mode),'rb') as file:
        gts = pickle.load(file)
    # Video to caption map
    vid_to_caption_map = iterator.vid_to_caption_map
    iterator.start()

    # ss_epsilon = 0 during testing (Completely autoregressive model)
    batch = iterator.next()
    ss_epsilon = 0
    res = []

    while batch != None:
        # output_captions obtained from model
        output_caption = model(batch, ss_epsilon)
        output_caption = torch.cat(output_caption, dim=1)
        output_caption = torch.max(output_caption, dim=2, keepdim=False)[1]

        # For each of the batch element
        for index in range(batch_size):
            # Corresponding video id
            video_id = batch["video_ids"][index]
            # Converting output caption to words (from the vocabulary)
            candidate = [id_to_word[id.item()] for id in output_caption[index,:]]
            # If the caption itself ended, then truncating the caption after that, otherwise, taking the entire caption
            if "<END>" in candidate:
                length = candidate.index("<END>") + 1
            else:
                length = max_caption_length
            candidate = candidate[:length]
            candidate = " ".join(candidate)
            # Appending the output to the res list
            res.append({"image_id" : video_id, "caption" : candidate})

        batch = iterator.next()
    res = {"annotations" : res}

    # Calculates relevant metrices
    metrics = calculate_metrics(gts,res)
    model.train()
    return metrics


if __name__ == "__main__":
    # Created the main model
    haca_model = HACAModel(config, device, batch_size, max_caption_length)
    if device.type == "cuda":
        haca_model.cuda()
    # Initialized the model parameters with uniform distribution
    for parameter in haca_model.parameters():
        torch.nn.init.uniform_(parameter, -0.08, 0.08)
    print("Starting Training")
    print(args)
    # Choosing the gradient optimizer
    if args.optimizer == "adadelta":
        optimizer = torch.optim.Adadelta(haca_model.parameters(), lr)
    else:
        optimizer = torch.optim.Adam(haca_model.parameters(), lr)
    # Scheduler to decay gradients
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    log_number = 0
    epochs = args.epochs
    patience = 0
    best_metric = 0
    total_start_time = time.time()
    total_train_loss_history = []

    for epoch_num in range(epochs):
        # If patience goes to 4, the learning rate is decayed by half
        if patience == 4:
            scheduler.step()
            patience = 0
        # Training and Validation epochs
        log_number, total_train_loss = run_epoch(epoch_num, "train", log_number)
        _, total_valid_loss = run_epoch(epoch_num, "valid", None)

        total_train_loss_history.append(total_train_loss)

        # Evaluating the model for all the three datasets
        for mode in ["train", "test", "valid"]:
            metrics = test_model(haca_model, mode)
            for key in metrics:
                writer.add_scalar("{}_{}".format(mode, key), metrics[key], epoch_num)
            if mode == "valid":
                valid_cider_score = metrics["CIDEr"]
            print(metrics)

        # Using valid cider score to determine the patience, and saving the model
        if valid_cider_score > best_metric:
            best_metric = valid_cider_score
            print("Saving the model")
            torch.save({
              'epoch': epoch_num,
              'args': args,
              'state_dict': haca_model.state_dict(),
              'loss_history': total_train_loss_history,
            }, "models/{}".format(args.model))
            patience = 0
        else:
            patience += 1