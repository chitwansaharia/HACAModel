import torch
import pickle
import json
import sys
from development_model import *
from model import *
import os
from data_iterator import *
from evaluate import *

vocab_path = "./meta_data/vocab.pkl"
vocab = pickle.load(open(vocab_path,'rb'))
id_to_word = {v: k for k, v in vocab.items()}

batch_size = 64
max_caption_length = 15

device = torch.device("cuda")

model_name = sys.argv[1]
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

haca_model =  HACAModel(config, device, batch_size, max_caption_length)

if "{}.pt".format(model_name) in os.listdir('models'):
    data = torch.load('models/{}.pt'.format(model_name))
    haca_model.load_state_dict(data["state_dict"])

haca_model.cuda()

mode = sys.argv[2]

iterator = SSIterator(batch_size, max_caption_length, 1, mode, device)
with open('./meta_data/cider_jsons/gts_{}.pkl'.format(mode),'rb') as file:
    gts = pickle.load(file)
vid_to_caption_map = iterator.vid_to_caption_map

iterator.start()
batch = iterator.next()
res = []
haca_model.eval()
ss_epsilon = 0

while batch != None:
    # output_captions obtained from model
    output_caption = haca_model(batch, ss_epsilon)
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
import ipdb; ipdb.set_trace()
res = {"annotations" : res}

metrics,element_wise_metric = calculate_metrics(gts,res)

if mode == "test":
    with open('/exp/data/audio-visual/MSR-VTT/data/test_videodatainfo_2017.json') as file:
        data = json.load(file)
else:
    with open('/exp/data/audio-visual/MSR-VTT/data/videodatainfo_2017.json') as file:
        data = json.load(file)

temp_dict = {}

for item in data['videos']:
    temp_dict[item['video_id']] = item['category']

cat_wise_count, cat_wise_score = [0]*20, [0]*20


for file in element_wise_metric:
    cat_wise_score[int(temp_dict[file])] += element_wise_metric[file]["CIDEr"]
    cat_wise_count[int(temp_dict[file])] += 1

for index in range(20):
    cat_wise_score[index] /= cat_wise_count[index]

import ipdb; ipdb.set_trace()
print(cat_wise_score)



