import sys
import numpy  as np
import os, gc
import pickle
import copy
import logging
import threading
from queue import Queue
import collections
import torch
import random
import itertools

logger = logging.getLogger(__name__)

vocab_path = "./meta_data/vocab.pkl"
vocab = pickle.load(open(vocab_path,'rb'))
id_to_word = {v: k for k, v in vocab.items()}



video_input_size = 2048
audio_input_size = 128
max_len = {"video" : 60, "audio" : 20}

END = vocab['<END>']
PAD = vocab['<PAD>']

def process(caption,max_length):
    caption_x  = caption[:-1]
    caption_y = caption[1:]
    if len(caption_x) > max_length:
        caption_x = caption[:max_length]
        caption_y = caption[1:max_length]
        caption_y.append(END)
    return (caption_x,caption_y)

def convert_to_batch(features, modality="video"):
    batch = np.zeros((len(features), max_len[modality], features[0][0].shape[0]))
    for counter in range(len(features)):
        batch[counter,:min(max_len[modality],len(features[counter])),:] = features[counter][:max_len[modality]]
    return batch

class SSFetcher(threading.Thread):
    def __init__(self, parent):
        threading.Thread.__init__(self)
        self.parent = parent
        self.video_files = self.parent.video_list
        self.indices = list(range(len(self.video_files)))
        self.caption_inds = list(range(min(20,parent.captions_per_vid)))
        self.indices = list(itertools.product(self.indices,self.caption_inds))
        assert len(self.indices) == parent.num_data_points
        np.random.shuffle(self.indices)

    def run(self):
        diter = self.parent
        offset = 0
        i = 0
        while not diter.exit_flag:
            last_batch = False
            counter = 0
            vid_features = []
            aud_features = []
            caption_x_batch = PAD*np.ones((diter.batch_size, diter.max_caption_length))
            caption_y_batch = PAD*np.ones((diter.batch_size, diter.max_caption_length))
            mask = np.zeros((diter.batch_size, diter.max_caption_length))
            video_ids = []
            while counter < diter.batch_size:
                if offset == diter.num_data_points:
                    last_batch = True
                    diter.queue.put(None)
                    return
                index = self.indices[offset]
                video_file = self.video_files[index[0]]
                caption = diter.vid_to_caption_map[video_file][index[1]]
                video_features = np.load(os.path.join(diter.video_root_dir, video_file+".npy"))
                audio_features = np.load(os.path.join(diter.audio_root_dir, video_file+".npy"))
                caption_x,caption_y = process(caption,diter.max_caption_length)
                vid_features.append(video_features)
                aud_features.append(audio_features)
                caption_x_batch[counter, :len(caption_x)] = caption_x
                caption_y_batch[counter, :len(caption_x)] = caption_y
                mask[counter, :len(caption_x)] = 1
                video_ids.append(video_file)
                counter += 1
                offset += 1

            if counter == diter.batch_size:
                vid_batch = convert_to_batch(vid_features, modality="video")
                aud_batch = convert_to_batch(aud_features, modality="audio")
                batch = {}
                batch["audio_features"] = torch.from_numpy(aud_batch).type(torch.float32)
                batch["visual_features"] = torch.from_numpy(vid_batch).type(torch.float32)
                batch["caption_x"] = torch.from_numpy(caption_x_batch).type(torch.long)
                batch["caption_y"] = torch.from_numpy(caption_y_batch).type(torch.long)
                batch["mask"] = torch.from_numpy(mask).type(torch.float32)
                batch["video_ids"] = video_ids
                if diter.device.type == "cuda":
                    for key in batch:
                        if type(batch[key]) != list:
                            batch[key] = batch[key].cuda()
                diter.queue.put(batch)
                i+=1

            if last_batch:
                diter.queue.put(None)
                return

class SSIterator(object):
    def __init__(self,
                 batch_size,
                 max_caption_length,
                 captions_per_vid,
                 mode,
                 device,
                 max_videos=-1):

        self.batch_size = batch_size
        self.max_caption_length = max_caption_length
        self.exit_flag = False
        self.use_infinite_loop = False
        self.mode = mode
        self.device = device
        self.captions_per_vid = captions_per_vid
        self.max_videos = max_videos
        self.load_files()

    def load_files(self):
        caption_file = "./meta_data/{}_vid_to_caption.pkl".format(self.mode)
        video_list_file = "./meta_data/{}_vid_ids.pkl".format(self.mode)

        if self.mode == "test":
            self.video_root_dir = "/exp/vishwajeet/test/Video_features/"
            self.audio_root_dir = "/exp/data/audio-visual/MSR-VTT_features/test/Audio_features/"
        else:
            self.video_root_dir = "/exp/vishwajeet/train/Video_features/"
            self.audio_root_dir = "/exp/data/audio-visual/MSR-VTT_features/Audio-Features/"

        with open(caption_file, "rb") as file:
            self.vid_to_caption_map = pickle.load(file)

        with open(video_list_file, "rb") as file:
            self.video_list = pickle.load(file)
        if self.max_videos != -1:
            self.video_list = self.video_list[:self.max_videos]
        self.num_data_points = len(self.video_list)*min(self.captions_per_vid, 20)

    def start(self):
        self.exit_flag = False
        self.queue = Queue(maxsize = 5)
        self.gather = SSFetcher(self)
        self.gather.daemon = True
        self.gather.start()

    def __del__(self):
        if hasattr(self, 'gather'):
            self.gather.exitFlag = True
            self.gather.join()

    def __iter__(self):
        return self

    def next(self):
        if self.exit_flag:
            return None

        batch = self.queue.get()
        if not batch:
            self.exit_flag = True
        return batch
