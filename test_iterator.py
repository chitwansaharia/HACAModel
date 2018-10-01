# Helper code used for testing and seeing the outputs from the iterator.
from data_iterator import *
import torch
device = torch.device("cpu")
iterator = SSIterator(64, 15, 20,"test", device,max_videos=200)
iterator.start()
batch = iterator.next()
counter = 0
while batch != None:
    counter+=1
    batch = iterator.next()
    print(counter)
print(counter)