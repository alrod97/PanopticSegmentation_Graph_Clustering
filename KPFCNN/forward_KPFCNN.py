import signal
import os
import numpy as np
import sys
import torch

from SemanticKitti import *
from torch.utils.data import DataLoader

from config_KPFCNN import Config
# from utils.tester import ModelTester
from KPFCNN import *

# from test_models import model_choice


def loop_through_batches(net, data_loader):
    t = [time.time()]
    processed = 0  # number of frames that processed
    while True:
        print('Initialize workers')
        for i, batch in enumerate(data_loader):
            # New time
            order = data_loader.dataset.rand_order
            t = t[-1:]
            t += [time.time()]
            if i == 4:  # xxx remove
                return

            if i == 0:
                print('Done in {:.1f}s'.format(t[1] - t[0]))

            flag = True
            if config.n_test_frames > 1:
                lengths = batch.lengths[0].cpu().numpy()
                for b_i, length in enumerate(lengths):
                    f_inds = batch.frame_inds.cpu().numpy()
                    f_ind = f_inds[b_i, 1]
                    if f_ind % config.n_test_frames != config.n_test_frames - 1:
                        flag = False

            if processed == data_loader.dataset.all_inds.shape[0]:
                return
            # if not flag:
            #    continue
            # else:
            processed += 1

            if 'cuda' in device.type:
                batch.to(device)

            with torch.no_grad():

                outputs, centers_output, var_output, embedding = net(batch, config)

                yield outputs, centers_output, var_output, embedding


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# choose pretrained model
chosen_log = "KPFCNN/Log_2020-10-06_16-51-05"
# chosen_log = model_choice(chosen_log)


config = Config()
config.load(chosen_log)

config.global_fet = False
config.validation_size = 200
config.input_threads = 16
config.n_frames = 1
config.n_test_frames = 1  # it should be smaller than config.n_frames
if config.n_frames < config.n_test_frames:
    config.n_frames = config.n_test_frames
# xxx
config.big_gpu = False
config.dataset_task = '4d_panoptic'
# config.sampling = 'density'
config.sampling = 'importance'
config.decay_sampling = 'None'
config.stride = 1
config.first_subsampling_dl = 0.061

config.dataset_task == '4d_panoptic'


test_dataset = SemanticKittiDataset(config, set="validation", balance_classes=False, seqential_batch=True)
test_sampler = SemanticKittiSampler(test_dataset)
collate_fn = SemanticKittiCollate

test_loader = DataLoader(test_dataset,
                         batch_size=1,
                         sampler=test_sampler,
                         collate_fn=collate_fn,
                         num_workers=0,  # config.input_threads,
                         pin_memory=True)

test_sampler.calibration(test_loader, verbose=True)

network = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)

if config.big_gpu and torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
network.to(device)

pred = []
cen = []
var = []
emb = []
for outputs, centers_output, var_output, embedding in loop_through_batches(network, test_loader):
    pred.append(outputs)
    cen.append(centers_output)
    var.append(var_output)
    emb.append(embedding)

print(len(emb))
print(emb[0].shape)
