# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Fixed bug associated with cdist on gpu
# Fixed bug in torch.index_select for subsetting lambdas

# Uses feature extracted from 10K dataset
# Kmer counts computed for >80K large contigs and <=80K chimeric contigs separately


import time
import itertools
import logging
import re
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import sklearn
from sklearn.metrics import accuracy_score



import sys
import math
import copy
import models
import datasets
import losses

import parameter_inits

from utils import *
from weight_inits import *



# Hyper-parameters
#input_size = 32896    # Canonical kmer count for k=8
#input_size = 8192      # Canonical kmer count for k=7
N = 10570             # Number of samples in dataset
#hidden_size_fc1 = 4000
hidden_size_fc1 = 2048
#hidden_size_fc2 = 2000
#embedding_size = 2 ** math.floor(math.log2(10 * N ** (1 / 2)))
embedding_size = 1024
start_epoch = 0
#num_epochs = 8000
batch_size = 16

learning_rate = 0.0001           # 1e-4
learning_rate_decay = 2000
learning_rate_base = 0.1
learning_rate_update_freq = 100


features_scaler = 1e4

train_test_split = 0.95
weight_decay = 1e-5     # L2 regularization
resume = False
seed = 16
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True



#### Dataset parameters ####



params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 1}



def classify_func(features_folder, feature_input, model_file, classification_result):

    since = time.time()

    level = logging.INFO
    format = '%(message)s'
    handlers = [logging.FileHandler(os.path.join(classification_result, 'classification.log'), 'w+'), logging.StreamHandler()]

    #logging.basicConfig(level=logging.NOTSET, format='%(asctime)s | %(levelname)s: %(message)s', handlers=handlers)

    logging.basicConfig(level=level, format=format, handlers=handlers)
    # logging.info('Hey, this is working!')


    #######################################################################
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #######################################################################
    logging.info('\n==> Input arguments...\n')


    logging.info('Feature directory: {}'.format(features_folder))
    logging.info('Model: {}'.format(model_file))


    #######################################################################
    # Prepare dataset
    logging.info('\n==> Preparing Data...\n')


    feature_input = feature_input.iloc[:,:]*features_scaler
    input_size = np.shape(feature_input)[1]
    logging.info("Dimensions of feature matrix rows: {}, cols: {}".format(np.shape(feature_input)[0], np.shape(feature_input)[1]))

    # #######################################################################
    # Get names
    backbone_names = feature_input.index.tolist()

    #######################################################################
    # Model
    logging.info('\n==> Building model...\n')


    ##### Load model #####
    state = torch.load(os.path.join(model_file, "classifier_model.ckpt"))


    input_size = state["model_input_size"]
    hidden_size_fc1 = state["model_hidden_size_fc1"]
    class_count = state["model_class_count"]

    # Need to find a way to save model name as well
    model = models.NeuralNetClassifierOnly(input_size, hidden_size_fc1, class_count)

    model.load_state_dict(state['state_dict'])
    # model.to(device)
    model.to("cpu")


    logging.info('Number of Classes: {}'.format(class_count))

    """
    #######################################################################
    logging.info('\n==> Model parameters----------')
    # for parameter in model.parameters():
    #     logging.info(parameter.shape)

    for name, param in model.named_parameters():
        logging.info("{} : {}".format(name, param.shape))

    # list(model.parameters())[0].grad

    # Total number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logging.info("Total parameters: {}".format(pytorch_total_params))

    # Total number of trainable parameters
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Trainable parameters: {}".format(pytorch_trainable_params))
    """

    #######################################################################
    # Training model
    logging.info('\n==> Compute model output...\n')

    # Compute model output

    model.eval()

    with torch.no_grad():
        model_class = model(torch.from_numpy(feature_input.values).float())


    ps = torch.exp(model_class)
    top_p, top_class = ps.topk(1, dim=1)

    # Get names
    backbone_names = feature_input.index.tolist()
    #print(backbone_names)


    #######################################################################
    # Compute distance matrix for entire set

    # Detach gradient and convert to numpy
    df_classes = pd.DataFrame(np.hstack((top_class.detach().numpy(), top_p.detach().numpy(), ps.detach().numpy())))


    # Attach species names

    df_classes.columns = ["top_class", "top_p"] + [str(x) for x in list(range(class_count))]
    df_classes.insert(loc=0, column='genome', value=backbone_names)


    #######################################################################
    # Generate Apples input



    # Write to file
    logging.info("Dimensions of class output rows:{} cols:{}".format(len(df_classes), len(df_classes.columns)))
    df_classes.to_csv(os.path.join(classification_result, "classes.out"), index=False, sep='\t')



    #######################################################################


    logging.info('\n==> Classification Completed!\n')

    time_elapsed = time.time() - since
    hrs, _min, sec = hms(time_elapsed)
    logging.info('Time: {:02d}:{:02d}:{:02d}'.format(hrs, _min, sec))







