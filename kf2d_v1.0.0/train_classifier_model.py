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



def train_classifier_model_func(features_folder, feature_input, clades_info, num_epochs, model_filepath):

    since = time.time()

    level = logging.INFO
    format = '%(message)s'
    handlers = [logging.FileHandler('run.log', 'w+'), logging.StreamHandler()]

    #logging.basicConfig(level=logging.NOTSET, format='%(asctime)s | %(levelname)s: %(message)s', handlers=handlers)

    logging.basicConfig(level=level, format=format, handlers=handlers)
    # logging.info('Hey, this is working!')


    #######################################################################
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #######################################################################
    logging.info('\n==> Input arguments...\n')


    logging.info('Feature directory: {}'.format(features_folder))
    logging.info('Clades information: {}'.format(clades_info))



    #######################################################################
    # Input parameters
    logging.info('\n==> Parameters...\n')


    logging.info('GPU Support: {}'.format('Yes' if str(device) != 'cpu' else 'No'))
    logging.info('Hidden Size fc1: {}'.format(hidden_size_fc1))
    # logging.info('Hidden Size fc2: {}'.format(hidden_size_fc2))
    # logging.info('Hidden Size fc3: {}'.format(hidden_size_fc3))
    logging.info('Embedding Size: {}'.format(embedding_size))
    logging.info('Starting Epoch: {}'.format(start_epoch))
    logging.info('Total Epochs: {}'.format(num_epochs))
    logging.info('Batch Size: {}'.format(batch_size))
    #logging.info('Learning Rate: %g', learning_rate)
    logging.info('Random Seed: {}'.format(seed))
    #logging.info('Resuming Training:{}'.format('Yes' if resume else 'No'))


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


    #feature_input = read_feat_mtrx(dataset_features)  # read from dataframe (1000 columns, normalized)
    label_idx_dict, idx_label_dict = get_label_idx_maps(feature_input)    # convert first column into dict

    clade_input = pd.read_csv(clades_info, sep=' ', header=0, index_col=0)  # read from csv

    clade_input_sorted = sort_df(clade_input, label_idx_dict)
    clade_input_sorted = np.concatenate(clade_input_sorted)
    #print(clade_input_sorted)



    # Prepare train/test dataset split
    partition = {}
    partition['train'] = backbone_names
    partition['test'] = []


    # Custom dataset

    training_set = datasets.Dataset(feature_input, partition['train'], label_idx_dict)
    test_set = datasets.Dataset(feature_input, partition['test'], label_idx_dict)
    #val_set = datasets.Dataset(dataset_fname, partition['val'])


    train_size = training_set.__len__()
    test_size = test_set.__len__()
    # val_size = testset.__len__()


    # Data loader
    train_loader = torch.utils.data.DataLoader(training_set, **params)


    if test_size !=0:
        test_loader = torch.utils.data.DataLoader(test_set, **params)
    #val_loader = torch.utils.data.DataLoader(val_set, **params)


    #check whether data are read correctly
    # for i, (data, labels) in enumerate(train_loader):
    #     logging.info(data)
    #     logging.info(labels)


    logging.info('Number of Train Samples: {}'.format(train_size))
    # logging.info('Number of Test Samples: {}'.format(test_size))
    # logging.info('Number of Validation Samples:{}'.format(val_size))


    #######################################################################
    # Model
    logging.info('\n==> Building model...\n')


    class_count = np.unique(clade_input_sorted).size
    logging.info('Number of Classes: {}'.format(class_count))


    model = models.NeuralNetClassifierOnly(input_size, hidden_size_fc1, class_count).to(device)

    # Custom weight initialization
    #model.apply(weight_init)
    #model.to(device)

    logging.info('\n==> Model parameters----------')
    # for parameter in model.parameters():
    #     logging.info(parameter)

    # list(model.parameters())[0].grad

    # Total number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logging.info("Total parameters: {}".format(pytorch_total_params))

    # Total number of trainable parameters
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Trainable parameters: {}".format(pytorch_trainable_params))


    #######################################################################
    # Custom parameter initialization
    #logging.info('\n==> Custom parameter initialization...\n')


    #######################################################################
    # Loss and optimizer
    criterion_b = nn.NLLLoss()
    criterion_b.to(device)


    # Construct optimizer object
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    time_elapsed = time.time() - since
    hrs, _min, sec = hms(time_elapsed)
    logging.info('Time: {:02d}:{:02d}:{:02d}'.format(hrs, _min, sec))


    #######################################################################
    # Training model
    logging.info('\n==> Training model...\n')

    total_step = len(train_loader)


    for epoch in range(num_epochs):

        #######################################################################
        # Train the model
        model.train()
        train_loss2 = 0.0
        running_acc = 0.0

        num_batches = len(train_loader)


        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, input_size).to(device)


            true_class = torch.from_numpy(clade_input_sorted[np.ix_(list(labels))]).to(device)


            # Forward pass
            model_class = model(images.float())


            loss_b = criterion_b(model_class, true_class)
            train_loss2 += loss_b.item()
            loss  = loss_b

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            ps = torch.exp(model_class)
            top_p, top_class = ps.topk(1, dim=1)


            running_acc += accuracy_score(true_class.to('cpu'), top_class.to('cpu'))


        train_loss2 /= num_batches
        running_acc /= num_batches

        time_elapsed = time.time() - since
        hrs, _min, sec = hms(time_elapsed)

        if (i+1) % 1 == 0:
            logging.info('Epoch [{}/{}], Step [{}/{}], Train loss: {:.20f}, {:.20f}, Time: {:02d}:{:02d}:{:02d}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, train_loss2, running_acc, hrs, _min, sec))


        ###########################################################################
        # Test the model

        if test_size != 0:

            model.eval()

            test_loss2 = 0.0
            test_running_acc = 0.0
            num_batches = len(test_loader)

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.reshape(-1, input_size).to(device)


                    true_class = torch.from_numpy(clade_input_sorted[np.ix_(list(labels))]).to(device)


                    model_class = model(images.float())


                    loss_b = criterion_b(model_class, true_class)
                    test_loss2 += loss_b.item()
                    loss = loss_b


                    ps = torch.exp(model_class)
                    top_p, top_class = ps.topk(1, dim=1)

                    test_running_acc += accuracy_score(true_class.to('cpu'), top_class.to('cpu'))


            test_loss2 /= num_batches
            test_running_acc /= num_batches

            time_elapsed = time.time() - since
            hrs, _min, sec = hms(time_elapsed)

            if (i+1) % 1 == 0:
                logging.info('Epoch [{}/{}], Step [{}/{}], Test loss: {:.20f}, {:.20f}, Time: {:02d}:{:02d}:{:02d}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, test_loss2, test_running_acc, hrs, _min, sec))


        # Output current learning rate
        curr_lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {}\t \
              LR:{:.20f}'.format(epoch + 1, curr_lr))

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = curr_lr+0.1



        # Update learning rate
        if (epoch) % learning_rate_update_freq == 0:
            lr = 3e-5 + learning_rate * (learning_rate_base ** (epoch / learning_rate_decay))


            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


    #######################################################################
    # Save the model
    model.to('cpu')

    state = {
        'model_name': "NeuralNetClassifierOnly",
        'model_input_size': input_size,
        'model_hidden_size_fc1': hidden_size_fc1,
        'model_class_count': class_count,
        'state_dict': model.state_dict(),
        #'optimizer': optimizer.state_dict()
    }

    torch.save(state, (os.path.join(model_filepath, "classifier_model.ckpt")))

    #torch.save(model.state_dict(), 'model.ckpt')
    #torch.save(optimizer.state_dict(), 'optimizer.ckpt')

    logging.info('\n==> Training Completed!\n')

    time_elapsed = time.time() - since
    hrs, _min, sec = hms(time_elapsed)
    logging.info('Time: {:02d}:{:02d}:{:02d}'.format(hrs, _min, sec))







