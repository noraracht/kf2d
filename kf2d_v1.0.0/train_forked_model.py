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
num_epochs = 8000
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


# Input files

#features_csv = "../new_cluster_data_wol/all_data_norm_grouped_NoPsdCnts.csv"
#features_csv = "../new_cluster_data_wol/fungi_dataset/all_data_norm_fungi.csv"

#true_dist_matrix = "../astral.rand.lpp.r100.csv"
#true_dist_matrix = "../new_cluster_data_wol/fungi_dataset/1672taxa_290genes_bb_1_100x.dist_mtrx"

#queries = "../new_cluster_data_wol/50_marker_genes_query.label"
#queries = "../new_cluster_data_wol/fungi_dataset/query_fungi.txt"

#clades_info = "../new_cluster_data_wol/clade_targets.txt"
#clades_info = "../new_cluster_data_wol/fungi_dataset/clade_targets_fungi.txt"



params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 1}


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def train_forked_model_func(features_folder, features_csv, clades_info, true_dist_matrix, num_epochs, hidden_size_fc1, model_filepath):

    since = time.time()

    level = logging.INFO
    format = '%(message)s'
    handlers = [logging.FileHandler(os.path.join(model_filepath, 'run.log'), 'w+'), logging.StreamHandler()]

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
    logging.info('Ground truth directory: {}'.format(true_dist_matrix))


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
    # Read classification information

    logging.info('\n==> Subtree training...\n')

    classification_df = pd.read_csv(clades_info, sep=' ', header=0, index_col=0)  # read from csv
    classification_df["clade"] = classification_df["clade"].astype(int)
    class_count = classification_df.clade.unique()

    # Compute total number of classes
    logging.info('Number of Classes: {}'.format(class_count.size))


    #######################################################################
    # Prepare dataset
    logging.info('\n==> Preparing Data...\n')

    # Subset feature input for a given clade
    feature_input = features_csv
    feature_input = feature_input.iloc[:, :] * features_scaler
    input_size = np.shape(feature_input)[1]

    # logging.info(feature_input)
    logging.info("Dimensions of feature matrix rows: {}, cols: {}".format(np.shape(feature_input)[0],
                                                                          np.shape(feature_input)[1]))

    ########################################################################
    # Get names
    backbone_names = feature_input.index.tolist()

    #######################################################################
    # Create mapping dictionaries
    label_idx_dict, idx_label_dict = get_label_idx_maps(feature_input)    # convert first column into dict



    ##### NEED TO IMPLEMENT ERROR IF DATA IS NOT FOUND #####


    # Load dictionary with true distances
    dir_list = os.listdir(true_dist_matrix)
    c = 0
    dir_list = [i for i in dir_list if "_subtree_{}.di_mtrx".format(c) in i]

    pdf = pd.read_csv(os.path.join(true_dist_matrix, dir_list[0]), sep='\t', header=0,
                      index_col=0)  # read DEPP distance matrix
    pdf_sorted = sort_df(pdf, label_idx_dict)  # sort in the same order as input features

    logging.info("Dimensions of true distance matrix rows: {}, cols: {}".format(np.shape(pdf_sorted)[0],
                                                                                np.shape(pdf_sorted)[1]))
    # logging.info(pdf_sorted)
    #######################################################################
    # Sort class dataframe

    clade_input_sorted = sort_df(classification_df, label_idx_dict)
    clade_input_sorted = np.concatenate(clade_input_sorted)
    # print(clade_input_sorted)


    #######################################################################
    # Prepare train/test dataset split
    partition = {}
    partition['train'] = backbone_names
    partition['test'] = []

    # Custom dataset
    training_set = datasets.Dataset(feature_input, partition['train'], label_idx_dict)
    test_set = datasets.Dataset(feature_input, partition['test'], label_idx_dict)
    # val_set = datasets.Dataset(dataset_fname, partition['val'])

    train_size = training_set.__len__()
    test_size = test_set.__len__()
    # val_size = testset.__len__()

    # Data loader
    train_loader = torch.utils.data.DataLoader(training_set, **params)

    if test_size != 0:
        test_loader = torch.utils.data.DataLoader(test_set, **params)
    # val_loader = torch.utils.data.DataLoader(val_set, **params)

    # check whether data are read correctly
    # for i, (data, labels) in enumerate(train_loader):
    #     logging.info(data)
    #     logging.info(labels)

    logging.info('Number of Train Samples: {}'.format(train_size))
    # logging.info('Number of Test Samples: {}'.format(test_size))
    # logging.info('Number of Validation Samples:{}'.format(val_size))


    #######################################################################
    # Model
    logging.info('\n==> Building model...\n')

    model = models.NeuralNetClassifierForked(input_size, hidden_size_fc1, embedding_size, class_count.size).to(device)

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
    criterion_a = losses.Loss()
    criterion_a.to(device)

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
        train_loss = 0.0
        train_loss2 = 0.0
        running_acc = 0.0

        num_batches = len(train_loader)


        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, input_size).to(device)

            real_dist = pairwise_true_dist(labels, pdf_sorted).to(device) # get true distances
            true_class = torch.from_numpy(clade_input_sorted[np.ix_(list(labels))]).to(device)


            # Forward pass
            outputs, model_class = model(images.float())
            train_dist = pairwise_train_dist(outputs)

            loss_a = criterion_a(train_dist, real_dist)
            train_loss +=loss_a.item() # running loss

            loss_b = criterion_b(model_class, true_class)
            train_loss2 += loss_b.item()

            loss = loss_a + loss_b

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ps = torch.exp(model_class)
            top_p, top_class = ps.topk(1, dim=1)


            running_acc += accuracy_score(true_class.to('cpu'), top_class.to('cpu'))


        train_loss /= num_batches
        train_loss2 /= num_batches
        running_acc /= num_batches

        time_elapsed = time.time() - since
        hrs, _min, sec = hms(time_elapsed)

        if (i+1) % 1 == 0:
            logging.info('Epoch [{}/{}], Step [{}/{}], Train loss: {:.20f}, {:.20f}, {:.20f}, Time: {:02d}:{:02d}:{:02d}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, train_loss, train_loss2, running_acc, hrs, _min, sec))


        ###########################################################################
        # Test the model

        if test_size != 0:

            model.eval()

            test_loss = 0.0
            test_loss2 = 0.0
            test_running_acc = 0.0
            num_batches = len(test_loader)

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.reshape(-1, input_size).to(device)

                    real_dist = pairwise_true_dist(labels, pdf_sorted).to(device)  # get true distances
                    true_class = torch.from_numpy(clade_input_sorted[np.ix_(list(labels))]).to(device)


                    outputs, model_class = model(images.float())
                    train_dist = pairwise_train_dist(outputs)


                    loss_a = criterion_a(train_dist, real_dist)
                    test_loss += loss_a.item()  # running loss

                    loss_b = criterion_b(model_class, true_class)
                    test_loss2 += loss_b.item()

                    loss = loss_a + loss_b


                    ps = torch.exp(model_class)
                    top_p, top_class = ps.topk(1, dim=1)

                    test_running_acc += accuracy_score(true_class.to('cpu'), top_class.to('cpu'))

            test_loss /= num_batches
            test_loss2 /= num_batches
            test_running_acc /= num_batches

            time_elapsed = time.time() - since
            hrs, _min, sec = hms(time_elapsed)

            if (i+1) % 1 == 0:
                logging.info('Epoch [{}/{}], Step [{}/{}], Test loss: {:.20f}, {:.20f}, {:.20f}, Time: {:02d}:{:02d}:{:02d}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, test_loss, test_loss2, test_running_acc, hrs, _min, sec))


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
        'model_name': "NeuralNetClassifierForked",
        'model_input_size': input_size,
        'model_hidden_size_fc1': hidden_size_fc1,
        'model_embedding_size': embedding_size,
        'model_class_count': class_count.size,
        'state_dict': model.state_dict(),
        # 'optimizer': optimizer.state_dict()
    }


    torch.save(state, (os.path.join(model_filepath, "model_forked.ckpt")))


    #######################################################################
    #### Output embeddings ####

    model.eval()

    with torch.no_grad():
        outputs, model_class = model(torch.from_numpy(feature_input.values).float())


    train_dist = pairwise_train_dist(outputs)
    pairwise_outputs3 = torch.square(train_dist)

    # Detach gradient and convert to numpy
    df_outputs = pd.DataFrame(pairwise_outputs3.detach().numpy())
    df_embeddings = pd.DataFrame(outputs.detach().numpy())

    # Attach species names
    df_outputs.columns = backbone_names
    df_outputs.insert(loc=0, column='', value=backbone_names)
    df_embeddings.insert(loc=0, column='', value=backbone_names)

    logging.info(
        "Dimensions of embedding output rows:{} cols:{}".format(len(df_embeddings), len(df_embeddings.columns)))
    df_embeddings.to_csv(os.path.join(model_filepath, 'embeddings_forked.csv'.format(c)), index=False, sep='\t',
                         header=False)


    # Safe model class information to compute misclassified
    ps = torch.exp(model_class)
    top_p, top_class = ps.topk(1, dim=1)
    df_classes = pd.DataFrame(np.hstack((top_class.detach().numpy(), top_p.detach().numpy(), ps.detach().numpy())))

    df_classes.columns = ["top_class", "top_p"] + [str(x) for x in list(range(class_count.size))]
    df_classes.insert(loc=0, column='genome', value=backbone_names)

    # Write to file
    logging.info("Dimensions of class output rows:{} cols:{}".format(len(df_classes), len(df_classes.columns)))
    df_classes.to_csv(os.path.join(model_filepath,'backbone_classes_forked.csv'), index=False, sep='\t')



    logging.info('\n==> Training Completed!\n')

    time_elapsed = time.time() - since
    hrs, _min, sec = hms(time_elapsed)
    logging.info('Time: {:02d}:{:02d}:{:02d}'.format(hrs, _min, sec))







