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
features_scaler = 1e4

seed = 16
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True


def query_func(features_folder, features_csv, model_file, classes, output_folder):

    since = time.time()

    level = logging.INFO
    format = '%(message)s'
    handlers = [logging.FileHandler(os.path.join(output_folder, 'query_run.log'), 'w+'), logging.StreamHandler()]

    #logging.basicConfig(level=logging.NOTSET, format='%(asctime)s | %(levelname)s: %(message)s', handlers=handlers)

    logging.basicConfig(level=level, format=format, handlers=handlers)
    # logging.info('Hey, this is working!')


    #######################################################################
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #######################################################################
    logging.info('\n==> Input arguments...\n')

    logging.info('Query directory: {}'.format(features_folder))
    logging.info('Model directory: {}'.format(model_file))
    logging.info('Class information: {}'.format(classes))


    #######################################################################
    # Read classification information

    logging.info('\n==> Querying...\n')


    #classification_df = pd.read_csv(os.path.join(os.getcwd(), classes), sep="\t", header=0)
    classification_df = pd.read_csv(os.path.join(classes, "classes.out") , sep="\t", header=0)

    classification_df["top_class"] = classification_df["top_class"].astype(int)
    class_count = classification_df.top_class.unique()

    # Compute total number of classes
    logging.info('Total subtrees to query: {}'.format(classification_df.top_class.unique().size))


    current_class_ids = {}


    for c in class_count:

        current_clade = classification_df.loc[classification_df["top_class"]==c]
        current_class_ids[c] = current_clade["genome"].to_list()

        #######################################################################
        logging.info('\n==> Working on subtree {}...\n'.format(c))


        #######################################################################
        # Prepare dataset
        logging.info('\n==> Preparing Data...\n')

        # Subset feature input for a given clade
        feature_input = features_csv.loc[features_csv.index.isin(current_class_ids[c])]
        feature_input = feature_input.iloc[:,:]*features_scaler
        input_size = np.shape(feature_input)[1]

        logging.info("Dimensions of feature matrix rows: {}, cols: {}".format(np.shape(feature_input)[0], np.shape(feature_input)[1]))

        ########################################################################
        # Get names
        query_names = feature_input.index.tolist()

        #######################################################################
        # Model
        logging.info('\n==> Building model...\n')


        ##### Load model #####
        # NEED TO CHECK IF FILE EXISTS
        state = torch.load(os.path.join(model_file, "model_subtree_{}.ckpt".format(c)))

        input_size = state["model_input_size"]
        hidden_size_fc1 = state["model_hidden_size_fc1"]
        embedding_size = state["model_embedding_size"]

        # Need to find a way to save model name as well
        model = models.NeuralNet(input_size, hidden_size_fc1, embedding_size)
        model.load_state_dict(state['state_dict'])
        # model.to(device)
        model.to("cpu")

        #######################################################################
        # Training model
        logging.info('\n==> Compute model output...\n')


        # Compute model output
        model.eval()

        with torch.no_grad():
            outputs = model(torch.from_numpy(feature_input.values).float())
        #logging.info(outputs)


        # Read embeddings
        df_embeddings = pd.read_csv(os.path.join(model_file, 'embeddings_subtree_{}.csv'.format(c)), sep="\t", header=None, index_col = 0)
        backbone_names = df_embeddings.index.tolist()


        # Compute pairwise distance matrix
        embeddings_tensor = torch.from_numpy(df_embeddings.values).float()
        pairwise_outputs = torch.cdist(outputs, embeddings_tensor, p=2, compute_mode='donot_use_mm_for_euclid_dist')
        pairwise_outputs2 = torch.square(pairwise_outputs)
        pairwise_outputs3 = torch.div(pairwise_outputs2, 1.0) # NEED TO FIX BUT GOOD FOR Current TEST

        # Round distances < 1e10-6 to 0 so apples can handle such values
        pairwise_outputs3 = torch.where(pairwise_outputs3 < 1.0e-6, torch.tensor(0, dtype=pairwise_outputs3.dtype),
                                        pairwise_outputs3)

        #######################################################################
        # Compute distance matrix for entire set

        # Detach gradient and convert to numpy
        df_outputs = pd.DataFrame(pairwise_outputs3.detach().numpy())

        # Attach species names
        df_outputs.columns = backbone_names
        df_outputs.insert(loc=0, column='', value=query_names)

        #######################################################################
        # Generate Apples input

        # Write to file
        logging.info("Dimensions of output matrix rows:{} cols:{}".format(len(df_outputs), len(df_outputs.columns)-1))
        df_outputs.to_csv(os.path.join(output_folder, 'apples_input_di_mtrx_query_subtree_{}.csv'.format(c)), index=False, sep='\t')


        logging.info('\n==> Computation is completed for subtree {}!\n'.format(c))

        time_elapsed = time.time() - since
        hrs, _min, sec = hms(time_elapsed)
        logging.info('Time: {:02d}:{:02d}:{:02d}'.format(hrs, _min, sec))



    logging.info('\n==> Computation Completed!\n'.format(c))

    time_elapsed = time.time() - since
    hrs, _min, sec = hms(time_elapsed)
    logging.info('Time: {:02d}:{:02d}:{:02d}'.format(hrs, _min, sec))






