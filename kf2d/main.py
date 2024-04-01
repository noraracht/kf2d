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
import fnmatch
import treeswift
from treeswift import read_tree_newick
import warnings



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
import os
import glob
import subprocess
from subprocess import call, check_output, STDOUT
import multiprocessing as mp
import math
import copy
import models
import datasets
import losses

import parameter_inits

from utils import *
from weight_inits import *
from train_classifier_model import *
from classify import *
from train_model_set import *
from query import *

default_k_len = 7
min_k_len = 3
max_k_len = 10
default_subtree_sz = 850

hidden_size_fc1 = 2048
embedding_size = 1024
batch_size = 16

default_cl_epochs = 2000
default_di_epochs = 8000

learning_rate = 0.00001     # 1e-5
learning_rate_min = 3e-6    # 3e-6
learning_rate_decay = 2000

seed = 16


__version__ = 'kf2d 1.0.20'


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.



def divide_tree(args):

    # Read tree file
    try:
        tree = treeswift.read_tree_newick(args.tree)
    except:
        print("No such file '{}'".format(args.tree), file=sys.stderr)
        exit(0)



    # Split path and filename
    head_tail = os.path.split(args.tree)
    tree_name = os.path.splitext(os.path.basename(args.tree))[0]


    # Set branch lengths to 1
    for node in tree.traverse_postorder():
        if node.label  != None:
            node.edge_length = 1.0
    tree_tmp = os.path.join(head_tail[0], "{}.{}".format(tree_name, "tree_tmp"))


    # Save tree output
    tree.write_tree_newick(tree_tmp)
    #d=tree.diameter()


    # Run TreeCluster
    subtree_tmp = os.path.join(head_tail[0] , "{}.{}".format(tree_name, "subtrees_tmp"))

    call(["TreeCluster.py", "-i", tree_tmp, "-o", subtree_tmp, "-m", "sum_branch", "-t", str(2* args.size)],
         stderr=open(os.devnull, 'w'))


    # Reformat TreeCluster output
    current_subclades = pd.read_csv(subtree_tmp, sep= '\t', header = 0)

    # Check for -1 subtrees
    labels = current_subclades.loc[current_subclades["ClusterNumber"] ==-1]
    problematic_labels = labels["SequenceName"].to_list()

    if len(problematic_labels) > 0:
        warnings.warn('{} samples are assigned to subtrees -1 and will be excluded.\n'
                      'Please check rooting of your phylogeny or increase subtree size.'.format(len(problematic_labels)))
    else:
        print("There are no -1 subtrees. Keep going...")


    current_subclades = current_subclades.rename({"SequenceName" : "genome", "ClusterNumber" : "clade"}, axis = 1)
    current_subclades["clade"] = current_subclades["clade"]-1
    current_subclades = current_subclades.loc[ current_subclades["clade"] !=-2]


    # Save to file
    subtrees = os.path.join(head_tail[0], "{}.{}".format(tree_name, "subtrees"))
    current_subclades.to_csv(subtrees, index = False, sep = " ", header = True)


    # Clean up
    os.remove(tree_tmp)
    os.remove(subtree_tmp)


def get_frequencies(args):

    # Check if input directory exist
    if os.path.exists(args.input_dir):
        pass
    else:
        print("No such directory '{}'".format(args.input_dir), file=sys.stderr)
        exit(0)

    # Check if output directory exist
    if os.path.exists(args.output_dir):
        pass
    else:
        print("No such directory '{}'".format(args.output_dir), file=sys.stderr)
        exit(0)


    # Making a list of sample names
    formats = ['.fq', '.fastq', '.fa', '.fna', '.fasta']
    files_names = [f for f in os.listdir(args.input_dir)
                   if True in (fnmatch.fnmatch(f, '*' + form) for form in formats)]
    samples_names = [f.rsplit('.f', 1)[0] for f in files_names]


    # Read kmer alphabet
    if args.k==7:
        my_alphabet_kmers = pd.read_csv(os.path.join(os.getcwd(), "test_kmers_7_sorted"), sep = " ", header = None, names = ["kmer"])
    elif args.k==3:
        my_alphabet_kmers = pd.read_csv(os.path.join(os.getcwd(), "vocab_generator_k3C_fin.fa"), sep = " ", header = None, names = ["kmer"])
    elif args.k==4:
        my_alphabet_kmers = pd.read_csv(os.path.join(os.getcwd(), "vocab_generator_k4C_fin.fa"), sep = " ", header = None, names = ["kmer"])
    elif args.k==5:
        my_alphabet_kmers = pd.read_csv(os.path.join(os.getcwd(), "vocab_generator_k5C_fin.fa"), sep = " ", header = None, names = ["kmer"])
    elif args.k==6:
        my_alphabet_kmers = pd.read_csv(os.path.join(os.getcwd(), "test_kmers_6_sorted"), sep = " ", header = None, names = ["kmer"])
    elif args.k==8:
        my_alphabet_kmers = pd.read_csv(os.path.join(os.getcwd(), "vocab_generator_k8C_fin.fa"), sep = " ", header = None, names = ["kmer"])
    elif args.k==9:
        my_alphabet_kmers = pd.read_csv(os.path.join(os.getcwd(), "vocab_generator_k9C_fin.fa"), sep = " ", header = None, names = ["kmer"])
    elif args.k==10:
        my_alphabet_kmers = pd.read_csv(os.path.join(os.getcwd(), "vocab_generator_k10C_fin.fa"), sep = " ", header = None, names = ["kmer"])



    # Compute kmer counts per file
    for i in range (0, len(files_names)):
        print(files_names[i])

        # Run jellyfish
        f1 = os.path.join(args.output_dir, "{}.{}".format(samples_names[i],"jf"))
        call(["jellyfish", "count", "-m", str(args.k), "-s", "100M", "-t", str(args.p),
               "-C", os.path.join( args.input_dir, files_names[i]) ,"-o", f1],
             stderr=open(os.devnull, 'w'))

        f2 = os.path.join(args.output_dir, "{}.{}".format(samples_names[i], "dump"))
        with open(f2, "w") as outfile:
            subprocess.run(["jellyfish", "dump", "-c", f1], stdout=outfile)

        # Read into dataframe
        my_current_kmers = pd.read_csv(f2, sep = " ", header = None, names = ["kmer", "counts"])


        # Merge dataframe with alphabet kmers
        my_merged_counts = pd.merge(my_alphabet_kmers, my_current_kmers, how = 'left', left_on="kmer", right_on="kmer")
        my_merged_counts = my_merged_counts[['counts']].fillna(0)


        # Add pseudocounts if flag is on
        if args.pseudocount:
            my_merged_counts["counts"] =  my_merged_counts["counts"] + 0.5
        else:
            pass


        # Normalize frequencies to 1
        my_merged_counts["counts"] = my_merged_counts["counts"]/my_merged_counts["counts"].sum()
        my_merged_counts["counts"] = my_merged_counts["counts"].astype(str)
        my_merged_list = my_merged_counts["counts"].to_list()


        # Output into file
        f3 = os.path.join(args.output_dir, "{}.{}".format(samples_names[i], "kf"))
        my_output = ",".join(my_merged_list)

        with open(f3, "w") as f:

            f.write("{},".format (str(samples_names[i])))
            f.write(my_output)
            f.write("\n")

        # Clean up
        os.remove(f1)
        os.remove(f2)


def train_classifier(args):

    # Concetenate kmer frequencies into single dataframe
    all_files = glob.glob(os.path.join(args.input_dir, "*.kf"))

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=None, sep= ',')
        li.append(df)


    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.set_index(0, inplace=True)

    # Concatenate inputs into single dataframe
    #frame = construct_input_dataframe(li)

    train_classifier_model_func(args.input_dir, frame, args.subtrees, args.e, args.hidden_sz, args.batch_sz, args.lr, args.lr_min, args.lr_decay, args.seed, args.o)


def classify(args):

    # Concetenate kmer frequencies into single dataframe
    all_files = glob.glob(os.path.join(args.input_dir, "*.kf"))

    li = []

    # # Delete previous log file if exist
    # try:
    #     os.remove(os.path.join(args.output_classes, 'classification.log'))
    # except OSError:
    #     pass

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=None, sep=',')

        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.set_index(0, inplace=True)

    # Concatenate inputs into single dataframe
    # frame = construct_input_dataframe(li)

    classify_func(args.input_dir, frame, args.model, args.seed, args.o)


def get_distances(args):

    # Read tree file
    try:
        tree = treeswift.read_tree_newick(args.tree)
    except:
        print("No such file '{}'".format(args.tree), file=sys.stderr)
        exit(0)


    # Split path and filename
    head_tail = os.path.split(args.tree)
    tree_name = os.path.splitext(os.path.basename(args.tree))[0]

    # Scale branches by 100
    tree.scale_edges(100)

    # Compute distance matrix for a full tree and convert into dataframe
    if args.mode == "full_only" or args.mode == "hybrid":

        only_leaves = tree.num_nodes(internal=False)  # exclude internal nodes

        # Warning for trees with more than 12K species
        if only_leaves > 12000:
            warnings.warn('Phylogeny contains {} samples which is above recommended threshold of 12000 species.\n'
                          'Computation of distance matrix might take long time.'.format(only_leaves))
        else:
            pass

        M = tree.distance_matrix(leaf_labels=True)
        df = pd.DataFrame.from_dict(M, orient='index').fillna(0)
        df.to_csv(os.path.join(head_tail[0], '{}_full.di_mtrx'.format(tree_name)), index=True, sep='\t')





    # Read clades information if provided by the user
    if args.mode == "hybrid" or args.mode == "subtrees_only":

        if args.subtrees is None:
            print("No such file '{}'. Please provide /.subtrees file or change mode to full_only".format(args.subtrees), file=sys.stderr)
            exit(0)

        else:
            clade_input = pd.read_csv(args.subtrees, sep=' ', header=0, index_col=0)
            clade_selection = list(set(clade_input["clade"].to_list()))


            # Compute distance matrices for subtrees
            for c in clade_selection:

                # Get labels of the subtree
                labels_to_keep = set(clade_input.loc[clade_input["clade"] == c].index.to_list())

                # NOTE: Here I am not checking for single species clades since
                # they should have been eliminated during clading step

                # Generate subtree
                tree2 = tree.extract_tree_with(labels_to_keep)

                # Compute distance matrix for a subtree and convert into dataframe
                M = tree2.distance_matrix(leaf_labels=True)
                df = pd.DataFrame.from_dict(M, orient='index').fillna(0)
                df.to_csv(os.path.join(head_tail[0], '{}_subtree_{}.di_mtrx'.format(tree_name, c)), index=True, sep='\t')



def train_model_set(args):

    # Concatenate kmer frequencies into single dataframe
    all_files = glob.glob(os.path.join(args.input_dir, "*.kf"))

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=None, sep= ',')
        li.append(df)


    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.set_index(0, inplace=True)

    # Concatenate inputs into single dataframe
    # frame = construct_input_dataframe(li)


    train_model_set_func(args.input_dir, frame, args.subtrees, args.true_dist, args.e, args.hidden_sz, args.embed_sz, args.batch_sz, args.lr, args.lr_min, args.lr_decay, args.seed, args.o)



def query(args):

    # Concatenate kmer frequencies into single dataframe
    all_files = glob.glob(os.path.join(args.input_dir, "*.kf"))

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=None, sep=',')
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.set_index(0, inplace=True)

    # Concatenate inputs into single dataframe
    # frame = construct_input_dataframe(li)

    query_func(args.input_dir, frame, args.model, args.classes, args.seed, args.o)



def build_library(args):


    print("\n==> Computing k-mer frequences\n")
    get_frequencies(args)


    print("\n==> Splitting phylogeny into subtrees\n")
    divide_tree(args)


    print("\n==> Computing distance matrices\n")

    # Split path and filename
    head_tail = os.path.split(args.tree)
    tree_name = os.path.splitext(os.path.basename(args.tree))[0]

    args.subtrees = os.path.join(head_tail[0], "{}.{}".format(tree_name, "subtrees"))
    get_distances(args)


    print("\n==> Training classifier model\n")
    args.input_dir = args.output_dir
    # args.subtrees = args.subtrees # defined above
    args.e = args.cl_epochs
    args.hidden_sz = args.cl_hidden_sz
    args.batch_sz = args.cl_batch_sz
    args.lr = args.cl_lr
    args.lr_min = args.cl_lr_min
    args.lr_decay = args.cl_lr_decay
    args.seed = args.cl_seed
    args.o = args.output_dir

    train_classifier(args)


    print("\n==> Training distance models\n")
    #args.input_dir = args.output_dir # defined above
    args.true_dist = head_tail[0]
    # args.subtrees = args.subtrees # defined above
    args.e = args.di_epochs
    args.hidden_sz = args.di_hidden_sz
    args.embed_sz = args.di_embed_sz
    args.batch_sz = args.di_batch_sz
    args.lr = args.di_lr
    args.lr_min = args.di_lr_min
    args.lr_decay = args.di_lr_decay
    args.seed = args.di_seed
    #args.o = args.output_dir # defined above

    train_model_set(args)


    print('\n==> Building library step is completed!\n')



def process_query_data(args):

    print("\n==> Computing k-mer frequences\n")
    get_frequencies(args)


    print("\n==> Classifying query samples\n")
    args.input_dir = args.output_dir
    args.model = args.classifier_model
    args.seed = args.cl_seed
    args.o = args.output_dir

    classify(args)


    print("\n==> Computing model distances\n")
    # args.input_dir = args.output_dir # defined above
    args.model = args.distance_model
    args.classes = args.output_dir
    args.seed = args.di_seed
    # args.o = args.output_dir # defined above

    query(args)


    print('\n==> Query processing step is completed!\n')




def main():
    # Input arguments parser
    parser = argparse.ArgumentParser(description='K-mer frequency to distance\n{}'.format(__version__),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', help='print the current version', version='{}'.format(__version__))
    # parser.add_argument('--debug', action='store_true', help='Print the traceback when an exception is raised')
    subparsers = parser.add_subparsers(title='commands',
                                       description='get_frequencies        Extract k-mer frequency from a reference genome-skims or assemblies\n'
                                                   'divide_tree            Divides input phylogeny into subtrees\n'
                                                   'get_distances          Compute distance matrices\n'
                                                   'train_classifier       Train classifier model based on backbone subtrees\n'
                                                   'classify               Classifies query samples using previously trained classifier model\n'
                                                   # 'train_model     Performs correction of subsampled distance matrices obtained for reference\n'
                                                   'train_model_set        Trains all models for subtrees consecutively\n'
                                                   'query                  Query subtree models\n'
                                                   # 'genome-skims or assemblies'
                                                   'build_library          Wrapper command to preprocess backbone sequences and phylogeny to train classifier and distance models\n'
                                                   'process_query_data     Wrapper command to preprocess query sequences, classify and compute distances to backbone species\n'
                                       ,
                                       help='Run kf2d {commands} [-h] for additional help',
                                       dest='{commands}')


    # Get_frequencies command subparser

    ### To invoke
    ### python main.py get_frequencies -input_dir /Users/nora/PycharmProjects/test_freq -output_dir /Users/nora/PycharmProjects/test_freq
    ### python main.py get_frequencies -input_dir /Users/nora/PycharmProjects/test_freq -pseudocount

    ### python main.py get_frequencies - input_dir.. / toy_example / train_tree_fna - output_dir.. / toy_example / train_tree_kf
    ### python main.py get_frequencies -input_dir ../toy_example/test_fna -output_dir ../toy_example/test_kf

    parser_freq = subparsers.add_parser('get_frequencies',
                                       description='Process a library of reference genome-skims or assemblies')
    parser_freq.add_argument('-input_dir',
                            help='Directory of input genomes or assemblies (dir of .fastq/.fq/.fa/.fna/.fasta files)')
    parser_freq.add_argument('-output_dir',
                             help='Directory for k-mer frequency outputs (dir for .kf files)')
    parser_freq.add_argument('-k', type=int, choices=list(range(min_k_len, max_k_len+1)), default=default_k_len, help='K-mer length [{}-{}]. '.format(min_k_len, max_k_len) +
                                                                                         'Default: {}'.format(default_k_len), metavar='K')
    parser_freq.add_argument('-p', type=int, choices=list(range(1, mp.cpu_count() + 1)), default=mp.cpu_count(),
                            help='Max number of processors to use [1-{0}]. '.format(mp.cpu_count()) +
                                 'Default for this machine: {0}'.format(mp.cpu_count()), metavar='P')
    parser_freq.add_argument('-pseudocount', action='store_true',
                           help='Computes k-mer counts with 0.5 pseudocount added to each frequency value')
    parser_freq.set_defaults(func=get_frequencies)


    # Divide_tree command subparser

    ### To invoke
    ### python main.py divide_tree -size 850 -tree /Users/nora/PycharmProjects/astral.rand.lpp.r100.EXTENDED.nwk

    ### python main.py divide_tree -tree ../toy_example/train_tree_newick/train_tree.nwk -size 2

    parser_div = subparsers.add_parser('divide_tree',
                                       description='Divides input phylogeny into subtrees.')
    parser_div.add_argument('-tree', help='Input phylogeny (a .newick/.nwk format)')
    parser_div.add_argument('-size', type=int, default=default_subtree_sz, help='Size of the subtree. ' +
                                                                                         'Default: {}'.format(default_subtree_sz))
    parser_div.set_defaults(func=divide_tree)


    # Get_distances command subparser

    ### To invoke
    ### python main.py get_distances -tree /Users/nora/PycharmProjects/test_tree.nwk  -subtrees  /Users/nora/PycharmProjects/my_test.subtrees -mode subtrees_only

    ### python main.py get_distances -tree ../toy_example/train_tree_newick/train_tree.nwk  -subtrees  ../toy_example/train_tree_newick/train_tree.subtrees -mode subtrees_only

    parser_distances = subparsers.add_parser('get_distances',
                                             description='Computes distance matrices')
    parser_distances.add_argument('-tree', help='Input phylogeny (a .newick/.nwk format)', required=True)
    parser_distances.add_argument('-subtrees',
                                  help='Classification file with subtrees information obtained from divide_tree command (a .subtrees format)')
    parser_distances.add_argument('-mode', type=str, metavar='',
                                  choices={"full_only", "hybrid", "subtrees_only"}, default="hybrid",
                                  help='Ways to perform distance computation [full_only, hybrid, subtrees_only]. ' +
                                       'Default: hybrid')

    parser_distances.set_defaults(func=get_distances)


    # Train_classifier command subparser

    ### To invoke
    ### python main.py train_classifier -input_dir /Users/nora/PycharmProjects/train_tree_kf -subtrees /Users/nora/PycharmProjects/my_test.subtrees -e 1 -o /Users/nora/PycharmProjects/my_toy_input

    ### python main.py train_classifier -input_dir ../toy_example/train_tree_kf -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 10 -o ../toy_example/train_tree_models
    ### python main.py train_classifier -input_dir ../toy_example/train_tree_kf -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 10  -hidden_sz 2000 -batch_sz 32 -o ../toy_example/train_tree_models

    parser_trclas = subparsers.add_parser('train_classifier',
                                        description='Train classifier model based on backbone subtrees')
    parser_trclas.add_argument('-input_dir',
                             help='Directory of input k-mer frequencies for assemblies or reads (dir of .kf files for backbone)')
    parser_trclas.add_argument('-subtrees', help='Classification file with subtrees information obtained from divide_tree command (a .subtrees format)')
    # parser_trclas.add_argument('-e', type=int, metavar='', choices=list(range(1, max_cl_epochs)), default=default_cl_epochs, help='Epochs [1-{}]. '.format(max_cl_epochs-1) +
    #                                                                                     'Default: {}'.format(default_cl_epochs))
    parser_trclas.add_argument('-e', type=int, default=default_cl_epochs, help='Number of epochs. ' +
                                                                               'Default: {}'.format(default_cl_epochs))
    parser_trclas.add_argument('-hidden_sz', type=int, default=hidden_size_fc1, help='Hidden size. ' +
                                                                                       'Default: {}'.format(hidden_size_fc1))
    parser_trclas.add_argument('-batch_sz', type=int, default=batch_size, help='Batch size. ' +
                                                                                        'Default: {}'.format(batch_size))
    parser_trclas.add_argument('-lr', type=float, default=learning_rate, help='Start learning rate. ' +
                                                                                        'Default: {}'.format(learning_rate))
    parser_trclas.add_argument('-lr_min', type=float, default=learning_rate_min, help='Minimum learning rate. ' +
                                                                                        'Default: {}'.format(learning_rate_min))
    parser_trclas.add_argument('-lr_decay', type=float, default=learning_rate_decay, help='Learning rate decay. ' +
                                                                                      'Default: {}'.format(learning_rate_decay))
    parser_trclas.add_argument('-seed', type=int, default=seed, help='Random seed. ' +
                                                                            'Default: {}'.format(seed))
    parser_trclas.add_argument('-o',
                               help='Model output path')

    parser_trclas.set_defaults(func=train_classifier)


    # Classify command subparser

    ### To invoke
    ### python main.py classify -input_dir /Users/nora/PycharmProjects/test_tree_kf -model /Users/nora/PycharmProjects/my_toy_input  -o /Users/nora/PycharmProjects/my_toy_input

    ### python main.py classify -input_dir ../toy_example/test_kf -model ../toy_example/train_tree_models -o ../toy_example/test_results

    parser_classify = subparsers.add_parser('classify',
                                          description='Classifies query inputs using previously trained classifier model')
    parser_classify.add_argument('-input_dir',
                               help='Directory of input k-mer frequencies for queries samples: assemblies or reads (dir of .kf files for queries)')
    parser_classify.add_argument('-model',
                               help='Classification model')
    parser_classify.add_argument('-seed', type=int, default=seed, help='Random seed. ' +
                                                                     'Default: {}'.format(seed))
    parser_classify.add_argument('-o',
                               help='Output path')

    parser_classify.set_defaults(func=classify)




    # Train_model_set command subparser

    ### To invoke
    ### python main.py train_model_set -input_dir /Users/nora/PycharmProjects/train_tree_kf  -true_dist /Users/nora/PycharmProjects  -subtrees /Users/nora/PycharmProjects/my_test.subtrees -e 1 -o /Users/nora/PycharmProjects/my_toy_input

    ### python main.py train_model_set -input_dir ../toy_example/train_tree_kf -true_dist ../toy_example/train_tree_newick  -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 1 -o ../toy_example/train_tree_models

    parser_train_model_set = subparsers.add_parser('train_model_set',
                                            description='Trains individual models for each subtree')
    parser_train_model_set.add_argument('-input_dir',
                               help='Directory of input k-mer frequencies for assemblies or reads (dir of .kf files for backbone)')
    parser_train_model_set.add_argument('-true_dist',
                                        help='Directory of distamce matrices for backbone subtrees (dir of *subtree_INDEX.di_mtrx files for backbone)')
    parser_train_model_set.add_argument('-subtrees',
                               help='Classification file with subtrees information obtained from divide_tree command (a .subtrees format)')
    parser_train_model_set.add_argument('-e', type=int, default=default_di_epochs, help='Number of epochs. ' +
                                                                                                 'Default: {}'.format(default_di_epochs))
    parser_train_model_set.add_argument('-hidden_sz', type=int, default=hidden_size_fc1, help='Hidden size. ' +
                                                                                                'Default: {}'.format(hidden_size_fc1))
    parser_train_model_set.add_argument('-embed_sz', type=int, default=embedding_size, help='Embedding size. ' +
                                                                                                'Default: {}'.format(embedding_size))
    parser_train_model_set.add_argument('-batch_sz', type=int, default=batch_size, help='Batch size. ' +
                                                                            'Default: {}'.format(batch_size))
    parser_train_model_set.add_argument('-lr', type=float, default=learning_rate, help='Start learning rate. ' +
                                                                        'Default: {}'.format(learning_rate))
    parser_train_model_set.add_argument('-lr_min', type=float, default=learning_rate_min, help='Minimum learning rate. ' +
                                                                         'Default: {}'.format(learning_rate_min))
    parser_train_model_set.add_argument('-lr_decay', type=float, default=learning_rate_decay, help='Learning rate decay. ' +
                                                                                          'Default: {}'.format(learning_rate_decay))
    parser_train_model_set.add_argument('-seed', type=int, default=seed, help='Random seed. ' +
                                                                       'Default: {}'.format(seed))
    parser_train_model_set.add_argument('-o',
                               help='Model output path')

    parser_train_model_set.set_defaults(func=train_model_set)



    # Query_model_set command subparser

    ### To invoke
    ### python main.py query -input_dir /Users/nora/PycharmProjects/test_tree_kf  -model /Users/nora/PycharmProjects/my_toy_input  -classes /Users/nora/PycharmProjects/my_toy_input  -o /Users/nora/PycharmProjects/my_toy_input

    ### python main.py query -input_dir ../toy_example/test_kf  -model ../toy_example/train_tree_models -classes ../toy_example/test_results  -o ../toy_example/test_results

    parser_query = subparsers.add_parser('query',
                                                   description='Query models')
    parser_query.add_argument('-input_dir',
                                        help='Directory of input k-mer frequencies for assemblies or reads (dir of .kf files for queries)')
    parser_query.add_argument('-model',
                                        help='Directory of models and embeddings (dir of model_subtree_INDEX.ckpt and embeddings_subtree_INDEX.csv files for backbone)')
    parser_query.add_argument('-classes',
                                        help='Path to classification file with subtrees information obtained from classify command (classes.out file)')
    parser_query.add_argument('-seed', type=int, default=seed, help='Random seed. ' +
                                                                              'Default: {}'.format(seed))

    parser_query.add_argument('-o',
                                        help='Output path')

    parser_query.set_defaults(func=query)






    # Build_library command subparser

    ### To invoke
    ### python main.py build_library -input_dir /Users/nora/PycharmProjects/train_tree_fna -output_dir /Users/nora/PycharmProjects/train_tree_output -size 2 -tree /Users/nora/PycharmProjects/test_tree.nwk -mode subtrees_only -cl_epochs 1 -di_epochs 1

    ### python main.py build_library -input_dir ../toy_example/train_tree_fna -output_dir ../toy_example/combo_models -size 2 -tree ../toy_example/train_tree_newick/train_tree.nwk -mode subtrees_only -cl_epochs 10 -di_epochs 1

    parser_build_library = subparsers.add_parser('build_library',
                                                   description='Wrapper command that combines subcommands: get_frequencies (from backbone sequences), divide_tree, get_distance, train_classifier and train_model_set')

    parser_build_library.add_argument('-input_dir',
                             help='Directory of input genomes or assemblies (dir of .fastq/.fq/.fa/.fna/.fasta files)')
    parser_build_library.add_argument('-output_dir',
                             help='Directory for all outputs (dir for .kf files)')
    parser_build_library.add_argument('-k', type=int, choices=list(range(min_k_len, max_k_len + 1)), default=default_k_len,
                             help='K-mer length [{}-{}]. '.format(min_k_len, max_k_len) +
                                  'Default: {}'.format(default_k_len), metavar='K')
    parser_build_library.add_argument('-p', type=int, choices=list(range(1, mp.cpu_count() + 1)), default=mp.cpu_count(),
                             help='Max number of processors to use [1-{0}]. '.format(mp.cpu_count()) +
                                  'Default for this machine: {0}'.format(mp.cpu_count()), metavar='P')
    parser_build_library.add_argument('-pseudocount', action='store_true',
                             help='Computes k-mer counts with 0.5 pseudocount added to each frequency value')

    parser_build_library.add_argument('-tree', help='Input phylogeny (a .newick/.nwk format)')
    parser_build_library.add_argument('-size', type=int, default=default_subtree_sz, help='Size of the subtree. ' +
                                                                 'Default: {}'.format(default_subtree_sz))

    parser_build_library.add_argument('-mode', type=str, metavar='',
                                  choices={"full_only", "hybrid", "subtrees_only"}, default="hybrid",
                                  help='Ways to perform distance computation [full_only, hybrid, subtrees_only]. ' +
                                       'Default: hybrid')

    parser_build_library.add_argument('-cl_epochs', type=int, default=default_cl_epochs, help='Number of epochs to train classifier model. ' +
                                                               'Default: {}'.format(default_cl_epochs))
    parser_build_library.add_argument('-cl_hidden_sz', type=int, default=hidden_size_fc1, help='Classifier hidden size. ' +
                                                                                       'Default: {}'.format(hidden_size_fc1))
    parser_build_library.add_argument('-cl_batch_sz', type=int, default=batch_size, help='Classifier batch size. ' +
                                                                                         'Default: {}'.format(batch_size))
    parser_build_library.add_argument('-cl_lr', type=float, default=learning_rate, help='Classifier start learning rate. ' +
                                                                              'Default: {}'.format(learning_rate))
    parser_build_library.add_argument('-cl_lr_min', type=float, default=learning_rate_min, help='Classifier minimum learning rate. ' +
                                                                              'Default: {}'.format(learning_rate_min))
    parser_build_library.add_argument('-cl_lr_decay', type=float, default=learning_rate_decay, help='Classifier learning rate decay. ' +
                                             'Default: {}'.format(learning_rate_decay))
    parser_build_library.add_argument('-cl_seed', type=int, default=seed, help='Classifier random seed. ' +
                                                                    'Default: {}'.format(seed))


    parser_build_library.add_argument('-di_epochs', type=int, default=default_di_epochs, help='Number of epochs to train distance models. ' +
                                                                        'Default: {}'.format(default_di_epochs))
    parser_build_library.add_argument('-di_hidden_sz', type=int, default=hidden_size_fc1, help='Hidden size for distance models. ' +
                                                                                                 'Default: {}'.format(hidden_size_fc1))
    parser_build_library.add_argument('-di_embed_sz', type=int, default=embedding_size, help='Distance model embedding size. ' +
                                                                                          'Default: {}'.format(embedding_size))
    parser_build_library.add_argument('-di_batch_sz', type=int, default=batch_size, help='Distance model batch size. ' +
                                             'Default: {}'.format(batch_size))
    parser_build_library.add_argument('-di_lr', type=float, default=learning_rate, help='Distance model start learning rate. ' +
                                                                                       'Default: {}'.format(learning_rate))
    parser_build_library.add_argument('-di_lr_min', type=float, default=learning_rate_min, help='Distance model minimum learning rate. ' +
                                             'Default: {}'.format(learning_rate_min))
    parser_build_library.add_argument('-di_lr_decay', type=float, default=learning_rate_decay, help='Distance learning rate decay. ' +
                                           'Default: {}'.format(learning_rate_decay))
    parser_build_library.add_argument('-di_seed', type=int, default=seed, help='Distance model random seed. ' +
                                                                               'Default: {}'.format(seed))


    parser_build_library.set_defaults(func=build_library)



    # Process_query_data command subparser

    ### To invoke
    ### python main.py process_query_data -input_dir /Users/nora/PycharmProjects/test_freq -output_dir /Users/nora/PycharmProjects/test_tree_output  -classifier_model /Users/nora/PycharmProjects/train_tree_output -distance_model /Users/nora/PycharmProjects/train_tree_output

    ### python main.py process_query_data -input_dir ../toy_example/test_fna -output_dir ../toy_example/combo_results   -classifier_model ../toy_example/combo_models -distance_model ../toy_example/combo_models

    parser_process_query_data = subparsers.add_parser('process_query_data',
                                         description='Wrapper command that combines subcommands: get_frequencies (from query samples), classify and query')

    parser_process_query_data.add_argument('-input_dir',
                                      help='Directory of input genomes or assemblies (dir of .fastq/.fq/.fa/.fna/.fasta files)')
    parser_process_query_data.add_argument('-output_dir',
                                      help='Directory for outputs (dir for .kf files)')
    parser_process_query_data.add_argument('-k', type=int, choices=list(range(3, 11)), default=7,
                                      help='K-mer length [3-10]. ' +
                                           'Default: 7', metavar='K')
    parser_process_query_data.add_argument('-p', type=int, choices=list(range(1, mp.cpu_count() + 1)),
                                      default=mp.cpu_count(),
                                      help='Max number of processors to use [1-{0}]. '.format(mp.cpu_count()) +
                                           'Default for this machine: {0}'.format(mp.cpu_count()), metavar='P')
    parser_process_query_data.add_argument('-pseudocount', action='store_true',
                                      help='Computes k-mer counts with 0.5 pseudocount added to each frequency value')

    parser_process_query_data.add_argument('-classifier_model',
                                 help='Classification model path')
    parser_process_query_data.add_argument('-cl_seed', type=int, default=seed, help='Clssification random seed. ' +
                                                                               'Default: {}'.format(seed))

    parser_process_query_data.add_argument('-distance_model',
                              help='Directory of models and embeddings (dir of model_subtree_INDEX.ckpt and embeddings_subtree_INDEX.csv files for backbone)')
    parser_process_query_data.add_argument('-di_seed', type=int, default=seed, help='Query random seed. ' +
                                                                                    'Default: {}'.format(seed))


    parser_process_query_data.set_defaults(func=process_query_data)


    args = parser.parse_args()
    args.func(args)






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm oop')
    main()




