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

    # Check if directory exist
    if os.path.exists(args.input_dir):
        pass
    else:
        print("No such directory '{}'".format(args.input_dir), file=sys.stderr)
        exit(0)


    # Making a list of sample names
    formats = ['.fq', '.fastq', '.fa', '.fna', '.fasta']
    files_names = [f for f in os.listdir(args.input_dir)
                   if True in (fnmatch.fnmatch(f, '*' + form) for form in formats)]
    samples_names = [f.rsplit('.f', 1)[0] for f in files_names]


    # Read kmer alphabet
    if args.k==7:
        my_alphabet_kmers = pd.read_csv(os.path.join(os.getcwd(), "test_kmers_7_sorted"), sep = " ", header = None, names = ["kmer"])
    elif args.k==6:
        my_alphabet_kmers = pd.read_csv(os.path.join(os.getcwd(), "test_kmers_6_sorted"), sep = " ", header = None, names = ["kmer"])


    # Compute kmer counts per file
    for i in range (0, len(files_names)):
        print(files_names[i])

        # Run jellyfish
        f1 = os.path.join(args.input_dir, "{}.{}".format(samples_names[i],"jf"))
        call(["jellyfish", "count", "-m", str(args.k), "-s", "100M", "-t", str(args.p),
               "-C", os.path.join( args.input_dir, files_names[i]) ,"-o", f1],
             stderr=open(os.devnull, 'w'))

        f2 = os.path.join(args.input_dir, "{}.{}".format(samples_names[i], "dump"))
        with open(f2, "w") as outfile:
            subprocess.run(["jellyfish", "dump", "-c", f1], stdout=outfile)

        # Read into dataframe
        my_current_kmers = pd.read_csv(f2, sep = " ", header = None, names = ["kmer", "counts"])


        # Merge dataframe with alphabet kmers
        my_merged_counts = pd.merge(my_alphabet_kmers, my_current_kmers, how = 'left', left_on="kmer", right_on="kmer")
        my_merged_counts = my_merged_counts[['counts']].fillna(0)


        # Normalize frequencies to 1
        my_merged_counts["counts"] = my_merged_counts["counts"]/my_merged_counts["counts"].sum()
        my_merged_counts["counts"] = my_merged_counts["counts"].astype(str)
        my_merged_list = my_merged_counts["counts"].to_list()


        # Output into file
        f3 = os.path.join(args.input_dir, "{}.{}".format(samples_names[i], "kf"))
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


    train_classifier_model_func(args.input_dir, frame, args.subtrees, args.e, args.o)


def classify(args):

    # Concetenate kmer frequencies into single dataframe
    all_files = glob.glob(os.path.join(args.input_dir, "*.kf"))

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=None, sep=',')
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.set_index(0, inplace=True)

    classify_func(args.input_dir, frame, args.model, args.output_classes)


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
    if args.mode == "hybrid" or args.subtrees is None:

        only_leaves = tree.num_nodes(internal=False)  # exclude internal nodes


        # Warning for trees with more than 12K species
        if only_leaves > 12000:
            warnings.warn('Phlogeny contains {} samples which is above recommended threshold of 12000 species.\n'
                          'Computation of distance matrix might take long time.'.format(only_leaves))
        else:
            pass

        M = tree.distance_matrix(leaf_labels=True)
        df = pd.DataFrame.from_dict(M, orient='index').fillna(0)
        df.to_csv(os.path.join(head_tail[0], '{}_full.di_mtrx'.format(tree_name)), index=True, sep='\t')


    # Read clades information if provided by the user
    if args.subtrees is not None:
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

    # Concetenate kmer frequencies into single dataframe
    all_files = glob.glob(os.path.join(args.input_dir, "*.kf"))

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=None, sep= ',')
        li.append(df)


    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.set_index(0, inplace=True)


    train_model_set_func(args.input_dir, frame, args.subtrees, args.true_dist, args.e, args.o)



def query(args):

    # Concetenate kmer frequencies into single dataframe
    all_files = glob.glob(os.path.join(args.input_dir, "*.kf"))

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=None, sep=',')
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.set_index(0, inplace=True)

    query_func(args.input_dir, frame, args.model, args.classes, args.o)


def main():
    # Input arguments parser
    parser = argparse.ArgumentParser(description=' K-mer frequency to distance',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('-v', '--version', action='store_true', help='print the current version')
    # parser.add_argument('--debug', action='store_true', help='Print the traceback when an exception is raised')
    subparsers = parser.add_subparsers(title='commands',
                                       description='get_frequencies     Extract k-mer frequency from a reference genome-skims or assemblies\n'
                                                   'divide_tree        Divides input phylogeny into subtrees\n'
                                                   'train_classifier       Train classifier model based on backbone subtrees\n'
                                                   'classify       Classifies query samples using previously trained classifier model\n'
                                                   'get_distance       Compute distance matrices\n'
                                                   # 'train_model     Performs correction of subsampled distance matrices obtained for reference\n'
                                                   'train_model_set     Trains all models for subtrees consecutively\n'
                                                   'query     Query subtree models\n'
                                                   # ' genome-skims or assemblies'
                                       ,
                                       help='Run skmer {commands} [-h] for additional help',
                                       dest='{commands}')

    # Get_frequencies command subparser

    ### To invoke
    ### python main.py get_frequencies -input_dir /Users/nora/PycharmProjects/test_freq

    parser_freq = subparsers.add_parser('get_frequencies',
                                       description='Process a library of reference genome-skims or assemblies')
    parser_freq.add_argument('-input_dir',
                            help='Directory of input genomes or assemblies (dir of .fastq/.fq/.fa/.fna/.fasta files)')
    parser_freq.add_argument('-k', type=int, choices=list(range(6, 7)), default=7, help='K-mer length [6-7]. ' +
                                                                                         'Default: 7', metavar='K')
    parser_freq.add_argument('-p', type=int, choices=list(range(1, mp.cpu_count() + 1)), default=mp.cpu_count(),
                            help='Max number of processors to use [1-{0}]. '.format(mp.cpu_count()) +
                                 'Default for this machine: {0}'.format(mp.cpu_count()), metavar='P')
    parser_freq.set_defaults(func=get_frequencies)



    # Divide_tree command subparser

    ### To invoke
    ### python main.py divide_tree -size 850 -tree /Users/nora/PycharmProjects/astral.rand.lpp.r100.EXTENDED.nwk

    parser_div = subparsers.add_parser('divide_tree',
                                       description='Divides input phylogeny into subtrees.')
    parser_div.add_argument('-tree', help='Input phylogeny (a .newick/.nwk format)')
    parser_div.add_argument('-size', type=int, default=850, help='Size of the subtree. ' +
                                                                                         'Default: 850')
    parser_div.set_defaults(func=divide_tree)



    # Train_classifier command subparser

    ### To invoke
    ### python main.py train_classifier -input_dir /Users/nora/PycharmProjects/train_tree_kf -subtrees /Users/nora/PycharmProjects/my_test.subtrees -e 1 -o /Users/nora/PycharmProjects/my_toy_input

    parser_trclas = subparsers.add_parser('train_classifier',
                                        description='Train classifier model based on backbone subtrees')
    parser_trclas.add_argument('-input_dir',
                             help='Directory of input k-mer frequencies for assemblies or reads (dir of .kf files for backbone)')
    parser_trclas.add_argument('-subtrees', help='Classification file with subtrees information obtained from divide_tree command (a .subtrees format)')
    parser_trclas.add_argument('-e', type=int, choices=list(range(1, 20001)), default=4000, help='Epochs [1-20000]. ' +
                                                                                        'Default: 4000')
    parser_trclas.add_argument('-o',
                               help='Model output path/filename prefix')

    parser_trclas.set_defaults(func=train_classifier)




    # Classify command subparser

    ### To invoke
    ### python main.py classify -input_dir /Users/nora/PycharmProjects/test_tree_kf -model /Users/nora/PycharmProjects/my_toy_input  -o /Users/nora/PycharmProjects/my_toy_input


    parser_classify = subparsers.add_parser('classify',
                                          description='Classifies query inputs using previously trained classifier model')
    parser_classify.add_argument('-input_dir',
                               help='Directory of input k-mer frequencies for queries samples: assemblies or reads (dir of .kf files for queries)')
    parser_classify.add_argument('-model',
                               help='Classification model')
    parser_classify.add_argument('-output_classes',
                               help='Model output path/filename prefix')

    parser_classify.set_defaults(func=classify)



    # Get_distances command subparser

    ### To invoke
    ### python main.py get_distances -tree /Users/nora/PycharmProjects/test_tree.nwk  -subtrees  /Users/nora/PycharmProjects/my_test.subtrees

    parser_distances = subparsers.add_parser('get_distances',
                                            description='Computes distance matrices')
    parser_distances.add_argument('-tree', help='Input phylogeny (a .newick/.nwk format)', required=True)
    parser_distances.add_argument('-subtrees',
                               help='Classification file with subtrees information obtained from divide_tree command (a .subtrees format)')
    parser_distances.add_argument('-mode', type=str,
                                  choices={"hybrid", "subtrees_only"}, default="hybrid",
                                  help='Ways to perform distance computation [hybrid, subtrees_only]. ' +
                                                                                        'Default: hybrid')


    parser_distances.set_defaults(func=get_distances)



    # Train_model_set command subparser

    ### To invoke
    ### python main.py train_model_set -input_dir /Users/nora/PycharmProjects/train_tree_kf  -true_dist /Users/nora/PycharmProjects  -subtrees /Users/nora/PycharmProjects/my_test.subtrees -e 1 -o /Users/nora/PycharmProjects/my_toy_input

    parser_train_model_set = subparsers.add_parser('train_model_set',
                                            description='Trains individual models for each subtree')
    parser_train_model_set.add_argument('-input_dir',
                               help='Directory of input k-mer frequencies for assemblies or reads (dir of .kf files for backbone)')
    parser_train_model_set.add_argument('-true_dist',
                                        help='Directory of distamce matrices for backbone subtrees (dir of *subtree_INDEX.di_mtrx files for backbone)')
    parser_train_model_set.add_argument('-subtrees',
                               help='Classification file with subtrees information obtained from divide_tree command (a .subtrees format)')
    parser_train_model_set.add_argument('-e', type=int, choices=list(range(1, 20001)), default=4000, help='Epochs [1-20000]. ' +
                                                                                                 'Default: 4000')
    parser_train_model_set.add_argument('-o',
                               help='Model output path/filename prefix')

    parser_train_model_set.set_defaults(func=train_model_set)



    # Query_model_set command subparser

    ### To invoke
    ### python main.py query -input_dir /Users/nora/PycharmProjects/test_tree_kf  -model /Users/nora/PycharmProjects/my_toy_input  -classes /Users/nora/PycharmProjects/my_toy_input  -o /Users/nora/PycharmProjects/my_toy_input

    parser_query = subparsers.add_parser('query',
                                                   description='Query models')
    parser_query.add_argument('-input_dir',
                                        help='Directory of input k-mer frequencies for assemblies or reads (dir of .kf files for queries)')
    parser_query.add_argument('-model',
                                        help='Directory of models and embeddings (dir of model_subtree_INDEX.ckpt and embeddings_subtree_INDEX.csv files for backbone)')
    parser_query.add_argument('-classes',
                                        help='Path to classification file with subtrees information obtained from classify command (classes.out file)')
    parser_query.add_argument('-o',
                                        help='Model output path/filename prefix')

    parser_query.set_defaults(func=query)


    args = parser.parse_args()
    args.func(args)






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm oop')
    main()




