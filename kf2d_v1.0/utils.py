import torch
import os
import pickle
import random
import itertools
import pandas as pd
import numpy as np
import argparse


def feature_normalization(features_txt):
    """
    Read input feature txt, convert counts to percentages per row
    to normalize for variable genomes lengths
    """
    coef = 1e+5                                                          # can try 1e+4

    data = pd.read_csv(features_txt, sep=" ", header=None)               # read txt file into dataframe
    data = data.dropna(axis='columns', how='all')                        # drop columns with all NAs
    try:
        data.iloc[:, 0] = data.iloc[:, 0].map(lambda x: x.split(".")[0]) # get rid of file extensions
    except:
        pass
    data = data.set_index(list(data.columns[[0]]))                       # set index to column 0
    data += 0.5                                                          # add +0.5 pseudocounts (will turn 0 to +0.5)
    data = data.div(data.sum(axis=1), axis=0)                            # normalize each row
    #data = np.log10(data+1)+10                                          # apply log 10+10
    print(data)

    #data.to_csv(r'/Users/admin/Documents/ml_meta/kmer_frequency/input_data/feat_all_exc_norm.csv', index=True, header = None)
    #data.to_csv("feat_matrix_10k_norm_7kC.csv", index = True, header = None)

    return data

def pdm_to_pickle(pdm, pickle_fname):
    """
    Save pdm to pickle
    """

    # create a binary pickle file
    f = open(pickle_fname,"wb")

    # write the python object (dict) to pickle file
    pickle.dump(pdm, f)

    # close file
    f.close()
    return

def get_label_idx_maps(data):
    """
    Create dictionaries to map sample_name to index and index to sample_name
    """

    # names = data.iloc[:, 0].values
    names = list(data.index.values)
    # print(names)

    # samples = list(map(lambda x: x.split(".")[0], names)) # file extensions were chopped before
    samples = names
    label_to_idx_dict = dict(map(reversed, enumerate(samples)))
    idx_to_label_dict = dict(enumerate(samples))

    # print('maps')
    # print(label_to_idx_dict)
    # print(idx_to_label_dict)

    return label_to_idx_dict, idx_to_label_dict


def read_dist_mtrx(true_dist_matrix):
    """
    Reading true distance dictionary from the file (pickled object)
    """
    with open(true_dist_matrix, 'rb') as handle:
        dtype = handle.read()

    print("Data type before reconstruction : ", type(dtype))

    # Reconstructing the data as dictionary
    d = pickle.loads(dtype)

    print("Data type after reconstruction : ", type(d))

    return d


def dict_to_df(dict_pdm):
    """
    Converts true distance matrix in dictionary format into dataframe
    Distance of leaf to itself will be filled with 0
    """
    print("Converting dictionary to dataframe ...")
    #df_try = DataFrame(pdm).T.fillna(0)
    df_pdm = pd.DataFrame.from_dict(dict_pdm, orient='index').fillna(0)

    #print(df_pdm)

    return df_pdm


def read_true_dist_mtrx(true_dist_matrix):
    """
    Reading true distance dictionary from the file
    Using DEPP matrix that already contains 0.0 on diagonal
    """

    df_pdm = pd.read_csv(true_dist_matrix, sep='\t', header=0, index_col=0)
    #print(df_pdm)

    return df_pdm

def read_feat_mtrx(true_dist_matrix):
    """
    Reading features dataframe from the file
    """

    df_pdm = pd.read_csv(true_dist_matrix, index_col=0, header=None, sep=',')
    #print(df_pdm)

    return df_pdm

# def label_index_dict(csv_file):
#     """
#     Creates label/sample_name to index dictionary
#     """
#     data = pd.read_csv(csv_file)
#     names = data.iloc[:, 0].values
#     samples = list(map(lambda x: x.split(".")[0], names))
#     label_dict = dict(map(reversed, enumerate(samples)))
#     #print(label_dict)
#
#     return label_dict


def sort_df(df_pdm, howToSortDict):
    """
    Reorder rows and columns in dataframe to match input by index
    Output: typecasted numpy array with correctly ordered
    """
    print("Reordering rows and columns of the dataframe ...")
    #print(howToSortDict)

    order = sorted(howToSortDict, key=lambda x: howToSortDict[x])
    #order = [o[0] for o in order]



    #print(order)

    # print(howToSortDict)
    # del howToSortDict['G000200715']
    # del howToSortDict['G001917965']
    # howToSortDict['G001917965']=1
    #howToSortDict.delete['G001917965']
    # print(howToSortDict)
    # order = sorted(howToSortDict, key=lambda x: howToSortDict[x])
    # print(order)

    # df = df.iloc[df.countries.map(howToSortDict).argsort()]
    ddf = df_pdm.reindex(index=order)

    if len(ddf.columns) > 1.0:
        ddf2 = ddf.reindex(columns=order)
    else:
        ddf2=ddf

    # print(ddf2)
    # ddf2.to_csv('ddf2.csv', index=True, sep=',')
    # print(len(ddf2))
    # print(len(ddf2.columns))

    # Verify sort correctness:

    # indicator = 0
    # print("Row, column, value")

    # for i in range (0, len(ddf2)):
    #     for j in range (0, len(ddf2.columns)):
    #         row_name = ddf2.iloc[j].index[i]
    #         col_name = ddf2.columns[j]
    #         value = ddf2.iat[i, j]
    #         #print(row_name, col_name, value)
    #         if value == df_pdm.at[row_name, col_name]:
    #             indicator +=1
    #             print(indicator)

    return ddf2.to_numpy()

def get_train_test_split(label_to_idx_dict, train_test_split, query_list):
    """
    Prepare train/test dataset split
    """

    # Compute counts based on required split
    N = len(label_to_idx_dict)
    N_train = int(N * train_test_split)
    N_test = N - N_train
    # N_val = N-(N_test+N_train)


    # Create dictionary partition
    random.seed(0)

    partition = {}
    # Test will be replaced with predetermined query list, but random for now
    #allnames = [label_to_idx_dict.keys()]

    #allnames = list(set([n[0] for n in label_to_idx_dict.keys()]))
    allnames = label_to_idx_dict.keys()


    if os.path.isfile(query_list):

        query_genomes  = list(set([line.strip() for line in open(query_list, 'r')]))
        #print(query_genomes)

        partition['test'] = [g for g in allnames if g in query_genomes]
        #print(partition['test'])

    else:
        partition['test'] = random.sample(list(allnames), N_test)

    partition['train'] = list(set(allnames) - set(partition['test']))

    # partition['train'] = list(range(0, N_train))
    # partition['test'] = list(range(N_train, N))
    # partition['val'] = list(range(N_test, N_val))

    #print(partition)

    return partition



def pairwise_train_dist(x):
    """
    Computes all pairwise L2 distances between rows of tensor
    Output: square matrix
    """
    t = torch.cdist(x, x, p=2, compute_mode='donot_use_mm_for_euclid_dist')

    return t


def pairwise_true_dist(labels, ddf2):
    """
    Subsets numpy array of true distances using indices from labels
    """
    r = ddf2[np.ix_(list(labels), list(labels))]

    return torch.from_numpy(r)


def select_indices(labels, ddf2):
    """
    Subsets torch tensor array using indices from labels
    """

    tmp = torch.index_select(ddf2, 0, torch.tensor(labels))
    r = torch.index_select(tmp, 1, torch.tensor(labels))

    return r

def collate_wrapper(batch):
    feat, labels = zip(*batch)

    feat_stacked = np.concatenate(([f for f in feat]), axis=0)
    labels_stacked = list(itertools.chain(*labels))
    #print("Dimensions of batch matrix rows: {}, cols: {}".format(np.shape(feat_stacked)[0], np.shape(feat_stacked)[1]))

    # print(feat_stacked)
    # print(labels_stacked)
    # print(type(batch))
    # print(len(batch))
    #return tuple(zip(*batch))
    return torch.from_numpy(feat_stacked), labels_stacked

def parse_args():

    parser = argparse.ArgumentParser(description='Script to generated per clade placement models.')
    parser.add_argument("-f", "--feat_tbl", help="Feature table.")
    parser.add_argument("-t", "--truth_tbl", help="Ground truth distance matrix.")
    parser.add_argument("-q", "--query_lst", help="List of queries.")
    parser.add_argument("-c", "--class_inf", help="Class information. Output from global model or true clades.")
    parser.add_argument("-d", "--clade_inf", help="Clade information. Result of splitting tree into clades.")
    parser.add_argument("-o", "--clade_total", help="Total number of clades. This requires to define model and perform classification.")
    args = parser.parse_args()

    inputfileone = args.feat_tbl  # plain filename
    inputfiletwo = args.truth_tbl
    inputfilethree = args.query_lst
    inputfilefour = args.class_inf
    inputfilefive = args.clade_inf
    inputfilesix = args.clade_total



    return inputfileone, inputfiletwo, inputfilethree, inputfilefour, inputfilefive, inputfilesix



def hms(seconds):
    """
    Converts elapsed time in seconds into H:M:S format
    """
    h = seconds // 3600
    m = seconds % 3600 // 60
    s = seconds % 3600 % 60

    return int(h), int(m), int(s)