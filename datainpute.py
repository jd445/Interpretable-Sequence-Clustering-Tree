import math
import multiprocessing
import random
import time
from joblib import Parallel, delayed
import numpy as np


def datainput(filepath):
    # db: list form of a sequence
    # strdb: string form of a sequence
    # data_label: list of the label of the sequence
    # itemset: the itemset of the sequence
    # max_sequence_length: the maximum length of the sequence in a dataset.
    seqlength = []
    file = open(filepath)
    db = []
    data_label = []
    itemset = []
    for i in file:
        temp = i.replace("\n", "").split("\t")
        seq_db = temp[1].split(" ")
        seqlength.append(len(seq_db))
        db.append(seq_db)
        data_label.append(str(temp[0]))
    # unique itemset
    itemset = set([item for sublist in db for item in sublist])
    itemset = list(itemset)
    int_itemset = [str(x) for x in itemset]
    int_itemset.sort()
    itemset = [str(x) for x in int_itemset]
    # print(itemset)
    # save itemset as text for decoulpin
    pattern_length = 0
    if min(seqlength) > 10:
        pattern_length = min(seqlength)
    else:
        pattern_length = 5

    return db, data_label, itemset, pattern_length

if __name__ == '__main__':
    dataset = ['epitope', 'aslbu', 'gene', 'reuters', 'pioneer', 'context', 'robot', 'auslan2', 'epitope', 'skating', 'unix', 'question']
    for i in dataset:
        list_i_length = []
        db, data_label, itemset, max_sequence_length = datainput('dataset/{}.txt'.format(i))    
        for j in db:
            list_i_length.append(len(j))
        # write to the file
        with open('dataset/{}_length.txt'.format(i), 'w') as f:
            for item in list_i_length:
                f.write("%s \n" % item)
        f.close()
    db, data_label, itemset, max_sequence_length = datainput('dataset/{}.txt'.format('context'))

