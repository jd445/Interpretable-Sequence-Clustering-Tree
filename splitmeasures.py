import numpy as np


def relative_risk_index(label, hit_index):
    # calculate the relative risk index on multi-class,label is a list,hit_index is a list,using one vs rest
    # 1. get the majority class of hit_index
    label_hit = [label[i] for i in hit_index]
    majority_class = max(set(label_hit), key=label_hit.count)
    
    # 2. using one vs rest to calculate the relative risk index
    contingency_table = np.zeros((2, 2))
    # add a small number to avoid 0
    contingency_table[0][0] = 0.1
    contingency_table[0][1] = 0.1
    contingency_table[1][0] = 0.1
    contingency_table[1][1] = 0.1
    for i in range(len(label)):
        if i in hit_index:
            if label[i] == majority_class:
                contingency_table[0][0] += 1
            else:
                contingency_table[0][1] += 1
        else:
            if label[i] == majority_class:
                contingency_table[1][0] += 1
            else:
                contingency_table[1][1] += 1
    # 2.1 calculate the relative risk index of majority class
    # 2.1.1 calculate the number of majority class
    relative_risk = (contingency_table[0][0] / (contingency_table[0][0] + contingency_table[0][1])) / (contingency_table[1][0] / (contingency_table[1][0] + contingency_table[1][1]))
    return relative_risk
def relative_risk_p(label, hit_index, label_size,lebel_set):
    # calculate the relative risk index on multi-class,label is a list,hit_index is a list,using one vs rest
    # 1. get the majority class of hit_index

    label_hit = np.array(label)[hit_index]
    unique_labels, counts = np.unique(label_hit, return_counts=True)
    show_ratio = []
    for labels in lebel_set:
        if labels in unique_labels:
            show_ratio.append(counts[np.where(unique_labels==labels)][0]/label_size[labels])
        else:
            show_ratio.append(0)
    show_ratio = np.array(show_ratio)
    majority_class = list(lebel_set)[np.argmax(show_ratio)]

    
    # 2. using one vs rest to calculate the relative risk index
    contingency_table = np.zeros((2, 2))
    contingency_table[0][0] = 0.1
    contingency_table[0][1] = 0.1
    contingency_table[1][0] = 0.1
    contingency_table[1][1] = 0.1
    
    for i in range(len(label)):
        if i in hit_index:
            if label[i] == majority_class:
                contingency_table[0][0] += 1
            else:
                contingency_table[0][1] += 1
        else:
            if label[i] == majority_class:
                contingency_table[1][0] += 1
            else:
                contingency_table[1][1] += 1
    # for i in range(len(label)):
    #     if i in hit_index:
    #         if label[i] == majority_class:
    #             contingency_table[0][0] += 1
    #         else:
    #             contingency_table[0][1] += 1
    #     else:
    #         if label[i] == majority_class:
    #             contingency_table[1][0] += 1
    #         else:
    #             contingency_table[1][1] += 1
    # 2.1 calculate the relative risk index of majority class
    # 2.1.1 calculate the number of majority class
    # relative_risk = (contingency_table[0][0] / (contingency_table[0][0] + contingency_table[0][1])) / (contingency_table[1][0] / (contingency_table[1][0] + contingency_table[1][1]))
    relative_risk = (contingency_table[0][0] / (contingency_table[0][0] + contingency_table[0][1])) / (contingency_table[1][0] / (contingency_table[1][0] + contingency_table[1][1]))
    return relative_risk


import numpy as np
def odd(label, hit_index):
    # calculate the relative risk index on multi-class,label is a list,hit_index is a list,using one vs rest
    # 1. get the majority class of hit_index
    label_hit = [label[i] for i in hit_index]
    majority_class = max(set(label_hit), key=label_hit.count)
    # 2. using one vs rest to calculate the relative risk index
    contingency_table = np.zeros((2, 2))
    contingency_table[0][0] = 0.5
    contingency_table[0][1] = 0.5
    contingency_table[1][0] = 0.5
    contingency_table[1][1] = 0.5
    for i in range(len(label)):
        if i in hit_index:
            if label[i] == majority_class:
                contingency_table[0][0] += 1
            else:
                contingency_table[0][1] += 1
        else:
            if label[i] == majority_class:
                contingency_table[1][0] += 1
            else:
                contingency_table[1][1] += 1
    # 2.1 calculate the relative risk index of majority class
    # 2.1.1 calculate the number of majority class
    odd = (contingency_table[0][0]*contingency_table[1][1])/(contingency_table[0][1]*contingency_table[1][0])   
    return odd
def GiniIndex(label, label_after_split):
    # calculate the gini index
    # label_after_split = [label[i] for i in hit_label_index]
    gini_index = 1 - sum([(label.count(i) / len(label))
                         ** 2 for i in set(label)])
    # get the remain_label from label: label - label_after_split
    temp_label = label.copy()
    temp_label_after_split = label_after_split.copy()
    remain_label = []
    while temp_label_after_split is not None:
        if len(temp_label_after_split) == 0:
            remain_label = temp_label
            break
        else:
            remain_label.append(temp_label.pop(
                temp_label.index(temp_label_after_split.pop(0))))
    # calculate the gini index after split
    discriminitive = gini_index - len(label_after_split) / len(label) * (1 - sum([(label_after_split.count(i) / len(label_after_split)) ** 2 for i in set(
        label_after_split)])) - (len(remain_label)) / len(label) * (1 - sum([(remain_label.count(i) / len(remain_label)) ** 2 for i in set(remain_label)]))
    return discriminitive
def gain_ratio(label, hit_index):
    # Calculate the gain ratio on multi-class using one vs rest
    # 1. Get the majority class of hit_index
    label_hit = [label[i] for i in hit_index]
    majority_class = max(set(label_hit), key=label_hit.count)
    # 2. Calculate the contingency table
    contingency_table = np.zeros((2, 2))
    contingency_table[0][0] = 0.5
    contingency_table[0][1] = 0.5
    contingency_table[1][0] = 0.5
    contingency_table[1][1] = 0.5
    for i in range(len(label)):
        if i in hit_index:
            if label[i] == majority_class:
                contingency_table[0][0] += 1
            else:
                contingency_table[0][1] += 1
        else:
            if label[i] == majority_class:
                contingency_table[1][0] += 1
            else:
                contingency_table[1][1] += 1
    # 3. Calculate the gain ratio
    # 3.1 Calculate the information gain
    p_hit = len(hit_index) / len(label)  # Probability of hit
    p_miss = 1 - p_hit  # Probability of miss
    entropy_before = -p_hit * np.log2(p_hit) - p_miss * np.log2(p_miss)  # Entropy before split
    p_hit_majority = contingency_table[0][0] / (contingency_table[0][0] + contingency_table[0][1])
    p_miss_majority = contingency_table[1][0] / (contingency_table[1][0] + contingency_table[1][1])
    entropy_after = -p_hit_majority * np.log2(p_hit_majority) - p_miss_majority * np.log2(p_miss_majority)  # Entropy after split
    information_gain = entropy_before - (p_hit * entropy_after)  # Information gain
    # 3.2 Calculate the intrinsic value
    num_classes = len(set(label))
    intrinsic_value = -p_hit * np.log2(p_hit) - p_miss * np.log2(p_miss)  # Intrinsic value
    # 3.3 Calculate the gain ratio
    gain_ratio = information_gain / intrinsic_value
    return gain_ratio


def odd_p(label, hit_index, label_size,lebel_set):
    # calculate the relative risk index on multi-class,label is a list,hit_index is a list,using one vs rest
    # 1. get the majority class of hit_index

    label_hit = np.array(label)[hit_index]
    unique_labels, counts = np.unique(label_hit, return_counts=True)
    show_ratio = []
    for labels in lebel_set:
        if labels in unique_labels:
            show_ratio.append(counts[np.where(unique_labels==labels)][0]/label_size[labels])
        else:
            show_ratio.append(0)
    show_ratio = np.array(show_ratio)
    majority_class = list(lebel_set)[np.argmax(show_ratio)]


    
    # 2. using one vs rest to calculate the relative risk index
    contingency_table = np.zeros((2, 2))
    contingency_table[0][0] = 0.1
    contingency_table[0][1] = 0.1
    contingency_table[1][0] = 0.1
    contingency_table[1][1] = 0.1
    
    for i in range(len(label)):
        if i in hit_index:
            if label[i] == majority_class:
                contingency_table[0][0] += 1
            else:
                contingency_table[0][1] += 1
        else:
            if label[i] == majority_class:
                contingency_table[1][0] += 1
            else:
                contingency_table[1][1] += 1
    # for i in range(len(label)):
    #     if i in hit_index:
    #         if label[i] == majority_class:
    #             contingency_table[0][0] += 1
    #         else:
    #             contingency_table[0][1] += 1
    #     else:
    #         if label[i] == majority_class:
    #             contingency_table[1][0] += 1
    #         else:
    #             contingency_table[1][1] += 1
    # 2.1 calculate the relative risk index of majority class
    # 2.1.1 calculate the number of majority class
    # odds = (contingency_table[0][0]*contingency_table[1][1])/(contingency_table[0][1]*contingency_table[1][0])
    if contingency_table[0][1] == 0 or contingency_table[1][0] == 0:
        odds = 0
    else:
        odds = (contingency_table[0][0]*contingency_table[1][1])/(contingency_table[0][1]*contingency_table[1][0])
    return odds
