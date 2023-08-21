#"Name","Number of legs","Color"
#"Lion","4","Yellow"
#"Monkey","4","Black"
#"Parrot","2","Green"
#"Snake","0","Green"
#"Bear","4","Black"

# Name of features, label
# Step1: Find the feature with the most information gain
# Step2: Split the dataset based on values of that feature. 
#   - if all values same stop splitting, 
#   - if no more features left assign the max 

import pandas as pd
import numpy as np

test_dataset = pd.read_csv('animals.csv')

# assume last index is label
labels = test_dataset.iloc[:,1]
label_name = test_dataset.keys()[0]
print(labels)
features = test_dataset.iloc[:,1:]
print(features)

# count of all current labels
label_count = labels.value_counts()

def get_entropy(label_col) -> float:
    # Entropy as defined as sum(-plogp)
    label_count = label_col.value_counts()
    total_labels = sum(label_count)*1.0
    entropy = 0.0
    for label in label_count.keys():
        count = label_count[label]
        p = count/total_labels
        entropy += -p*np.log(p)
    return entropy

def get_info_gain(orig_table, feature_name):
    labels = orig_table[label_name]

    orig_ent = get_entropy(labels)
    feature_values = orig_table[feature_name]
    all_uniq_feature_values = feature_values.value_counts()
    total_labels = len(labels)*1.0
    # Calculate entropy if you split on that unique feature value
    total_ent = 0.0
    for uniq_feature in all_uniq_feature_values.keys():
        uniq_feature_count = all_uniq_feature_values[uniq_feature]
        new_entropy = get_entropy(orig_table.loc[orig_table[feature_name] == uniq_feature, label_name])
        # add fraction to total entropy
        fraction = (uniq_feature_count/total_labels)*new_entropy
        total_ent += (uniq_feature_count/total_labels)*new_entropy
    info_gain = orig_ent - total_ent
    return info_gain

def find_feature_with_most_info_gain(orig_table):
    max_info_gain = -9999
    ret_feat = orig_table.keys()[1]
    for feature_name in orig_table.keys()[1:]:
        info_gain = get_info_gain(orig_table, feature_name)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            ret_feat = feature_name
    return ret_feat

def get_subtable(orig_table, feature_name, feature_value):
    return orig_table[orig_table[feature_name] == feature_value].reset_index(drop=True)


def build_tree(orig_table, root):
    label_name = test_dataset.keys()[0]
    
    feature_name = find_feature_with_most_info_gain(orig_table)

    if root is None:
        root = {}
        root[feature_name] = {}

    uniq_feature_val = orig_table[feature_name].unique()
    # Recursively construct the tree, stop if a subset has only 1 value
        
    for feature_val in uniq_feature_val:
        subtable = get_subtable(orig_table, feature_name, feature_val)
        label_vals = subtable[label_name].unique()
        num_labels = len(label_vals)
        if num_labels == 1: # leaf node
            root[feature_name][feature_val] = label_vals[0]
        if len(uniq_feature_val) == 1:
            root[feature_name][feature_val] = subtable[label_name].mode()[0]
        else:
            root[feature_name][feature_val] = build_tree(subtable, None)
    return root

def predict(tree, datapoint):
    node = tree
    while not isinstance(node, str):
        feature_name = list(node.keys())[0]
        feat_val = datapoint[feature_name]
        node = node[feature_name][feat_val]

    return node

print(get_entropy(test_dataset[label_name]))
print(get_info_gain(test_dataset, 'Number of legs'))
print(find_feature_with_most_info_gain(test_dataset))
tree = build_tree(test_dataset, root=None)
print(tree)
datapoint = {"Number of legs": 4, "Color": 'Black'}
print(predict(tree, datapoint))
