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

test_dataset = pd.read_csv('animals.csv')
#print(test_dataset)

# assume last index is label
labels = test_dataset.iloc[:,-1]
print(labels)
features = test_dataset.iloc[:,:-1]
print(features)
