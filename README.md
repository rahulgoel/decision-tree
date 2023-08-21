# decision-tree

Contains a toy decision tree implementatin based on id3. 

We find the feature with most info gain and split the tree based on that feature. 

#"Name","Number of legs","Color"
#"Lion","4","Yellow"
#"Monkey","4","Black"
#"Parrot","2","Green"
#"Snake","0","Green"
#"Bear","4","Black"

tree =build_tree(test_dataset, root=None)
>>> print(tree)
{'Color': {'Yellow': {'Number of legs': {4: 'Lion'}}, 'Black': {'Number of legs': {4: 'Bear'}}, 'Green': {'Number of legs': {2: {'Number of legs': {2: 'Parrot'}}, 0: {'Number of legs': {0: 'Snake'}}}}}}
>>> datapoint = {"Number of legs": 4, "Color": 'Black'}
>>> print(predict(tree, datapoint))
Bear
