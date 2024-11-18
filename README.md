# Data Science HW3

## for training the model
* main.py

## You can create the environment with anaconda
```
conda create --name hw1 pip -y
conda activate hw1
```
## Install packages
* scipy, networkx
```
pip install scipy networkx
```
* dgl, pytorch
'''
* Install dgl
  * https://www.dgl.ai/pages/start.html

* Install pytorch
  * https://pytorch.org/get-started/locally/
  * recommend 2.2.0 version
'''

## Dataset
* Unknown graph data
  * Label Split:
    * Train: 60, Valid: 30, Test: 1000
* File name description
```
  dataset
  │   ├── private_features.pkl # node feature
  │   ├── private_graph.pkl # graph edges
  │   ├── private_num_classes.pkl # number of classes
  │   ├── private_test_labels.pkl # X
  │   ├── private_test_mask.pkl # nodes indices of testing set
  │   ├── private_train_labels.pkl # nodes labels of training set
  │   ├── private_train_mask.pkl # nodes indices of training set
  │   ├── private_val_labels.pkl # nodes labels of validation set
  │   └── private_val_mask.pkl # nodes indices of validation set
```

Semi-Supervised Graph Neural Network Training and Testing
Objective
The goal of this project was to develop a semi-supervised graph neural network (GNN) model using PyTorch and DGL for node classification tasks. The pipeline includes data preprocessing, training on labeled and unlabeled data, validation, applying early stopping, and generating predictions for test data.
The graph used in this project contains 20,000 nodes with more than 400 features each. However, only 90 nodes are annotated, including both training and validation samples. To address this, I implemented a semi-supervised learning approach. After training the model on the labeled data, I utilized the trained model to generate predictions for unlabeled nodes and calculated the total loss, assuming the predicted labels for the unlabeled nodes were reasonably accurate.
Model Implementation
To avoid overfitting, I determined that a simple model was sufficient for the data. Increasing the model's complexity might hinder generalization on public datasets. As a result, I used a single-layer graph convolutional network (GCN) with dropout and a hidden size of 32.
The following techniques were applied:
•	Early stopping: To prevent overfitting, the training was halted when validation performance plateaued.
•	Best model saving: The model with the lowest validation loss and highest accuracy was saved.
•	Cross-entropy loss: Used as the loss function for classification tasks.
________________________________________
Graph Transformations
Preprocessing steps were performed using DGL to enhance the graph structure:
Random Walk Positional Encoding
o	I applied Random Walk Positional Encoding with k=16 to encode nodes with positional information.
o	A random walk is a mathematical process involving a sequence of steps between neighboring nodes in a graph. It captures the connectivity and local structure of the graph, enabling the model to distinguish nodes with similar features but differing graph contexts. By aggregating statistics from multiple random walks, the positional information was embedded into the graph.
Fail Attempts:
The implementation of the Graph Convolutional Network with calculating contrastive loss didn’t improve accuracy. Initial attempts to compute loss for pseudo-labeled data led to inconsistencies due to insufficiently defined random sampling and edge cases where unlabeled masks were sparse. Contrastive loss calculations also encountered issues in maintaining a balance between positive and negative pairs, resulting in unstable gradients.
