# Data Science HW3

## for training the Semisupervised model, run:
* main.py
## for training the OGC model, run:
ogc_main.py

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

Semi-Supervised Graph Neural Network 
I. Objective: 
The goal of this project was to develop a semi-supervised graph neural network 
(GNN) model using PyTorch and DGL for node classification tasks. The pipeline 
includes data preprocessing, training on labeled and unlabeled data, validation, 
applying early stopping, and generating predictions for test data. 
II. Approaches: 
1. Semi-Supervised Graph Neural Network 
The graph used in this project contains 20,000 nodes with over 500 features each. 
However, only 90 nodes are annotated, including both training and validation 
samples. To address this, I implemented a semi-supervised learning approach. 
After training the model on the labeled data, I utilized the trained model to 
generate predictions for unlabeled nodes and calculated the total loss, assuming the 
predicted labels for the unlabeled nodes were reasonably accurate. 
1.1. Model Implementation 
To avoid overfitting, I determined that a simple model was sufficient for the data. 
Increasing the model's complexity might hinder generalization on public datasets. 
As a result, I used a single-layer graph convolutional network (GCN) with dropout 
and a hidden size of 32. 
The following techniques were applied: 
• Early stopping: To prevent overfitting, the training was halted when 
validation performance plateaued. 
• Best model saving: The model with the lowest validation loss and highest 
accuracy was saved. 
• Cross-entropy loss: Used as the loss function for classification tasks. 
1.2. Graph Transformations 
Preprocessing steps were performed using DGL to enhance the graph structure 
using Random Walk Positional Encoding: 
o A random walk is a mathematical process involving a sequence of 
steps between neighboring nodes in a graph. It captures the 
connectivity and local structure of the graph, enabling the model to 
distinguish nodes with similar features but differing graph contexts. 
By aggregating statistics from multiple random walks, the positional 
information was embedded into the graph. 
o I applied Random Walk Positional Encoding with k=16 to encode 
nodes with positional information 
The final model achieves 82.69% accuracy on the public dataset and 79.52% on 
the private dataset. 
1.3. Fail Attempts 
The implementation of the Graph Convolutional Network with calculating 
contrastive loss didn’t improve accuracy. Initial attempts to compute loss for 
pseudo-labeled data led to inconsistencies due to insufficiently defined random 
sampling and edge cases where unlabeled masks were sparse. Contrastive loss 
calculations also encountered issues in maintaining a balance between positive and 
negative pairs, resulting in unstable gradients. 
2. Optimized Graph Convolution (OGC) 
The paper "From Cluster Assumption to Graph Convolution: Graph-based Semi
Supervised Learning Revisited" revisits the relationship between traditional graph
based semi-supervised learning (GSSL) methods and graph convolutional 
networks (GCNs) The authors highlight a key issue in GCNs: they may not 
effectively combine the graph structure and label information at each layer, which 
is crucial for improving performance. To address this, the paper introduces three 
new graph convolution methods, including a supervised method (OGC) and two 
unsupervised methods (GGC and GGCM), which aim to preserve graph structure 
while enhancing label learning. These methods are built on novel operators—
 Supervised EmBedding (SEB) and Inverse Graph Convolution (IGC)—and are 
shown to outperform existing GCN-based approaches. 
2.1. OGC (Optimized Graph Convolution) Class 
Implements the core model that integrates graph convolution and supervised 
learning for node classification. The model optimizes node embeddings through a 
combination of a graph-based loss (LGC) and a supervised loss (SEB). 
The OGC model combines graph-based convolution and supervised learning to 
optimize node embeddings for classification tasks. It consists of two primary 
modules: the LinearNeuralNetwork and the OGC class. The LinearNeuralNetwork 
is a simple classifier that performs a linear transformation of node features to 
predict the node's class. It uses a sparse adjacency matrix to propagate node 
features through neighboring nodes and update the embeddings. The supervised 
loss is calculated via the classifier, and the embeddings are adjusted accordingly. 
The model uses training masks and labels for supervised learning, while 
embedding updates combine graph structure and label information. The final goal 
is to optimize the node embeddings for better classification performance by 
adjusting the embeddings using both graph convolution and supervised loss. 
2.2. Training 
The training process uses a combination of LGC (graph-based loss) and SEB 
(supervised loss) to update the embeddings and improve the classification 
performance. The model is trained with Adam optimization, and the embeddings 
are updated in each iteration using the update_embeds method. This method 
applies a graph convolution-based update for smoothing the embeddings through 
the graph structure and a supervised loss function to improve the accuracy of node 
classification. 
This model helps achieve 83.70% accuracy on the public dataset and 81.90% on 
the private dataset. 
III. Conclusion 
In this project, I implemented a Semi-Supervised Graph Neural Network (GNN) 
model for node classification tasks, utilizing PyTorch and DGL. By focusing on a 
simple yet effective model architecture, I applied early stopping to prevent 
overfitting and used random walk positional encoding to enhance the graph 
structure. The final model achieved a solid performance, with an accuracy of 
82.69% on the public dataset and 79.52% on the private dataset. 
I also explored Optimized Graph Convolution (OGC), a method designed to 
better combine graph structure and label information, resulting in more accurate 
node embeddings. By integrating graph-based and supervised loss functions, the 
OGC model demonstrated a further improvement, achieving 83.70% accuracy on 
the public dataset and 81.90% on the private dataset. 
While some approaches, such as using contrastive loss for pseudo-labeled data, did 
not yield improvements due to issues with sampling and gradient stability, the 
OGC model showed promising results in node classification, proving the 
effectiveness of combining graph convolution with supervised learning in a semi
supervised framework. This work highlights the potential of semi-supervised 
GNNs in real-world applications where labeled data is limited and provides a solid 
foundation for future research in graph-based machine learning models. 
IV: References: 
[1]: Wang, Z., Ding, H., Pan, L., Li, J., Gong, Z. and Philip, S.Y., 2024. From 
cluster assumption to graph convolution: Graph-based semi-supervised learning 
revisited. IEEE Transactions on Neural Networks and Learning Systems. 
[2]: https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogc 
[3]: https://github.com/zhengwang100/ogc_ggcm