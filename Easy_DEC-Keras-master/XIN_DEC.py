# -*- coding: utf-8 -*-
"""
Created on Sun May  2 14:58:08 2021

@author: Kutay
"""


import scprep
import os
import numpy as np
# import pdb #For debugging

#====================LABELS================
import pandas as pd

df = pd.read_csv('Labels_XIN.csv')

number_of_unique_class = df["x"].describe()
class_types = df["x"].value_counts()

#========================PREPROCESSING================
data_uf = scprep.io.load_csv('Filtered_Xin_HumanPancreas_data.csv' ,sparse=True)
scprep.plot.plot_library_size(data_uf)


# Filtering Cells By Library Size
data_uf_2 = scprep.filter.filter_library_size(data_uf, cutoff=1e6/1.5, keep_cells='below') 
data = scprep.filter.filter_library_size(data_uf_2, cutoff=1e6/2.5, keep_cells='above') 
scprep.plot.plot_library_size(data)
genes_per_cell = np.sum(data > 0, axis=0)
data_rem = scprep.filter.remove_rare_genes(data_uf, cutoff=0, min_cells=10)
data_ln = scprep.normalize.library_size_normalize(data_rem)
data_sq = scprep.transform.sqrt(data_ln)

from keras_dec import DeepEmbeddingClustering
# from keras.datasets import mnist
import numpy as np

#Data Prep
data_sq = data_sq.iloc[:1448,:]
df = df.iloc[:1448,:]


train_data = data_sq.iloc[:724,:]
train_data_labels = df.iloc[:724,:]

test_data = data_sq.iloc[724:,:]
test_data_labels = df.iloc[724:,:]



c = DeepEmbeddingClustering(n_clusters=4,input_dim = 24391) #Model oluşturma
c.initialize(train_data,layerwise_pretrain_iters=80, finetune_iters=40) #layerwise pretrain,Finetuning autoencoder


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

cell_names = ['alpha','beta','delta','gamma']

le.fit(cell_names)

test_data_labels_last = le.transform(np.ravel(test_data_labels))

c.cluster(test_data,y=test_data_labels_last,iter_max=10) 

final_output = c.y_pred


predicted_cells = le.inverse_transform(final_output)
#DEC Confusion Matrix and ACC
acc_out = c.cluster_acc(final_output,test_data_labels_last)
print('Accuracy = {}'.format(acc_out[0]))

# from sklearn.metrics import precision_score

# precision = precision_score(test_data_labels, predicted_cells, average='macro')
# print('Precision = {}'.format(precision))

from sklearn.metrics.cluster import adjusted_mutual_info_score
adj_m = adjusted_mutual_info_score(test_data_labels_last, final_output)
print('Adjusted Mutual Information = {}'.format(adj_m))

from sklearn.metrics.cluster import adjusted_rand_score
adj_r = adjusted_rand_score(test_data_labels_last, final_output)
print('Adjusted Random Score = {}'.format(adj_r))

cell_names_true = ['beta','alpha','delta','gamma']

cluster_centers = c.cluster_centres
encoded = c.encoder.predict(test_data)

#===============================VISUALIZATION===========================

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


#===============================2D===========================

tsne = TSNE(n_components=2)
x_tsne = tsne.fit_transform(encoded)

plt.scatter(x_tsne[final_output==0,0],x_tsne[final_output==0,1],s=10)
plt.scatter(x_tsne[final_output==1,0],x_tsne[final_output==1,1],s=10)
plt.scatter(x_tsne[final_output==2,0],x_tsne[final_output==2,1],s=10)
plt.scatter(x_tsne[final_output==3,0],x_tsne[final_output==3,1],s=10)


plt.legend(cell_names_true)
plt.xlabel('TSNE_1')
plt.ylabel('TSNE_2')

#===============================3D===========================

tsne = TSNE(n_components=3)
x_tsne = tsne.fit_transform(encoded)

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(x_tsne[final_output==0,0],x_tsne[final_output==0,1],x_tsne[final_output==0,2])
ax.scatter(x_tsne[final_output==1,0],x_tsne[final_output==1,1],x_tsne[final_output==1,2])
ax.scatter(x_tsne[final_output==2,0],x_tsne[final_output==2,1],x_tsne[final_output==2,2])
ax.scatter(x_tsne[final_output==3,0],x_tsne[final_output==3,1],x_tsne[final_output==3,2])
ax.legend(cell_names_true)
ax.set_xlabel('TSNE_1')
ax.set_ylabel('TSNE_2')
ax.set_zlabel('TSNE_3')
plt.show()

















