import scprep
import os
import numpy as np
# import pdb #For debugging

#====================LABELS================
import pandas as pd

df = pd.read_csv('Labels_NEW.csv')

number_of_unique_class = df["x"].describe()
class_types = df["x"].value_counts()

#========================PREPROCESSING================
data_uf = scprep.io.load_csv('10x_5cl_data.csv' ,sparse=True)
scprep.plot.plot_library_size(data_uf)


# Filtering Cells By Library Size
data_uf_2 = scprep.filter.filter_library_size(data_uf, cutoff=100000, keep_cells='below') 
data = scprep.filter.filter_library_size(data_uf_2, cutoff=10000, keep_cells='above') 
scprep.plot.plot_library_size(data)
genes_per_cell = np.sum(data > 0, axis=0)
data_rem = scprep.filter.remove_rare_genes(data_uf, cutoff=0, min_cells=5)
data_ln = scprep.normalize.library_size_normalize(data_rem)
data_sq = scprep.transform.sqrt(data_ln)


#========================DEEP EMBEDDED CLUSTERING================

from keras_dec import DeepEmbeddingClustering
# from keras.datasets import mnist
import numpy as np

'''
from sklearn.model_selection import train_test_split
train_data,train_data_labels,test_data,test_data_labels = train_test_split(data_sq,df,test_size = 0.5, random_state=42)

X = np.concatenate((train_data, test_data), axis = 1)
Y = np.concatenate((train_data_labels, test_data_labels), axis = 1)
'''

data_sq = data_sq.iloc[:3802,:]
df = df.iloc[:3802,:]


train_data = data_sq.iloc[:2000,:]
train_data_labels = df.iloc[:2000,:]

test_data = data_sq.iloc[1802:,:]
test_data_labels = df.iloc[1802:,:]



c = DeepEmbeddingClustering(n_clusters=5,input_dim = 11778,alpha=0.125) #Model oluşturma


'''
# =====================MNIST PARALELINDE===========0
c.initialize(data_sq,layerwise_pretrain_iters=500, finetune_iters=250) #layerwise pretrain,Finetuning autoencoder

#=======NEW DATASET LABELS========
df_last = df.replace('A549',3)
df_last = df_last.replace('H838',0)
df_last = df_last.replace('H2228',1)
df_last = df_last.replace('HCC827',2)
df_last = df_last.replace('H1975',4)

df_last = df_last.to_numpy(dtype='int')


c.cluster(data_sq,y=df_last) 
final_output = c.y_pred

#DEC Confusion Matrix and ACC
acc_out = c.cluster_acc(final_output,df_last)

'''

# ==============TRAIN TEST SPLIT==============

c.initialize(train_data,layerwise_pretrain_iters=1000, finetune_iters=500) #layerwise pretrain,Finetuning autoencoder

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

cell_names = ['A549','H838','H2228','HCC827','H1975']

le.fit(cell_names)

test_data_labels_last = le.transform(test_data_labels)


c.cluster(test_data,y=test_data_labels_last,iter_max=10) 
final_output = c.y_pred


predicted_cells = le.inverse_transform(final_output)

#DEC Confusion Matrix and ACC
acc_out = c.cluster_acc(final_output,test_data_labels_last)

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test_data_labels_last, final_output))

cluster_centers = c.cluster_centres

#===============================VISUALIZATION===========================
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt

pca = PCA(n_components=2)
x_pca = pca.fit_transform(encoded)

#===============================2D===========================

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
x_tsne = tsne.fit_transform(encoded)

plt.scatter(x_tsne[final_output==0,0],x_tsne[final_output==0,1],s=10)
plt.scatter(x_tsne[final_output==1,0],x_tsne[final_output==1,1],s=10)
plt.scatter(x_tsne[final_output==2,0],x_tsne[final_output==2,1],s=10)
plt.scatter(x_tsne[final_output==3,0],x_tsne[final_output==3,1],s=10)


pca = PCA(n_components=3)
x_pca = pca.fit_transform(encoded)

#===============================3D===========================
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(x_tsne[final_output==0,0],x_tsne[final_output==0,1],x_tsne[final_output==0,2])
ax.scatter(x_tsne[final_output==1,0],x_tsne[final_output==1,1],x_tsne[final_output==1,2])
ax.scatter(x_tsne[final_output==2,0],x_tsne[final_output==2,1],x_tsne[final_output==2,2])
ax.scatter(x_tsne[final_output==3,0],x_tsne[final_output==3,1],x_tsne[final_output==3,2])
ax.scatter(x_tsne[final_output==4,0],x_tsne[final_output==4,1],x_tsne[final_output==4,2])

plt.show()


'''
#=======OLD DATASET LABELS========

df_last = df.replace('delta',1)
df_last = df_last.replace('beta',2)
df_last = df_last.replace('ductal',4)
df_last = df_last.replace('alpha',10)
df_last = df_last.replace('quiescent_stellate',0)
df_last = df_last.replace('endothelial',3)
df_last = df_last.replace('gamma',5)
df_last = df_last.replace('macrophage',12)
df_last = df_last.replace('activated_stellate',11)
df_last = df_last.replace('B_cell',9)
df_last = df_last.replace('immune_other',6)
df_last = df_last.replace('T_cell',8)
df_last = df_last.replace('schwann',7)

df_last = df_last.to_numpy(dtype='int')


test_data_labels_last = test_data_labels.replace('A549',3)
test_data_labels_last = test_data_labels_last.replace('H838',4)
test_data_labels_last = test_data_labels_last.replace('H2228',2)
test_data_labels_last = test_data_labels_last.replace('HCC827',1)
test_data_labels_last = test_data_labels_last.replace('H1975',0)

test_data_labels_last = test_data_labels_last.to_numpy(dtype='int')


'''
