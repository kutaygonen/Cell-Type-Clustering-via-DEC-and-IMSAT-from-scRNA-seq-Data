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


data_sq = data_sq.iloc[:1448,:]
df = df.iloc[:1448,:]


train_data = data_sq.iloc[:850,:]
train_data_labels = df.iloc[:850,:]

test_data = data_sq.iloc[850:,:]
test_data_labels = df.iloc[850:,:]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

# cell_names = ['A549','H838','H2228','HCC827','H1975']
cell_names = ['alpha','beta','gamma','delta']


le.fit(cell_names)

test_data_labels= le.transform(np.ravel(test_data_labels))
train_data_labels=le.transform(np.ravel(train_data_labels))


class Dataset():
	def __init__(self, train, test):
		self.images_train, self.labels_train = train
		self.images_test, self.labels_test = test
		self.dataset_indices = np.arange(0, len(self.images_train))
		self.shuffle()

	def sample_minibatch(self, batchsize):
		batch_indices = self.dataset_indices[:batchsize]
		x_batch = self.images_train[batch_indices]
		y_batch = self.labels_train[batch_indices]
		self.dataset_indices = np.roll(self.dataset_indices, batchsize)
		return x_batch, y_batch

	def shuffle(self):
		np.random.shuffle(self.dataset_indices)

def load_mnist_whole():
    dataset = Dataset(train=(train_data, train_data_labels), test=(test_data, test_data_labels))
    return dataset







