# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 23:44:00 2020

@author: Kutay
"""


import pandas as pd

df = pd.read_csv('Labels.csv')

number_of_unique_class = df["x"].describe()
number_of_unique_subclass = df["Subclass"].describe()
number_of_unique_cluster = df["cluster"].describe()


class_types = df["x"].value_counts()
subclass_types = df["Subclass"].value_counts()
cluster_types = df["cluster"].value_counts()