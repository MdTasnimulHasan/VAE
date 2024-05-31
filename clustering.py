import time
import warnings
from itertools import cycle, islice

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

import pandas as pd
import ast
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--src_filename', type = str, default = 'E://Tasnim//SonyCSL//input_data.csv', help='source txt file')
parser.add_argument('--output_filename', type = str, default = 'E://Tasnim//SonyCSL//clustered_data.csv', help='output_file')
parser.add_argument('--cluster_algorithm', type = str, default = 'gaussian_mixture', help='kmeans, agglomerative_clustering, gaussian_mixture')
parser.add_argument('--no_of_clusters', type = int, default = 10, help='no of clusters')

args = parser.parse_args()


df = pd.read_csv(args.src_filename, sep="\t", header=None)
# print((df[0][1]))

sample_id= []
x = []
y = []
for i in range (1, len(df), 1):
    str_data = df[0][i].split(',')
    # print(str_data)
    sample_id.append(str_data[0])
    x.append(float(str_data[1]))
    y.append(float(str_data[2]))


data = np.column_stack((x, y))



if args.cluster_algorithm == 'kmeans':
    
    ### K means clustering
    
    k = args.no_of_clusters
    
    # Create a KMeans instance
    kmeans = KMeans(n_clusters=k)
    
    # Fit the model to the data
    kmeans.fit(data)
    
    # Get the cluster labels
    labels = kmeans.labels_
    kmean_labels = labels
    
    # Get the cluster centroids
    centroids = kmeans.cluster_centers_
    
    print("Cluster Labels:", labels)
    print("Cluster Centroids:", centroids)
    
    # Plot the data points and centroids
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-Means Clustering')
    
    plt.show()


### Agglomerative (Hierarchical) Clustering

if args.cluster_algorithm == 'agglomerative_clustering':
    
    # Define the number of clusters
    n_clusters = args.no_of_clusters
    
    # Create an AgglomerativeClustering instance
    agglomerative_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    
    # Fit the model to the data
    agglomerative_clustering.fit(data)
    
    # Get the cluster labels
    labels = agglomerative_clustering.labels_
    agglomerative_clustering_labels = labels
    
    print("Cluster Labels:", labels)
    
    # Plot the data points with cluster labels
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Agglomerative Clustering')
    
    plt.show()


### Gaussian Mixture clustering
if args.cluster_algorithm == 'gaussian_mixture':

    
    # Define the number of clusters
    n_components = args.no_of_clusters
    
    # Create a GaussianMixture instance
    gmm = GaussianMixture(n_components=n_components)
    
    # Fit the model to the data
    gmm.fit(data)
    
    # Get the cluster labels
    labels = gmm.predict(data)
    gaussian_mixture_labels = labels
    
    # Get the cluster probabilities
    probabilities = gmm.predict_proba(data)
    
    print("Cluster Labels:", labels)
    print("Cluster Probabilities:\n", probabilities)
    
    # Plot the data points with cluster labels
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
    
    # Plot the cluster centers
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', s=100)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Gaussian Mixture Clustering')
    
    plt.show()


print(len(sample_id), len(x), len(y), len(labels))
df_cluster = pd.DataFrame({'id': sample_id, 'x': x, 'y': y, args.cluster_algorithm: labels})
df_cluster.to_csv(args.output_filename, index=False)

