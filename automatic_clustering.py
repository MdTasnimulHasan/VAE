
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

import time
import warnings
from itertools import cycle, islice

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

import pandas as pd
import ast
import os
import argparse

parser = argparse.ArgumentParser()

# parser.add_argument('--src_filename', type = str, default = 'E://Tasnim//SonyCSL//input_data.csv', help='source txt file')
parser.add_argument('--src_filename', type = str, default = 'E://Tasnim//SonyCSL//cVAE//cvae_output.txt', help='source txt file')
parser.add_argument('--output_filename', type = str, default = 'E://Tasnim//SonyCSL//clustered_data.csv', help='output_file')
parser.add_argument('--cluster_algorithm', type = str, default = 'gaussian_mixture', help='kmeans, agglomerative_clustering, gaussian_mixture')
parser.add_argument('--optimal_cluster_selection', type = str, default = 'davies_bouldin', help='elbow, silhouette, davies_bouldin, gap_statistic')
parser.add_argument('--no_of_clusters', type = int, default = 100, help='no of clusters')

args = parser.parse_args()



# # load data to np array

df = pd.read_csv(args.src_filename, sep="\t", header=None)
data = []
for i in range (0, len(df[2][:]), 1):
    str_2d_cleaned = df[2][i].replace("[[", "").replace("]]", "").split("], [")
    nested_list = [list(map(float, item.split())) for item in str_2d_cleaned]
    data.append(nested_list)   
data = np.array(data)
data = np.squeeze(data, axis=1)


# provided 2d data
# df = pd.read_csv(args.src_filename, sep="\t", header=None)
# sample_id= []
# x = []
# y = []
# for i in range (1, len(df), 1):
#     str_data = df[0][i].split(',')
#     # print(str_data)
#     sample_id.append(str_data[0])
#     x.append(float(str_data[1]))
#     y.append(float(str_data[2]))
# data = np.column_stack((x, y))





def KMeans_clustering(data, k):
    # Create a KMeans instance
    kmeans = KMeans(n_clusters=k)
    
    # Fit the model to the data
    kmeans.fit(data)
    
    # Get the cluster labels
    labels = kmeans.labels_
    kmean_labels = labels
    
    # Get the cluster centroids
    centroids = kmeans.cluster_centers_
    
    # print("Cluster Labels:", labels)
    # print("Cluster Centroids:", centroids)
    
    # Plot the data points and centroids
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'{args.cluster_algorithm} for optimal K ={k} ({args.optimal_cluster_selection})')
    
    plt.show()
    
    return

def GaussianMixture_clustering (data, k):
    
    # Create a GaussianMixture instance
    gmm = GaussianMixture(n_components=k)
    
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
    plt.title(f'{args.cluster_algorithm} for optimal K ={k} ({args.optimal_cluster_selection})')
    
    plt.show()
    
    
def Agglomerative_clustering (data, k):
    
    # Create an AgglomerativeClustering instance
    agglomerative_clustering = AgglomerativeClustering(n_clusters=k)
    
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
    plt.title(f'{args.cluster_algorithm} for optimal K ={k} ({args.optimal_cluster_selection})')
    
    plt.show()


'''
Methods for Determining the Optimal Number of Clusters
1. Elbow Method: The Elbow Method involves plotting the within-cluster sum of squares (WCSS, also known as inertia) against the number of clusters 𝐾 and looking for an "elbow point," where the rate of decrease slows.
2. Silhouette Analysis: The Silhouette Score measures how similar each point is to its own cluster compared to other clusters. It ranges from -1 to 1, where higher values indicate better-defined clusters.
3. Gap Statistic: 
4. Davies-Bouldin Index: 
5. X-Means Clustering
'''

def find_elbow_point(inertia):
    """
    Find the elbow point in the inertia values list.
    
    Args:
    inertia (list): List of inertia values for different values of K.
    
    Returns:
    int: The index of the elbow point in the inertia list.
    """
    elbow_point = 0
    max_reduction = 0
    
    for i in range(1, len(inertia)):
        reduction = inertia[i-1] - inertia[i]
        if reduction > max_reduction:
            max_reduction = reduction
            elbow_point = i
            
    return elbow_point + 1

def Elbow (X):
    
    def find_elbow_point(inertia):
        elbow_point = 0
        max_reduction = 0
        
        for i in range(1, len(inertia)):
            reduction = inertia[i-1] - inertia[i]
            if reduction > max_reduction:
                max_reduction = reduction
                elbow_point = i
                
        return elbow_point + 1
    
    inertia = []
    K = range(1, args.no_of_clusters)
    
    for k in K:
        if args.cluster_algorithm =='kmeans':
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
        elif args.cluster_algorithm =='gaussian_mixture':
            gmm = GaussianMixture(n_components=k)
            gmm.fit(X)
            labels = gmm.predict()
            # inertia.append()
    
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For optimal k')
    plt.show()
    elbow_index = find_elbow_point(inertia)
    return inertia, elbow_index

def Silhouette (X):
    silhouette_scores = []
    K = range(2, args.no_of_clusters)
    
    for k in K:
        if args.cluster_algorithm =='kmeans':
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)
        elif args.cluster_algorithm =='gaussian_mixture':
            gmm = GaussianMixture(n_components=k)
            gmm.fit(X)
            labels = gmm.predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        elif args.cluster_algorithm =='agglomerative_clustering':
            agg = AgglomerativeClustering(n_clusters=k)
            labels = agg.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)  
    
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method For Optimal k')
    plt.show()
    
    optimal_k = silhouette_scores.index(max(silhouette_scores))
    return silhouette_scores, optimal_k + 2

def davies_bouldin (X):
    
    # A lower Davies-Bouldin Index indicates better clustering.
    davies_bouldin_scores = []
    K = range(2, args.no_of_clusters)
    
    
    for k in K:
        if args.cluster_algorithm =='kmeans':
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            score = davies_bouldin_score(X, kmeans.labels_)
            davies_bouldin_scores.append(score)
        elif args.cluster_algorithm =='gaussian_mixture':
            gmm = GaussianMixture(n_components=k)
            gmm.fit(X)
            labels = gmm.predict(X)
            score = davies_bouldin_score(X, labels)
            davies_bouldin_scores.append(score)
        elif args.cluster_algorithm =='agglomerative_clustering':
            agg = AgglomerativeClustering(n_clusters=k)
            labels = agg.fit_predict(X)
            score = davies_bouldin_score(X, labels)
            davies_bouldin_scores.append(score)
    
    plt.plot(K, davies_bouldin_scores, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Index For Optimal k')
    plt.show() 
    
    optimal_k = davies_bouldin_scores.index(min(davies_bouldin_scores))
    return davies_bouldin_scores, optimal_k + 2


def gap_statistic(X, nrefs=5, maxClusters=args.no_of_clusters):
    gaps = np.zeros(maxClusters)
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})

    for k in range(1, maxClusters+1):
        # Fit KMeans and calculate Wk
        kmeans = KMeans(n_clusters=k).fit(X)
        Wk = np.log(kmeans.inertia_)

        # Create reference datasets
        Wkbs = np.zeros(nrefs)
        for i in range(nrefs):
            randomReference = np.random.random_sample(size=X.shape)
            kmeans = KMeans(n_clusters=k, random_state=42).fit(randomReference)
            Wkbs[i] = np.log(kmeans.inertia_)

        gap = np.mean(Wkbs) - Wk
        gaps[k-1] = gap
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
    
    # Identify the optimal number of clusters
    optimal_k = np.argmax(gaps) + 1
    plt.plot(range(1, len(gaps)+1), gaps, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Gap Statistic')
    plt.title('Gap Statistic Method For Optimal k')
    plt.show()
    return gaps, resultsdf, optimal_k



if args.cluster_algorithm == 'kmeans':
    
    if args.optimal_cluster_selection == 'elbow':
        inertia, elbow_index = Elbow(data)
        k = elbow_index
        print('optimal number of clusters: ', k)
        KMeans_clustering(data, k)
        
    elif args.optimal_cluster_selection == 'silhouette':
        silhouette_scores, optimal_k = Silhouette(data)
        k = optimal_k 
        print('optimal number of clusters: ', k)
        KMeans_clustering(data, k)
        
    elif args.optimal_cluster_selection == 'davies_bouldin':
        davies_bouldin_scores, optimal_k = davies_bouldin(data)
        k = optimal_k 
        print('optimal number of clusters: ', k)
        KMeans_clustering(data, k)    
        
    elif args.optimal_cluster_selection == 'gap_statistic':
        gaps, resultsdf, optimal_k = gap_statistic(data)
        k = optimal_k 
        print('optimal number of clusters: ', k)
        KMeans_clustering(data, k) 

elif args.cluster_algorithm == 'gaussian_mixture':
    
    if args.optimal_cluster_selection == 'silhouette':
        silhouette_scores, optimal_k = Silhouette(data)
        k = optimal_k 
        print('optimal number of clusters: ', k)
        GaussianMixture_clustering (data, k)
        
    elif args.optimal_cluster_selection == 'davies_bouldin':
        davies_bouldin_scores, optimal_k = davies_bouldin(data)
        k = optimal_k 
        print('optimal number of clusters: ', k)
        GaussianMixture_clustering (data, k) 

elif args.cluster_algorithm == 'agglomerative_clustering':
    
    if args.optimal_cluster_selection == 'silhouette':
        silhouette_scores, optimal_k = Silhouette(data)
        k = optimal_k 
        print('optimal number of clusters: ', k)
        Agglomerative_clustering(data, k)
    
    elif args.optimal_cluster_selection == 'davies_bouldin':
        davies_bouldin_scores, optimal_k = davies_bouldin(data)
        k = optimal_k 
        print('optimal number of clusters: ', k)
        Agglomerative_clustering(data, k) 