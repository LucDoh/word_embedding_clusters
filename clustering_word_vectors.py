import io
import sys
import numpy as np
import json
from sklearn.cluster import DBSCAN, KMeans, OPTICS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# Cluster these 300D vectors into 10 groups?
# Do clusters seem to contain similar words?

# Word vectors from fastText (2M words, 300 dimensions)
# Two approaches for training word representations:
# 1) skipgram: predict target word based on a random nearby word
# 2) cbow (continuous BoW): predict using a bag of words in fixed window around target
# In practice, they've seen skipgram with subword info to work best


def load_vectors(fname, limit=10):
    # Loading fastText vectors (fasttext.cc/docs/en/english-vectors.html)
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # (Num words, Dimensionality)
    n, d = map(int, fin.readline().split())
    data = {}
    for i, line in enumerate(fin):
        #if i % 1000 == 0: print(i)
        if i > limit:
            break
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))

    return data

def save_to_json(data, fname):
    with open(fname, 'w') as f:
        json.dump(data, f)


def load_subset(fname):
    print('Loading...')
    with open(fname) as f:
        data = json.load(f)
    return data


def distance(vec1, vec2):
    diff = np.array(vec1) - np.array(vec2)
    return np.sqrt(np.sum([d**2 for d in diff]))


def compare(data, word1, word2):
    dist = distance(data[word1], data[word2])
    print(f'Distance from {word1} to {word2}: {dist:.2f}')


def cluster(x, method='kmeans', **kwargs):
    # Density based clustering
    if method == 'OPTICS':
        # Improved DBSCAN (outward radiating from highly-dense cores, faster, scans over eps)
        clustering = OPTICS(min_samples=kwargs['min_samples']).fit(X)
    elif method == 'DBSCAN':
         # Worst-case O(n^2), eps = neighborhood radius
        clustering = DBSCAN(eps=kwargs['eps']).fit(X)  # 7).fit(X)
    else:
        # Partition-based clustering
        clustering = KMeans(
            n_clusters=kwargs['n_clusters'], random_state=0).fit(X)  # 10

    # Number of elements in model's clusters:
    print(f"Clustering with {method}")
    print(np.unique(clustering.labels_, return_counts=True))
    return clustering.labels_.tolist()


def plot_word_embeddings(X, words, reduction_method='TSNE'):
    # Reduce dimensionality to 2D to visualize
    # PCA favors preserving global structure, is a linear
    # dimensionality reduction technique.
    # TSNE  is non-linear and tries to preserve local neighborhoods.
    # It seems to group similar words tighter and overall "looks better".
    if reduction_method == 'TSNE':
        x = TSNE(n_components=2).fit_transform(X)
    else:
        x = PCA(n_components=2).fit_transform(X)

    fig, ax = plt.subplots()
    ax.scatter([x_i[0] for x_i in x], [x_i[1] for x_i in x])
    for i, txt in enumerate(words):
        ax.annotate(txt, (x[i][0], x[i][1]))
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        # Load word vectors then save to a dict, limiting to
        # 1/4 for memory
        data = load_vectors('data/crawl-300d-2M.vec', limit=500000)
        save_to_json(data, 'data/wordvecs.json')
        sys.exit(0)
    # After creating wordvecs.json, with {word: embedding, ...}
    data = load_subset('data/wordvecs.json')
    vals = list(data.values())

    print(f'{len(data)} words loaded')
    print(f'Words: {str(list(data.keys())[:5])[:-1]}, ... {str(list(data.keys())[-5:])[1:]}')
    # Compute L2 distance between words
    compare(data, 'refer', 'referring')
    compare(data, 'refering', 'referring')
    compare(data, 'refer', 'Karo')
    compare(data, 'refer', '.')
    compare(data, 'man', 'woman')

    n = 1000
    print(f"Grabbing {n} words and embeddings")
    words = list(data.keys())[:n]
    X = np.array(vals[:n])

    print("Clustering...")
    cluster_labels = cluster(X, n_clusters=10)
    cluster_labels = cluster(X, method='DBSCAN', eps=7)
    cluster_labels = cluster(X, method='OPTICS', min_samples=int(n/200))


    # Save clusters
    data_clusters = list(zip(words, cluster_labels))
    save_to_json(data_clusters, 'data/1000_word_clusters_optics.json')

    for k in np.unique(cluster_labels):
        if k == -1:
            continue
        print(f"Cluster {k}: {[words[i] for i in range(len(words)) if cluster_labels[i] == k]}")
    

    # Plotting word_embeddings using dimensionality reduction
    print("Plotting...")
    plot_word_embeddings(X, words, 'TSNE')

    # AND multiple clusters, i.e. vote from multiple models...
