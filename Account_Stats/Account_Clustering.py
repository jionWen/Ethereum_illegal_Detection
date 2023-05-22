import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import io, math, ui

NORMAL_FLAG = 0
ILLICIT_FLAG = 1

def read_file(filename):
    import pandas as pd

    csv_file = filename
    df = pd.read_csv(csv_file)

    df = remove_fields(df)

    df.fillna(0, inplace=True)
    return df

#  pandas 返回dataframe 文件
def read_file_w_flag(filename, flag):
    csv_file = filename
    df = pd.read_csv(csv_file)

    if flag == NORMAL_FLAG:
        normal_accounts = df['FLAG'] == NORMAL_FLAG
        df = df[normal_accounts]

    elif flag == ILLICIT_FLAG:
        illicit_accounts = df['FLAG'] == ILLICIT_FLAG
        df = df[illicit_accounts]
    # Y = df['FLAG']
    # X = df.loc[:, df.columns != 'FLAG']
    df = remove_fields(df)
    df.fillna(0, inplace=True)
    return df


# 移除非必要属性
def remove_fields(df):
    df.pop('Index')
    df.pop('Address')
    df.pop('ERC20_most_sent_token_type')
    df.pop('ERC20_most_rec_token_type')
    df.pop('ERC20_uniq_sent_token_name')
    df.pop('ERC20_uniq_rec_token_name')
    return df

def PCA_plot(df, no_of_components):
    X = df.loc[:, df.columns != 'FLAG']
    Y = df['FLAG']
    print(X.shape)
    pca = PCA(n_components=no_of_components)
    pca_result = pca.fit_transform(X)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]
    print(pca.explained_variance_ratio_)
    if no_of_components > 3:
        print(np.sum(pca.explained_variance_ratio_))


def TSNE_plot(df):
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000, n_iter_without_progress=20)
    X = df.loc[:, df.columns != 'FLAG']
    Y = df['FLAG']
    tsne_results = tsne.fit_transform(X)
    X['first_dimension'] = tsne_results[:, 0]
    X['second_dimension'] = tsne_results[:, 1]
    #X['third_dimension'] = tsne_results[:,2]

    fig1 = plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="first_dimension", y="second_dimension",
        hue=Y,
        palette=['blue','red'],
        data=X,
        legend="full",
        alpha=0.2
    )
    fig1.show()


def k_means(df, K_neighbors):
    Y = df['FLAG']
    X = df.loc[:, df.columns != 'FLAG']
    X = X.loc[:, :]
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)
    X['tsne-3d-one'] = tsne_results[:, 0]
    X['tsne-3d-two'] = tsne_results[:, 1]
    X['tsne-3d-three'] = tsne_results[:,2]
    classifier = KMeans(n_clusters=K_neighbors)
    classifier.fit(tsne_results[:,:3])
    y_kmeans = classifier.predict(tsne_results[:,:3])

    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(2,2,1, projection='3d')
    ax.scatter(X['tsne-3d-one'], X['tsne-3d-two'],X['tsne-3d-three'], c=y_kmeans, s=20, cmap='viridis')
    centers = classifier.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
    ax.view_init(30, 45)

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.scatter(X['tsne-3d-one'], X['tsne-3d-two'], X['tsne-3d-three'], c=y_kmeans, s=20, cmap='viridis')
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
    ax.view_init(30, 135)

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.scatter(X['tsne-3d-one'], X['tsne-3d-two'], X['tsne-3d-three'], c=y_kmeans, s=20, cmap='viridis')
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
    ax.view_init(30, 225)

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.scatter(X['tsne-3d-one'], X['tsne-3d-two'], X['tsne-3d-three'], c=y_kmeans, s=20, cmap='viridis')
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
    ax.view_init(30, 315)
    plt.show()

if __name__ == '__main__':
    accounts = read_file("./Account_Stats/Complete.csv")
    TSNE_plot(accounts)

    # PCA_plot(accounts, no_of_components=3)
    # PCA_plot(accounts, no_of_components=10)
    #TSNE_plot(accounts)

    # illicit_accounts = read_file_w_flag("./Account_Stats/Complete.csv", 1)
    # k_means(accounts, 5)
