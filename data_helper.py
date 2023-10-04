import numpy as np
import gzip
import pickle
from sklearn.decomposition import PCA
from itertools import combinations
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

'''
    Remove the classes unrelated to the current task
'''
def remove_unrela_data(xs, ys, classes):
    keep = []
    for y in ys:
        if y in classes:
            keep.append(True)
        else:
            keep.append(False)
    xs = xs[keep]
    ys = ys[keep]
    return xs, ys


'''
    Pre-processing the dataset
'''
def preprocess_data(xs, ys, d):
    xs = xs / 255.0
    pca = PCA(n_components=d)
    pca.fit(xs)
    pca_data = pca.transform(xs)

    pca_descaler = [[] for _ in range(d)]
    # Data Normalization 
    for i in range(d):
        if pca_data[:,i].min() < 0:
            pca_descaler[i].append(pca_data[:,i].min())
            pca_data[:,i] += np.abs(pca_data[:,i].min())
        else:
            pca_descaler[i].append(pca_data[:,i].min())
            pca_data[:,i] -= pca_data[:,i].min()
        pca_descaler[i].append(pca_data[:,i].max())
        pca_data[:,i] /= pca_data[:,i].max()

    # Remove outliers
    valid_ind = [True for _ in range(len(pca_data))]
    for col in range(pca_data.shape[1]):
        t_data_mean = pca_data[:,col].mean()
        t_data_std = pca_data[:,col].std()
        valid_upper_bound = pca_data[:,col] < t_data_mean+t_data_std*2
        valid_lower_bound = pca_data[:,col] > t_data_mean-t_data_std*2
        valid = np.logical_and(valid_upper_bound,valid_lower_bound)
        valid_ind = np.logical_and(valid_ind, valid)

    pca_data = pca_data[valid_ind]
    pca_data = 2 * np.pi * pca_data
    ys = ys[valid_ind]

    return pca_data, ys


def load_pca_data(imgsize, classes):
    f = gzip.open('./MORE/dataset/mnist/mnist.pkl.gz', 'rb')
    trainset, testset = pickle.load(f, encoding="bytes")

    x_train, y_train = trainset
    x_test, y_test = testset

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # remove unrelated data
    x_train, y_train = remove_unrela_data(x_train, y_train, classes)
    x_test, y_test = remove_unrela_data(x_test, y_test, classes)

    # Data preprocessing
    x_train, y_train = preprocess_data(x_train, y_train, imgsize)
    x_test, y_test = preprocess_data(x_test, y_test, imgsize)

    print("Number of training examples:", len(x_train))
    print("Number of test examples:", len(x_test))

    return x_train, y_train, x_test, y_test


def generate_data_pairs(xs, ys, num):
    ind_list = [i for i in range(len(xs))]
    comb = combinations(ind_list, 2)
    comb = shuffle(list(comb))

    if len(comb) < num:
        num = len(comb)
    
    x_pairs = []
    y_pairs = []
    for index_tuple in comb[:num]:
        tmp = [xs[t] for t in index_tuple]
        x_pairs.append(tmp)
        tmp = [ys[t] for t in index_tuple]
        y_pairs.append(tmp)

    return shuffle(np.array(x_pairs), np.array(y_pairs))

def mse(img1, img2):
    error = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    error /= float(img1.shape[0])
    return  round(error, 3)

def calc_class_rela(x, y, label_list, dataset, task, path):
    num_data = 100
    if dataset == 'mnist':
        class_num = 10
    elif dataset == 'bars' or dataset == 'iris':
        class_num = 3
        num_data = 35

    image_list = [[] for _ in range(class_num)]

    for label in label_list:
        keep = y == label
        x_keep = x[keep]
        x_keep = x_keep[:num_data]
        x_keep = np.mean(x_keep, 0)
        image_list[label] = x_keep

    mse_arr = np.zeros((class_num, class_num))

    for i in label_list:
        for j in label_list:
            mse_arr[i, j] = mse(image_list[i], image_list[j])

    # Normolization
    max_s = np.max(mse_arr)
    min_s = np.min(mse_arr)
    new_mse = (mse_arr - min_s) / max_s - min_s
    mse_arr = np.round(new_mse, 3)

    for i in range(class_num):
        for j in range(class_num):
            if i == j:
                mse_arr[i, j] = -1.0

    fig, ax = plt.subplots(figsize=(13,7))
    title = "MSE"
    ax.xaxis.set_ticks(label_list)
    ax.yaxis.set_ticks(label_list)
    # ax.set_xticklabels(['0', '1', '2'], fontsize=20)
    # ax.set_yticklabels(['0', '1', '2'], fontsize=20)

    plt.title(title,fontsize=20)
    ttl = ax.title
    ttl.set_position([0.5,1.05])
    df_cm = pd.DataFrame(mse_arr, index = [i for i in range(class_num)],columns = [i for i in range(class_num)])

    tmp = sns.heatmap(df_cm,annot=mse_arr,fmt="",cmap='RdYlGn',linewidths=0.30,ax=ax, annot_kws={"size": 20})
    
    # sns.set(font_scale=5)
    plt.savefig(path + 'mse_heatmap.jpg')
    plt.close()

    return mse_arr



def sample_data(x, y, classes, sample_num):
    new_x = []
    new_y = []

    for i in range(len(classes)):
        keep = y == classes[i]
        points = x[keep][:sample_num]
        new_x.append(points)
        new_y.append(y[keep][:sample_num])

    new_x = np.concatenate(new_x, axis=0)
    new_y = np.concatenate(new_y, axis=0)

    return shuffle(new_x, new_y)
