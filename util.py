import xlwt
import numpy as np
import matplotlib.pyplot as plt
from myBloch import myBloch
import os
import json
import seaborn as sns
import pandas as pd


def init_log():
    f = xlwt.Workbook()

    sheet1 = f.add_sheet('results',cell_overwrite_ok=True)
    row = ['settings']
    row += '-'
    row.append(['train_acc'])
    row.append(['train_loss'])
    row += '-'
    row.append(['val_acc'])
    row.append(['val_loss'])
    row += '-'
    row.append(['test_acc'])
    row.append(['test_loss'])

    style = xlwt.XFStyle()
    style.alignment.wrap = 1

    for i in range(len(row)):
        sheet1.write(0, i, row[i])

    return f, sheet1


def resize_vectors(vectors):
    scales = [np.linalg.norm(v) for v in vectors]
    vectors = [vectors[i]/scales[i] for i in range(len(vectors))]
    return vectors


'''
    Plot state vectors in Bloch Sphere
'''
def plot_vectorsinBS(vectors, labels, img_name, points=None):
    plt.close()
    vectors = resize_vectors(vectors)

    fig = plt.figure(figsize = (5, 5))
    ax = plt.axes(projection='3d')

    B = myBloch(color_list=labels, axes=ax)
    # B.set_label_convention('xyz')
    # B.figsize = [4, 4]
    B.add_vectors(vectors)
    if points is not None:
        for i in range(len(points)):
            B.add_points(points[i])

    B.render()

    fig.savefig(img_name)
    plt.close()

def cos_dist(a, b):
    a = np.asarray(a) / np.linalg.norm(a, axis=0, keepdims=True)
    b = np.asarray(b) / np.linalg.norm(b, axis=0, keepdims=True)
    return np.round(1 - np.dot(a, b.T), 3)


def draw_heatmap(arr, title, taskpath):
    fig, ax = plt.subplots(figsize=(13,7))
    plt.title(title,fontsize=25)
    ttl = ax.title
    ttl.set_position([0.5,1.05])
    df_cm = pd.DataFrame(arr, index = [i for i in range(len(arr))],columns = [i for i in range(len(arr))])

    tmp = sns.heatmap(df_cm,annot=arr,fmt="",cmap='Blues',linewidths=0.30,ax=ax, annot_kws={"size": 18})
    
    sns.set(font_scale=5)
    plt.savefig(taskpath + '/clt_centers_distance.jpg')
    np.save(taskpath + '/clt_centers_distance.npy', arr)


def draw_dist_centers(taskpath):
    center_file = os.path.join(taskpath, 'clt_centers.json')
    with open(center_file, 'r') as f:
        dic = json.load(f)
    classes = 10
    center_list = np.zeros((classes, 3))

    keys = [int(i) for i in dic.keys()]

    for key, value in dic.items():
        center_list[int(key)] = value

    dist_array = np.zeros((classes, classes))
    for i in keys:
        for j in keys:
            dist_array[i, j] = cos_dist(center_list[i], center_list[j])

    draw_heatmap(dist_array, 'Distance between quantum labels', taskpath)