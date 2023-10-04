import os
import argparse
import numpy as np
import json
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import clear_output
from qiskit.algorithms.optimizers import COBYLA

from data_helper import load_pca_data, calc_class_rela, sample_data, generate_data_pairs
from model import build_qcnn
from myNeuralNetworkClassifier_1 import myNeuralNetworkClassifier
from util import plot_vectorsinBS, resize_vectors, draw_dist_centers


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='clt_03_5', help='task name: [step]_[classes]_[samples]')
    
    parser.add_argument('--clt_samples', type=int, default=5, help='the number of training samples of each class')
    parser.add_argument('--COBYLAiter', type=int, default=2,help="the number of epochs")
    parser.add_argument('--pairs_num', type=int, default=1000, help='the threshold of pairs')

    parser.add_argument('--inputsize', type=int, default=8, help='the input size')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    
    args = parser.parse_args()
    return args

args = args_parser()

savepath = './MORE/save_more/'
if not os.path.exists(savepath):
    os.mkdir(savepath)
task_path = './MORE/save_more/{}/'.format(args.task)
weight_path = task_path + 'clt_checkpoints/'


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.savefig(task_path + 'clt_Loss.jpg'.format(args.task))

    # if len(objective_func_vals)%100 == 0:
    np.save(weight_path + 'ckp_{}.npy'.format(len(objective_func_vals)), weights)


def find_center(vectors, labels):
    classes = args.task.split('_')[1]
    classes = [int(s) for s in classes]

    centers = {}
    centers_ = []

    vectors = np.array(resize_vectors(vectors))

    for i in range(len(classes)):
        cluster_id = labels == classes[i]
        points = vectors[cluster_id]
        center = np.median(points, axis=0)
        center = resize_vectors([center])[0]
        centers[str(classes[i])] = center.tolist()
        centers_.append(center)

    with open(task_path + 'clt_centers.json'.format(args.task), 'w') as f:
        json.dump(centers, f)
    
    return centers_, centers


if __name__ == "__main__":
    if not os.path.exists(task_path):
        os.mkdir(task_path)
        os.mkdir(weight_path)

    qubit_num = args.inputsize
    classes = args.task.split('_')[1]
    classes = [int(s) for s in classes]

    log_dic = {}
    log_dic['task'] = args.task
    log_dic['classes'] = classes
    log_dic['smp/cla'] = args.clt_samples
    log_dic['COBYLAiter'] = args.COBYLAiter
    log_dic['inputsize'] = args.inputsize
    log_dic['dataset'] = args.dataset

    print('Data loading ... ')

    # --- Prepare classical datasets
    x_train, y_train, x_test, y_test = load_pca_data(args.inputsize, classes)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

    log_dic['test_num'] = len(y_test)
    log_dic['val_num'] = len(y_val)
    
    # --- calculate relationships between classes
    mse_arr = calc_class_rela(x_train, y_train, classes, args.dataset, args.task, task_path)
    np.save( task_path + 'mse_rela.npy', np.array(mse_arr))

    x_train_clt, y_train_clt = sample_data(x_train, y_train, classes, args.clt_samples)
    x_train_clt_pair, y_train_clt_pair = generate_data_pairs(x_train_clt, y_train_clt, args.pairs_num)

    log_dic['trainpairs_num'] = len(y_train_clt_pair)


    # Build QCNN ansatz
    
    # ======= Record raw distribution =============================================
    plt.close()
    qnn = build_qcnn(qubit_num)
    classifier = myNeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=0),  # Set max iterations here
        scale=mse_arr
    )

    # get the distribution before training
    classifier.fit(x_train_clt_pair[:1], y_train_clt_pair[:1])
    np.save(weight_path + 'ckp_0.npy', classifier.weights)
    init_point = classifier.weights

    vectors = classifier.predict_vector(x_train_clt)
    plot_vectorsinBS(vectors, y_train_clt, img_name =task_path + 'bs_train_raw.jpg')
    np.save(task_path + 'clt_train_raw_vec.npy', vectors)
    np.save(task_path + 'train_x.npy', x_train_clt)
    np.save(task_path + 'train_y.npy', y_train_clt)

    # ========== Clustering ==========================================

    print('Training ... ')
    objective_func_vals = []
    plt.close()
    plt.rcParams["figure.figsize"] = (12, 6)

    qnn = build_qcnn(qubit_num)

    classifier = myNeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=args.COBYLAiter),  # Set max iterations here
        callback=callback_graph,
        scale=mse_arr,
        initial_point=init_point,
        warm_start=True
    )
    classifier.fit(x_train_clt_pair, y_train_clt_pair)
    # np.save(task_path + 'clt_model.npy'.format(args.task), classifier.weights)

    vectors = classifier.predict_vector(x_train_clt)
    np.save(task_path + 'clt_train_end_vec.npy', vectors)
    centers, centers_dic = find_center(vectors, y_train_clt)
    plot_vectorsinBS(vectors, y_train_clt, points=centers, img_name=task_path+'bs_train_done.jpg')

    vectors = classifier.predict_vector(x_test[:20])
    plot_vectorsinBS(vectors, y_test[:20], img_name=task_path+'bs_test.jpg')

    val_acc = classifier.evaluation(x_val, y_val, centers_dic)
    log_dic['val_acc'] = val_acc

    test_acc = classifier.evaluation(x_test, y_test, centers_dic)
    log_dic['test_acc'] = test_acc

    with open(task_path + 'log.json', "w") as outfile:
        json.dump(log_dic, outfile)

    draw_dist_centers(task_path)




    