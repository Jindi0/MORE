import os
import argparse
import numpy as np
import json
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from IPython.display import clear_output
from qiskit.algorithms.optimizers import COBYLA

from data_helper import load_pca_data, calc_class_rela, sample_data, generate_data_pairs
from model import build_qcnn
from myNeuralNetworkClassifier_2 import myNeuralNetworkClassifier
from util import plot_vectorsinBS, resize_vectors, init_log


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='clf_03', help='classification task name')
    parser.add_argument('--clt_task', type=str, default='clt_03_5', help='clustering task name')
    parser.add_argument('--resume', type=bool, default=False, help='task name')

    parser.add_argument('--samples', type=int, default=30, help='the number of training samples')
    parser.add_argument('--COBYLAiter', type=int, default=4,help="the number of epochs")
    parser.add_argument('--adaptor_thd', type=float, default=1, help='the range of loss adjuster')
    parser.add_argument('--term_weight', type=float, default=0.2, help='weight of loss adjuster')

    parser.add_argument('--inputsize', type=int, default=8, help='the input size')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")

    args = parser.parse_args()
    return args

args = args_parser()

savepath = './MORE/save_more/'
if not os.path.exists(savepath):
    os.mkdir(savepath)
clt_path = './MORE/save_more/{}/'.format(args.clt_task)
task_path = './MORE/save_more/{}/'.format(args.task)
weight_path = task_path + 'clt_checkpoints/'


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)

    if args.resume:
        plt.savefig(task_path + 'clt_Loss_resume.jpg'.format(args.task))
    else:
        plt.savefig(task_path + 'clt_Loss.jpg'.format(args.task))
    # sheet.write(int(len(objective_func_vals)+1), 2, obj_func_eval)

    # if len(objective_func_vals)%100 == 0:
    np.save(weight_path + 'ckp_{}.npy'.format(len(objective_func_vals)+ finished_iter), weights)
        


if __name__ == "__main__":
    if not os.path.exists(task_path):
        os.mkdir(task_path)
        os.mkdir(weight_path)

    f, sheet = init_log()

    qubit_num = args.inputsize
    classes = args.task.split('_')[1]
    classes = [int(s) for s in classes]

    log_dic = {}
    log_dic['task'] = args.task
    log_dic['clt_task'] = args.clt_task
    log_dic['classes'] = classes
    log_dic['train_smp'] = args.samples
    log_dic['COBYLAiter'] = args.COBYLAiter
    log_dic['inputsize'] = args.inputsize
    log_dic['dataset'] = args.dataset

    print('Data loading ... ')

    # --- Prepare classical datasets
    x_train, y_train, x_test, y_test = load_pca_data(args.inputsize, classes)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    x_train = x_train[:args.samples]
    y_train = y_train[:args.samples]

    x_test = x_test[:500]
    y_test = y_test[:500]

    log_dic['test_num'] = len(y_test)
    
    # Build QCNN ansatz

    print('Load clustering data ... ')
    mse_arr = np.load(clt_path + 'mse_rela.npy')

    list_of_files = glob.glob(clt_path + 'clt_checkpoints/*') # * means all if need specific format then *.csv
    clt_model = max(list_of_files, key=os.path.getctime)
    initial_point = np.load(clt_model)

    finished_iter = 0

    if args.resume:
        list_of_files = glob.glob(task_path + 'clt_checkpoints/*')
        clt_model = max(list_of_files, key=os.path.getctime)
        initial_point = np.load(clt_model)
        finished_iter = len(list_of_files)


    with open(clt_path + 'clt_centers.json', 'r') as f:
        centers = json.load(f)

    center_dist_arr = None 
    if args.adaptor_thd > 0:
        center_dist_arr = np.load(clt_path + 'clt_centers_distance.npy')
    
    # ======= Record raw distribution =============================================
    if not args.resume:
        qnn = build_qcnn(qubit_num)
        
        classifier = myNeuralNetworkClassifier(
            qnn,
            optimizer=COBYLA(maxiter=0),  # Set max iterations here
            warm_start=True,
            scale=mse_arr,
            initial_point=initial_point,
            cluster=False,
            centers=centers,
            adaptor_thd=args.adaptor_thd,
            center_dist_arr = center_dist_arr,
            term_weight = args.term_weight
        )

        # get the distribution before training
        classifier.fit(x_train[:5], y_train[:5])
        vectors = classifier.predict_vector(x_train[:20])
        plot_vectorsinBS(vectors, y_train[:20], img_name =task_path + 'class_train_raw.jpg')
        np.save(task_path + 'class_train_raw_vec.npy', vectors)
        np.save(task_path + 'train_x.npy', x_train[:20])
        np.save(task_path + 'train_y.npy',  y_train[:20])
    
    # ========== Classification ==========================================
    print('Training ... ')
    objective_func_vals = []
    plt.rcParams["figure.figsize"] = (12, 6)

    qnn = build_qcnn(qubit_num)

    classifier = myNeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=args.COBYLAiter),  # Set max iterations here
        warm_start=True,
        callback=callback_graph,
        scale=mse_arr,
        initial_point=initial_point,
        cluster=False,
        centers=centers,
        adaptor_thd=args.adaptor_thd,
        center_dist_arr = center_dist_arr,
        term_weight = args.term_weight
    )

    classifier.fit(x_train, y_train)
    # np.save(task_path + 'clt_model.npy'.format(args.task), classifier.weights)

    vectors = classifier.predict_vector(x_train[:20])
    np.save(task_path + 'class_train_end_vec.npy', vectors)
   
    plot_vectorsinBS(vectors, y_train[:20], img_name=task_path+'class_train_done.jpg')

    # classifier.evaluation(x_test, y_test, centers)

    vectors = classifier.predict_vector(x_test[:20])
    plot_vectorsinBS(vectors, y_test[:20], img_name=task_path+'class_test.jpg')

    val_acc = classifier.evaluation(x_val, y_val, centers)
    log_dic['val_acc'] = val_acc

    test_acc = classifier.evaluation(x_test, y_test, centers)
    log_dic['test_acc'] = test_acc

    with open(task_path + 'log.json', "w") as outfile:
        json.dump(log_dic, outfile)


    # ==================== Testing ==================================

    acc_f, sheet = init_log()
    qnn = build_qcnn(qubit_num)
    row = 1
    for i in np.arange(1, args.COBYLAiter+1):
        ckp_path = task_path + 'clt_checkpoints/ckp_{}.npy'.format(i)
    
        initial_point = np.load(ckp_path)

        classifier = myNeuralNetworkClassifier(
            qnn,
            optimizer=COBYLA(maxiter=0),  # Set max iterations here
            warm_start=True,
            # callback=callback_graph,
            scale=mse_arr,
            initial_point=initial_point,
            cluster=False,
            centers=centers,
            adaptor_thd=args.adaptor_thd,
            center_dist_arr = center_dist_arr,
            term_weight = args.term_weight
        )

        classifier.fit(x_train[:1], y_train[:1])

        # val_acc = classifier.evaluation(x_val, y_val, centers)
        # sheet.write(row, 5, val_acc)

        test_acc = classifier.evaluation(x_test, y_test, centers)
        sheet.write(row, 8, test_acc)
        print(test_acc)

        row += 1
        acc_f.save(task_path + '/ckp_accs_lesstest.xls')


    