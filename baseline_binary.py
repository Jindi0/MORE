'''
    Baseline classifier
    Z-measurement 
'''
import os
import argparse
import numpy as np
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import clear_output
from qiskit.algorithms.optimizers import COBYLA

from data_helper import load_pca_data
from model import build_qcnn_baseline_8
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from util import init_log


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='basez_01_8q', help='task name:[modelType]_[classes]_[modelSize]q')

    parser.add_argument('--samples', type=int, default=1000, help='the number of training samples')
    parser.add_argument('--COBYLAiter', type=int, default=20,help="the number of epochs")
    parser.add_argument('--train', type=bool, default=True, help='if perform training procedure')
    parser.add_argument('--test', type=bool, default=True, help='if perform test procedure')

    parser.add_argument('--inputsize', type=int, default=8, help='the input size')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
   
    args = parser.parse_args()
    return args

args = args_parser()

savepath = './MORE/save_baseline_8q_bin/'
if not os.path.exists(savepath):
    os.mkdir(savepath)
task_path = './MORE/save_baseline_8q_bin/{}/'.format(args.task)
weight_path = task_path + 'clt_checkpoints/'


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.savefig(task_path + 'clt_Loss.jpg'.format(args.task))

    # if len(objective_func_vals)%50 == 0:
    np.save(weight_path + 'ckp_{}.npy'.format(len(objective_func_vals)), weights)


        





def bin_label(ys, classes):  
    ys = np.int8(ys)
    
    ind_0 = ys == classes[0]
    ys[ind_0] = -1

    ind_1 = ys == classes[1]
    ys[ind_1] = 1

    return ys



if __name__ == "__main__":
    path = './MORE/'
    
    if not os.path.exists(task_path):
        os.mkdir(task_path)
        os.mkdir(weight_path)

    f, sheet = init_log()

    qubit_num = args.inputsize
    classes = args.task.split('_')[1]
    classes = [int(s) for s in classes]

    log_dic = {}
    log_dic['task'] = args.task
    log_dic['classes'] = classes
    log_dic['train_smp'] = args.samples
    log_dic['COBYLAiter'] = args.COBYLAiter
    log_dic['inputsize'] = args.inputsize
    log_dic['dataset'] = args.dataset

    print('Data loading ... ')

    # --- Prepare classical datasets
    x_train, y_train, x_test, y_test = load_pca_data(args.inputsize, classes)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

    x_train = x_train[:args.samples]
    y_train = y_train[:args.samples]
    y_train = bin_label(y_train, classes)

    y_test = bin_label(y_test, classes)
    y_val = bin_label(y_val, classes)

    log_dic['test_num'] = len(y_test)
    log_dic['val_num'] = len(y_val)


    # ======= Training =============================================

    plt.close()
    print('Training ... ')

    qnn = build_qcnn_baseline_8(qubit_num)

    if args.train:

        objective_func_vals = []
        plt.rcParams["figure.figsize"] = (12, 6)

        classifier = NeuralNetworkClassifier(
            qnn,
            optimizer=COBYLA(maxiter=args.COBYLAiter),  # Set max iterations here
            warm_start=True,
            callback=callback_graph
        )

        classifier.fit(x_train, y_train)

        print(f"Accuracy from the train data : {np.round(100 * classifier.score(x_train, y_train), 2)}%")

        test_acc = np.round(100 * classifier.score(x_test, y_test), 2)
        print(f"Accuracy from the test data : {test_acc}%")
        log_dic['test_acc'] = test_acc

        val_acc = np.round(100 * classifier.score(x_val, y_val), 2)
        print(f"Accuracy from the val data : {val_acc}%")
        log_dic['val_acc'] = val_acc

        objective_func_vals = np.array(objective_func_vals)
        np.save(task_path + 'training_loss.npy', objective_func_vals)

        with open(task_path + 'log.json', "w") as outfile:
            json.dump(log_dic, outfile)


    # ========== Testing ==========================================

    if args.test:

        print('Testing ... ')
        acc_f, sheet = init_log()
        
        for r in range(len(objective_func_vals)):
            sheet.write(r+1, 3, objective_func_vals[r])
        acc_f.save(task_path + '/ckp_accs.xls')

        row = 1
        for i in np.arange(args.COBYLAiter):
            ckp_path = task_path + 'clt_checkpoints/ckp_{}.npy'.format(i+1)
        
            initial_point = np.load(ckp_path)

            classifier = NeuralNetworkClassifier(
                qnn,
                optimizer=COBYLA(maxiter=0),  # Set max iterations here
                warm_start=True,
                # callback=callback_graph,
                initial_point=initial_point
            )

            classifier.fit(x_train[:5], y_train[:5])

            test_acc = np.round(100 * classifier.score(x_test, y_test), 2)
            sheet.write(row, 8, test_acc)

            # val_acc = np.round(100 * classifier.score(x_val, y_val), 2)
            # sheet.write(row, 5, val_acc)
            row += 1

            acc_f.save(task_path + '/ckp_accs.xls')
            # print(' --- {} ---'.format(row))

    
    

    

    


    