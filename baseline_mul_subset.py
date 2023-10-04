import os
import argparse
import numpy as np
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import clear_output
from qiskit.algorithms.optimizers import COBYLA

from data_helper import load_pca_data
from model import build_qcnn_baseline_subset4, build_qcnn_baseline_subset8
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from util import init_log


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='subset_012_8q', help='task name:[modelType]_[classes]_[modelSize]q')
    
    parser.add_argument('--samples', type=int, default=1000, help='the number of training samples')
    parser.add_argument('--COBYLAiter', type=int, default=1000,help="the number of epochs")
    parser.add_argument('--train', type=bool, default=True, help='if perform training procedure')
    parser.add_argument('--test', type=bool, default=True, help='if perform test procedure')

    parser.add_argument('--inputsize', type=int, default=8, help='the input size')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")

    args = parser.parse_args()
    return args

args = args_parser()

savepath = './MORE/save_baseline_8q_mul_subset/'
if not os.path.exists(savepath):
    os.mkdir(savepath)
task_path = './MORE/save_baseline_8q_mul_subset/{}/'.format(args.task)
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

   


def onehot_label(ys, classes):  
    ys = np.int8(ys)
    new_ys = np.zeros((len(ys), len(classes)))
    for i in range(len(classes)):
        ind = ys == classes[i]
        new_ys[ind, i] = 1
    return new_ys



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
    y_train = onehot_label(y_train, classes)

    x_test = x_test[:500]
    y_test = y_test[:500]
    y_test = onehot_label(y_test, classes)
    y_val = onehot_label(y_val, classes)

    log_dic['test_num'] = len(y_test)
    log_dic['val_num'] = len(y_val)

    acc_f, sheet = init_log()


    # ======= Training =============================================
   
    plt.close()

    if len(classes) > 4:
        qnn = build_qcnn_baseline_subset8(args.inputsize, len(classes))
    else:
        qnn = build_qcnn_baseline_subset4(args.inputsize, len(classes))

    
    if args.train:
        print('-- Training ... ')

        objective_func_vals = []
        plt.rcParams["figure.figsize"] = (12, 6)

        classifier = NeuralNetworkClassifier(
            qnn,
            loss = "cross_entropy",
            optimizer=COBYLA(maxiter=args.COBYLAiter),  # Set max iterations here
            one_hot=True,
            # warm_start=True,
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

        for r in range(len(objective_func_vals)):
            sheet.write(r+1, 3, objective_func_vals[r])
        acc_f.save(task_path + '/ckp_accs.xls')
       

    # ========== Testing ==========================================
    if len(classes) > 4:
        qnn = build_qcnn_baseline_subset8(args.inputsize, len(classes))
    else:
        qnn = build_qcnn_baseline_subset4(args.inputsize, len(classes))

    if True:

        print('Testing ... ')

        row = 1
        for i in np.arange(args.COBYLAiter):
            ckp_path = task_path + 'clt_checkpoints/ckp_{}.npy'.format(i+1)
        
            initial_point = np.load(ckp_path)

            classifier = NeuralNetworkClassifier(
                qnn,
                loss = "cross_entropy",
                optimizer=COBYLA(maxiter=0),  # Set max iterations here
                one_hot=True,
                warm_start=True,
                # callback=callback_graph,
                initial_point=initial_point
            )

            classifier.fit(x_train[:1], y_train[:1])

            test_acc = np.round(100 * classifier.score(x_test, y_test), 2)
            sheet.write(row, 8, test_acc)

            # val_acc = np.round(100 * classifier.score(x_val, y_val), 2)
            # sheet.write(row, 5, val_acc)
            
            print(args.task + ' --- {} '.format(row))
            row += 1

            acc_f.save(task_path + '/ckp_accs.xls')
            # print(' --- {} ---'.format(row))

    
    

    

    


    