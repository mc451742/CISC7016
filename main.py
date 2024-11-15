"""
CISC7016 Advanced Topics in Computer Science
Author: Yumu Xie
"""

# -*- coding: utf-8 -*-

from cnn_classifier import main_cnn
from mlp_classifier import main_mlp
from transfer_learning_classifier import main_tl
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    cnn_training_loss_list, cnn_testing_loss_list = main_cnn()
    print("=" * 40)
    mlp_training_loss_list, mlp_testing_loss_list = main_mlp()
    print("=" * 40)
    tl_training_loss_list, tl_testing_loss_list = main_tl()

    # print("*" * 40)

    # load epoch round from 1 to 36 (from 0 to 35)
    # x = []
    # for i in range(36):
    #     x.append(i + 1)
    x = list(range(1, 37))

    # train loss
    y_training_cnn = cnn_training_loss_list # 36
    # y_training_mlp = mlp_training_loss_list # 18
    y_training_tl = tl_training_loss_list # 36
    y_training_mlp = np.pad(mlp_training_loss_list, (0, 36 - len(mlp_training_loss_list)), constant_values=np.nan) # fill in remaining 18 elements with N/A
    # create a new figure
    plt.figure()
    # add lines
    plt.plot(x, y_training_cnn, marker='o', linestyle='-', color='b', label='CNN + MLP')
    plt.plot(x, y_training_mlp, marker='s', linestyle='--', color='r', label='MLP')
    plt.plot(x, y_training_tl, marker='^', linestyle='-.', color='g', label='Transfer Learning')
    # add labels and title
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Train Loss')
    # add grid and legend
    plt.grid(True)
    # put the legends in the best place automatically
    plt.legend(loc='best')
    # save the plot as an image file
    plt.savefig('train_loss.png') # save as PNG
    plt.savefig('train_loss.pdf') # to save as PDF, use this instead
    # show the plot
    # plt.show()

    # test loss
    y_testing_cnn = cnn_testing_loss_list
    # y_testing_mlp = mlp_testing_loss_list
    y_testing_tl = tl_testing_loss_list
    y_testing_mlp = np.pad(mlp_testing_loss_list, (0, 36 - len(mlp_testing_loss_list)), constant_values=np.nan) # fill in remaining 18 elements with N/A
    # create a new figure
    plt.figure()
    # add lines
    plt.plot(x, y_testing_cnn, marker='o', linestyle='-', color='b', label='CNN + MLP')
    plt.plot(x, y_testing_mlp, marker='s', linestyle='--', color='r', label='MLP')
    plt.plot(x, y_testing_tl, marker='^', linestyle='-.', color='g', label='Transfer Learning')
    # add labels and title
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Test Loss')
    # add grid and legend
    plt.grid(True)
    # put the legends in the best place automatically
    plt.legend(loc='best')
    # save the plot as an image file
    plt.savefig('test_loss.png') # save as PNG
    plt.savefig('test_loss.pdf') # to save as PDF, use this instead
    # show the plot
    # plt.show()

    # print("*" * 40)
