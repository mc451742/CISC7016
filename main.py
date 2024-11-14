"""
CISC7016 Advanced Topics in Computer Science
Author: Yumu Xie
"""

# -*- coding: utf-8 -*-

from cnn_classifier import main_cnn
from mlp_classifier import main_mlp
import matplotlib.pyplot as plt

if __name__ == "__main__":
    cnn_training_loss_list, cnn_testing_loss_list = main_cnn()
    mlp_training_loss_list, mlp_testing_loss_list = main_mlp()

    # print("*" * 40)

    # load epoch round from 1 to 36 (from 0 to 35)
    x = []
    for i in range(36):
        x.append(i + 1)

    # CNN + MLP
    y_training_cnn = cnn_training_loss_list
    y_training_mlp = mlp_training_loss_list
    # add lines
    plt.plot(x, y_training_cnn, marker='o', linestyle='-', color='b', label='CNN + MLP')
    plt.plot(x, y_training_mlp, marker='s', linestyle='--', color='r', label='MLP')
    # add labels and title
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training Loss of CNN and MLP')
    # add grid and legend
    plt.grid(True)
    plt.legend()
    # save the plot as an image file
    plt.savefig('train_loss.png') # save as PNG
    plt.savefig('train_loss.pdf') # to save as PDF, use this instead
    # show the plot
    # plt.show()

    # MLP
    y_testing_cnn = cnn_testing_loss_list
    y_testing_mlp = mlp_testing_loss_list
    # add lines
    plt.plot(x, y_testing_cnn, marker='o', linestyle='-', color='b', label='CNN + MLP')
    plt.plot(x, y_testing_mlp, marker='s', linestyle='--', color='r', label='MLP')
    # add labels and title
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Validation Loss (Testing Loss) of CNN and MLP')
    # add grid and legend
    plt.grid(True)
    plt.legend()
    # save the plot as an image file
    plt.savefig('test_loss.png') # save as PNG
    plt.savefig('test_loss.pdf') # to save as PDF, use this instead
    # show the plot
    # plt.show()

    # print("*" * 40)
