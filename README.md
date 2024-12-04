# CISC7016 Advanced Topics In Computer Science
CISC7016 Course Paper Implementation Code

The implementation code of this course paper is based on PyTorch framework: https://pytorch.org/

The dataset (CIFAR-10) used in this implementation: https://www.cs.toronto.edu/~kriz/cifar.html

Programming environment: WSL Linux Sub-system of Windows (Debian) & Anaconda

Python version: 3.10.15

Hardware specification: AMD Ryzen 7 6800H with Radeon Graphics CPU (16 GB) & NVIDIA GeForce RTX 3050 Ti GPU

Title: A Comparative Study of Multi-layer Perceptron, Convolutional Neural Network, and Transfer Learning Architectures for CIFAR-10 Image Classification

Abstract: This report explores image classification methods using Deep Learning (DL) on the CIFAR-10 dataset, focusing on three architectures: Multi-layer Perceptron (MLP), Convolutional Neural Network (CNN), and Transfer Learning (TL). Customized models based on these architectures are proposed, implemented, and evaluated. The report concludes with experimental results and a comparative analysis of the model's performance.

The architecture of multi-layer perceptron (MLP):

![MLP](/figure/mlp.PNG?raw=true "MLP")

![MLP_Layer](/figure/mlp_layer.PNG?raw=true "MLP_Layer")

The architecture of convolutional neural network (CNN):

![CNN](/figure/cnn.PNG?raw=true "CNN")

The architecture of transfer learning (TL):

![Transfer Learning](/figure/transfer.PNG?raw=true "Transfer Learning")

If you want to reproduce the experiments, please ensure your environment has been configured correctly and execute the following commands step by step:

```bash
chmod +x script.sh
```

```bash
./script.sh
```

Contributor:

[Yumu Xie](https://github.com/mc451742) (MC451742) mc45174@um.edu.mo

Department of Computer and Information Science, Faculty of Science and Technology, University of Macau
