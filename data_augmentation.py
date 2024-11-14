"""
CISC7016 Advanced Topics in Computer Science
Code is sourced from: https://blog.csdn.net/qq_36756866/article/details/108255715
Modified by Yumu Xie
"""

# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os.path
import copy

# Salt and Pepper
def SaltAndPepper(src,percetage=0.3):
    SP_NoiseImg=src.copy()
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(SP_NoiseNum):
        randR=np.random.randint(0,src.shape[0]-1)
        randG=np.random.randint(0,src.shape[1]-1)
        randB=np.random.randint(0,3)
        if np.random.randint(0,1)==0:
            SP_NoiseImg[randR,randG,randB]=0
        else:
            SP_NoiseImg[randR,randG,randB]=255
    return SP_NoiseImg

# Gaussian Noise
def addGaussianNoise(image,percetage=0.3):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0,h)
        temp_y = np.random.randint(0,w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg

# Darker
def darker(image,percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = int(image[xj,xi,0]*percetage)
            image_copy[xj,xi,1] = int(image[xj,xi,1]*percetage)
            image_copy[xj,xi,2] = int(image[xj,xi,2]*percetage)
    return image_copy

# Brighter
def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = np.clip(int(image[xj,xi,0]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,1] = np.clip(int(image[xj,xi,1]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,2] = np.clip(int(image[xj,xi,2]*percetage),a_max=255,a_min=0)
    return image_copy

# Rotate
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # if no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated

# Flip
def flip(image):
    flipped_image = np.fliplr(image)
    return flipped_image

# Test case
def main():
    
    img_path = './image/my_image.jpg'
    img = cv2.imread(img_path)

    img_salt = SaltAndPepper(img, 0.3)
    cv2.imwrite('./out/my_image_salt.jpg', img_salt)

    img_gauss = addGaussianNoise(img, 0.3)
    cv2.imwrite('./out/my_image_gaussian.jpg', img_gauss)

    img_heavy_salt = SaltAndPepper(img, 0.9)
    cv2.imwrite('./out/my_image_heavy_salt.jpg', img_heavy_salt)

    img_heavy_gauss = addGaussianNoise(img, 0.9)
    cv2.imwrite('./out/my_image_heavy_gaussian.jpg', img_heavy_gauss)

    img_darker = darker(img)
    cv2.imwrite('./out/my_image_darker.jpg', img_darker)

    img_brighter = brighter(img)
    cv2.imwrite('./out/my_image_brighter.jpg', img_brighter)

    img_blur = cv2.GaussianBlur(img, (7, 7), 1.5)
    cv2.imwrite('./out/my_image_blur.jpg', img_blur)

    img_flip = flip(img)
    cv2.imwrite('./out/my_image_flip.jpg', img_flip)

    rotated_90 = rotate(img, 90)
    cv2.imwrite('./out/my_image_r90.jpg', rotated_90)

    rotated_180 = rotate(img, 180)
    cv2.imwrite('./out/my_image_r180.jpg', rotated_180)

if __name__ == "__main__":
    main()
    
# image file path
# file_dir = r'E:/PycharmProjects/image_cluster-master/data/smoke_call/train/1/' 
# for img_name in os.listdir(file_dir):
#     img_path = file_dir + img_name
#     img = cv2.imread(img_path)
#     # cv2.imshow("1",img)
#     # cv2.waitKey(5000)
#     # rotate
#     rotated_90 = rotate(img, 90)
#     cv2.imwrite(file_dir + img_name[0:-4] + '_r90.jpg', rotated_90)
#     rotated_180 = rotate(img, 180)
#     cv2.imwrite(file_dir + img_name[0:-4] + '_r180.jpg', rotated_180)

# for img_name in os.listdir(file_dir):
#     img_path = file_dir + img_name
#     img = cv2.imread(img_path)
#     # flip
#     flipped_img = flip(img)
#     cv2.imwrite(file_dir +img_name[0:-4] + '_fli.jpg', flipped_img)

#     # add noises
#     # img_salt = SaltAndPepper(img, 0.3)
#     # cv2.imwrite(file_dir + img_name[0:7] + '_salt.jpg', img_salt)
#     img_gauss = addGaussianNoise(img, 0.3)
#     cv2.imwrite(file_dir + img_name[0:-4] + '_noise.jpg',img_gauss)

#     # brighter and darker
#     img_darker = darker(img)
#     cv2.imwrite(file_dir + img_name[0:-4] + '_darker.jpg', img_darker)
#     img_brighter = brighter(img)
#     cv2.imwrite(file_dir + img_name[0:-4] + '_brighter.jpg', img_brighter)

#     blur = cv2.GaussianBlur(img, (7, 7), 1.5)
#     #      cv2.GaussianBlur(image, kernel, standard deviation）
#     cv2.imwrite(file_dir + img_name[0:-4] + '_blur.jpg',blur)
