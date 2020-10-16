import matplotlib.pyplot as plt
import numpy as np


#y = np.loadtxt('psnr_mod.txt', dtype='str', delimiter=', ', unpack=True)
#x = range(0, len(y))
#arr = np.genfromtxt('psnr_mod.txt', delimiter=',', dtype=str)
arr = np.genfromtxt('ssim_mod.txt', delimiter=',', dtype=str)
arr = list(map(float, arr))
print('arr length: ', len(arr))
print('arr average: ', float(sum(arr)) / len(arr))
print('arr maximum: ', max(arr))
print('arr minimum: ', min(arr))

