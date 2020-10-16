import matplotlib.pyplot as plt
import numpy as np


#y = np.loadtxt('psnr_mod.txt', dtype='str', delimiter=', ', unpack=True)
#x = range(0, len(y))
arr = np.genfromtxt('psnr_original.txt', delimiter=',', dtype=str)
x = range(0, len(arr))
#arr = np.genfromtxt('ssim_mod.txt', delimiter=',', dtype=str)
arr = list(map(float, arr))
#print(arr)
plt.plot(arr)
#print('arr length: ', len(arr))
y = []
average = float(sum(arr)) / len(arr)
for i in range(len(arr)):
	y.append(average)
plt.plot(x, y, color='red', linestyle='--')
plt.text(len(arr)/2, average, '%d' % average, size=12)
plt.ylim(0, 100)
#plt.ylim(0, 1.1)
#plt.yticks(np.arange(0.0, 100.0, 20.0))
#plt.yticks([0, 20, 40, 60, 80, 100], [0, 20, 40, 60, 80, 100])
#plt.locator_params('y', nbins=5)

plt.xlabel('sample')
#plt.ylabel('ssim')
plt.ylabel('psnr')
#plt.title('SSIM')
plt.title('PSNR')
#plt.legend
#plt.show()
plt.savefig('psnr__original_legend.png', dpi=300)
