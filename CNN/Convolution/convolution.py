#%%
import numpy as np
from scipy.signal import convolve2d
from imageio import imread
import matplotlib.pyplot as plt

#%% Manual 2D convolution in numpy/scipy
# image
imgN = 20
image = np.random.randn(imgN,imgN)

# convolution kernel
kernelN = 7
Y,X = np.meshgrid(np.linspace(-3,3,kernelN),np.linspace(-3,3,kernelN))
kernel = np.exp(-(X**2+Y**2)/7)

fig,ax = plt.subplots(1,2,figsize = (8,6))
ax[0].imshow(image)
ax[0].set_title('Original Image')

ax[1].imshow(kernel)
ax[1].set_title('Convolution Kernel')
plt.show()

#%%
convoutput = np.zeros((imgN,imgN))
halfKr = kernelN//2
for rowi in range(halfKr,imgN-halfKr):
    for coli in range(halfKr,imgN-halfKr):
        pieceOfImg = image[rowi-halfKr:rowi+halfKr+1,:]
        pieceOfImg = pieceOfImg[:,coli-halfKr:coli+halfKr+1]

        dotprod = np.sum(pieceOfImg*kernel[::-1,::-1])
        convoutput[rowi,coli] = dotprod

#%%
convoutput2 = convolve2d(image,kernel,mode='valid')

fig,ax = plt.subplots(2,2,figsize = (8,8))
ax[0,0].imshow(image)
ax[0,0].set_title('Original Image')

ax[0,1].imshow(kernel)
ax[0,1].set_title('Convolution Kernel')

ax[1,0].imshow(convoutput)
ax[1,0].set_title('Manual Convolution Output')

ax[1,1].imshow(convoutput2)
ax[1,1].set_title('Scipy Convolution Output')

plt.show()

