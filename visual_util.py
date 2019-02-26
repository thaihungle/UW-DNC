import tensorflow as tf
import numpy as np
import pickle
import time
import sys
import os
import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

def plot_memory(emem_view, dmem_view, hold, brin, batch_ind=0):
    print(brin[batch_ind])


    eraddress = emem_view['read_weightings'][batch_ind,:,:,0]
    for i in range(eraddress.shape[0]):
        if hold[i]:
            eraddress[i,:] = np.zeros(eraddress[i,:].shape)
    ewaddress = emem_view['write_weightings'][batch_ind,:,:]
    for i in range(ewaddress.shape[0]):
        if  hold[i]:
            ewaddress[i,:] = np.zeros(ewaddress[i,:].shape)

    ewgates = emem_view['write_gates'][batch_ind, :,]

    # ewaddress2 = ewaddress[:-1,:]
    # ewaddress = ewaddress2

    draddress = dmem_view['read_weightings'][batch_ind, :, :, 0]
    # draddress2 = draddress[1:,:]
    # draddress = draddress2

    dwaddress = dmem_view['write_weightings'][batch_ind, :, :]

    fig = plt.figure()



    # a = fig.add_subplot(1, 3, 1)
    # a.set_title('E Read Weight')
    # a.set_aspect('auto')
    # a.xaxis.set_ticks(np.arange(0, eraddress.shape[1]),eraddress.shape[1]//min(eraddress.shape[1],10))
    # a.yaxis.set_ticks(np.arange(0, eraddress.shape[0]),eraddress.shape[0]//min(eraddress.shape[0],10))
    # plt.imshow(eraddress, interpolation='nearest', cmap='gray', aspect='auto')

    a = fig.add_subplot(1, 3, 1)
    a.set_title('Encoding Write gate')
    a.xaxis.set_ticks(np.arange(0, 1))
    a.yaxis.set_ticks(np.arange(0, ewgates.shape[0]),ewgates.shape[0]//min(ewgates.shape[0],10))
    plt.imshow(ewgates, interpolation='nearest', cmap='gray')

    a = fig.add_subplot(1, 3, 2)
    a.set_title('Encoding Write Weight')
    a.set_aspect('auto')
    a.xaxis.set_ticks(np.arange(0, ewaddress.shape[1]), ewaddress.shape[1]//min(ewaddress.shape[1],10))
    a.yaxis.set_ticks(np.arange(0, ewaddress.shape[0]), ewaddress.shape[0]//min(ewaddress.shape[0],10))
    plt.imshow(ewaddress, interpolation='nearest', cmap='gray', aspect='auto')

    a = fig.add_subplot(1, 3, 3)
    a.set_title('Decoding Read Weight')
    a.set_aspect('auto')
    a.xaxis.set_ticks(np.arange(0, draddress.shape[1]), draddress.shape[1]//min(draddress.shape[1],10))
    a.yaxis.set_ticks(np.arange(0, draddress.shape[0]), draddress.shape[0]//min(draddress.shape[0],10))
    plt.imshow(draddress, interpolation='nearest', cmap='gray', aspect='auto')

    # a = fig.add_subplot(2, 2, 4)
    # a.set_title('D Write Weight')
    # a.set_aspect('auto')
    # a.xaxis.set_ticks(np.arange(0, dwaddress.shape[1]), dwaddress.shape[1]//min(dwaddress.shape[1],10))
    # a.yaxis.set_ticks(np.arange(0, dwaddress.shape[0]), dwaddress.shape[0]//min(dwaddress.shape[0],10))
    # plt.imshow(dwaddress, interpolation='nearest', cmap='gray', aspect='auto')

    plt.show()

