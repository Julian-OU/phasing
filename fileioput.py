import cupy as cp
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm
from numpy.fft import ifftshift as ishift
from PIL import Image
from scipy import io


def ReadFile(path: str, node: str = None):
    type = path.split(".")[-1]
    if type in ["tif", "tiff", "png", "jpg", "jpeg"]:
        data = cp.array(Image.open(path).convert("L"))
    elif type == "mat":
        mat = io.loadmat(path)
        data = cp.array(eval("mat['" + node + "']"))
    elif type == "npy":
        data = cp.array(np.load(path, mmap_mode="r"))
    elif type == "h5" or type == "cxi":
        h5 = h5py.File(path, "r")
        data = cp.array(eval("h5['" + "']['".join(node.split("->")) + "'][()]"))
        h5.close()
    elif type == "mrc":
        data=cp.asarray(ReadMRC(path))
    else:
        RuntimeError("Unsupported file type")
    return data


def showimage(data, x=None, y=None, cmap="gray", vmax=None, vmin=None):
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_ticks_position("top")
    if type(x) == type(None):
        x = np.arange(0, data.shape[0] + 1)
    if type(y) == type(None):
        y = np.arange(1, data.shape[0] + 1)
    plt.pcolormesh(
        y, x, data, cmap=cmap, vmax=vmax, vmin=vmin, shading="auto", rasterized=True
    )
    pass


def WriteH5(data, path):
    h5 = h5py.File(path, "w")
    h5.create_dataset("data", data=data)
    h5.close()
    pass


def WriteMAT(data, path):
    io.savemat(path, {"data": data})
    pass


def WritePNG(data, path):
    data = np.abs(ishift(data))
    data = np.array(np.floor(data * (255 / np.max(data))), dtype="uint8")
    Image.fromarray(data).save(path)
    pass


def WriteTIF(data, path):
    data = np.abs(ishift(data))
    data = np.array(np.floor(data * (65535 / np.max(data))), dtype="uint16")
    Image.fromarray(data).save(path)
    pass


def WriteCSV(data, path):
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    pass

def ReadMRC(filename, dtype=float, order="C"):
    """
    * readMRC *
    Read in a volume in .mrc file format. See http://bio3d.colorado.edu/imod/doc/mrc_format.txt
    :param filename: Filename of .mrc
    :return: NumPy array containing the .mrc data
    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved.
    """
    import struct
    headerIntNumber = 56
    sizeof_int = 4
    headerCharNumber = 800
    sizeof_char = 1
    with open(filename,'rb') as fid:
        int_header = struct.unpack('=' + 'i'*headerIntNumber, fid.read(headerIntNumber * sizeof_int))
        char_header = struct.unpack('=' + 'c'*headerCharNumber, fid.read(headerCharNumber * sizeof_char))
        dimx, dimy, dimz, data_flag= int_header[:4]
        if (data_flag == 0):
            datatype='u1'
        elif (data_flag ==1):
            datatype='i1'
        elif (data_flag ==2):
            datatype='f4'
        elif (data_flag ==3):
            datatype='c'
        elif (data_flag ==4):
            datatype='f4'
        elif (data_flag ==6):
            datatype='u2'
        else:
            raise ValueError("No supported datatype found!\n")

        return np.fromfile(file=fid, dtype=datatype,count=dimx*dimy*dimz).reshape((dimx,dimy,dimz),order=order).astype(dtype)


def WriteMRC(filename, arr, datatype='f4', order="C", pixel_size=1):
    """
    * writeMRC *
    Write a volume to .mrc file format. See http://bio3d.colorado.edu/imod/doc/mrc_format.txt
    This version is bare-bones and doesn't write out the full header -- just the critical bits and the
    volume itself
    :param filename: Filename of .mrc file to write
    :param arr: NumPy volume of data to write
    :param dtype: Type of data to write
    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved
    """
    dimx, dimy, dimz = np.shape(arr)
    if datatype != arr.dtype:
        arr = arr.astype(datatype)
    # int_header = np.zeros(56,dtype='int32') #must be 4-byte ints
    int_header1 = np.zeros(10,dtype='int32') #must be 4-byte ints
    float_header1 = np.zeros(6,dtype='float32') #must be 4-byte ints
    int_header2 = np.zeros(3,dtype='int32') #must be 4-byte ints
    float_header2 = np.zeros(3,dtype='float32') #must be 4-byte ints
    int_header3 = np.zeros(34,dtype='int32') #must be 4-byte ints

    if (datatype == 'u1'):
        data_flag = 0
    elif (datatype =='i1'):
        data_flag = 1
    elif (datatype =='f4'):
        data_flag = 2
    elif (datatype =='c'):
        data_flag = 3
    elif (datatype =='f4'):
        data_flag = 4
    elif (datatype =='u2'):
        data_flag = 6
    else:
        raise ValueError("No supported datatype found!\n")
    int_header1[:4] = (dimx,dimy,dimz,data_flag)
    int_header1[7:10] = (dimx,dimy,dimz)
    float_header1[:3] = (pixel_size * dimx, pixel_size * dimy, pixel_size * dimz)
    int_header2[:3] = (1, 2, 3)
    float_header2[:3] = np.min(arr), np.max(arr), np.mean(arr)
    char_header = str(' '*800)
    with open(filename,'wb') as fid:
        fid.write(int_header1.tobytes())
        fid.write(float_header1.tobytes())
        fid.write(int_header2.tobytes())
        fid.write(float_header2.tobytes())
        fid.write(int_header3.tobytes())
        fid.write(char_header.encode('UTF-8'))
        fid.write(arr.tobytes(order=order))


def plotModel(pattern0, pattern1, pic, path):
    size = np.shape(pattern0)
    fig = plt.figure(figsize=(9, 3))
    ax1 = fig.add_axes([0.1, 0.1, 0.29, 0.87])
    ax1.pcolor(
        np.linspace(-0.5, 0.5, size[0] + 1),
        np.linspace(-0.5, 0.5, size[1] + 1),
        np.abs(pattern1),
        cmap="jet",
        norm=SymLogNorm(1, vmin=0, vmax=65536, base=2),
    )
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel("Noise-free oversampled diffraction pattern")
    ax2 = fig.add_axes([0.4, 0.1, 0.29, 0.87])
    ax2.pcolor(
        np.linspace(-0.5, 0.5, size[0] + 1),
        np.linspace(-0.5, 0.5, size[1] + 1),
        np.abs(pattern1),
        cmap="jet",
        norm=SymLogNorm(1, vmin=0, vmax=65536, base=2),
    )
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("Oversampled diffraction pattern with 10% noise")
    ax3 = fig.add_axes([0.7, 0.1, 0.29, 0.87])
    ax3.pcolor(np.flip(pic, axis=0), vmax=1, vmin=0, cmap="gray")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlabel("Model image")
    fig.savefig(path, dpi=600)
    plt.close()


def plotResult(
    frc0,
    frc1,
    pic,
    twin,
    path: str,
    iter: int,
    method: str,
    error0: float,
    error1: float,
):
    f = np.linspace(0, 0.5, np.shape(frc0)[0] + 1)[:-1]
    fig = plt.figure(figsize=(9, 3))
    ax1 = fig.add_axes([0.06, 0.15, 0.33, 0.82])
    ax1.plot(f, frc0, "C03", linewidth=1, label="Ideal R=%.3f" % error0)
    ax1.plot(f, frc1, "C00", linestyle="--", linewidth=1, label="Twin R=%.3f" % error1)
    ax1.set_xlim(-0.025, 0.525)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("Frequency (1/pixel)")
    ax1.set_ylabel("FRC")
    ax1.legend()

    ax2 = fig.add_axes([0.4, 0.1, 0.29, 0.87])
    ax2.pcolor(twin, vmax=1, vmin=-1, cmap="jet")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("Fourier space orientation")

    ax3 = fig.add_axes([0.7, 0.1, 0.29, 0.87])
    ax3.pcolor(np.flip(pic, axis=0), vmax=1, vmin=0, cmap="gray")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlabel("Reconstructed image")
    fig.text(0.01, 0.08, method)
    fig.text(0.01, 0.02, "iter=" + str(iter))
    fig.savefig(path, dpi=600)
    plt.close()


def plotFRC(frc0, frc1, pic, path, method):
    t = cp.shape(frc0)[0]
    fig = plt.figure(figsize=(9, 3))
    ax1 = fig.add_axes([0.08, 0.15, 0.33, 0.82])
    ax1.pcolor(
        np.linspace(0, 0.5, np.shape(frc0)[1] + 1),
        np.arange(0, t + 1),
        frc0,
        vmax=1,
        vmin=0,
    )
    ax1.set_xlabel("Frequency (1/pixel)")
    ax1.set_ylabel("Iteration")
    ax2 = fig.add_axes([0.41, 0.15, 0.33, 0.82])
    im = ax2.pcolor(
        np.linspace(0, 0.5, np.shape(frc1)[1] + 1),
        np.arange(0, t + 1),
        frc1,
        vmax=1,
        vmin=0,
    )
    ax2.set_xlabel("Frequency (1/pixel)")
    ax2.set_xticks(np.arange(0.1, 0.6, 0.1))
    ax2.set_yticks([])
    ax3 = fig.add_axes([0.7, 0.1, 0.29, 0.87])
    ax3.imshow(pic, vmax=1, vmin=0, cmap="gray")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlabel("Ideal image (sample)")
    fig.colorbar(im, ax=[ax1, ax2], pad=0.01)
    fig.text(0.09, 0.04, "Ideal", color="C03")
    fig.text(0.37, 0.04, "Twin", color="C00")
    fig.text(0.66, 0.06, "FRC")
    fig.text(0.01, 0.02, method)
    fig.savefig(path, dpi=600)


def plottwin(twin, path):
    plt.imshow(twin, cmap="jet", vmax=1.5, vmin=-1.5)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path, dpi=600)
    plt.close()
