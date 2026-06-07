import matplotlib.pyplot as plt
import numpy as np

types = {
    "Gaussian white noise" : "awgn.csv",
    "Impulse noise" : "impulse_magnitude.csv",
    "Multipath attenuations": "multipath_attenuations.csv",
    "Sinusoidal interference" : "sin_strength.csv"
}

for key, value in types.items():
    titles = np.loadtxt(value, delimiter =",", max_rows=1,dtype=str)
    data = np.loadtxt(value, delimiter = ",",skiprows=1)


    fig, ax = plt.subplots()

    ax.set_xlabel(titles[0])
    ax.set_ylabel("BER")
    ax.set_title(key)

    dataT = np.transpose(data)
    ax.plot(dataT[0], dataT[1])

    plt.savefig(key + ".jpg",dpi = 800)
    plt.show()
