# Signal Processing module

import numpy as np
from functools import lru_cache

class Data:
    def __init__(self):
        self.__data = []
        self.__freq = 4e10

    def setSamplingFreq(self, samplingFreq):
        self.__freq = samplingFreq

    def setDataFromFile(self, fileName):
        with open(fileName, "rb") as file:
            readData = file.read()
            self.__data = list(readData)

    def setDataFromBytes(self, byteString):
        self.__data = list(byteString)

    def setDataFromBin(self, binString):
        self.__data = []
        length = int(len(binString)/8)
        for i in range(length):
            self.__data.append(int(binString[i*8:i*8+8],2))

    def setDataFromIntList(self, intList):
        self.__data = intList

    def dataIntList(self):
        return self.__data

    def dataBytes(self):
        return bytes(self.__data)

    def saveFile(self, fileName):
        with open(fileName, "wb") as file:
            file.write(self.dataBytes())

    def toBinaryStr(self):
        lst = ""
        for i in self.__data:
            lst += format(i, "08b")

        return lst

    def bpskMod(self, freq=10e9, Tb=1e-7, Eb=40e-6):
        binlst = self.toBinaryStr()

        lst = np.array(list(binlst), dtype=int)

        # Function for bpsk wave
        A = np.sqrt(2*Eb/Tb)
        omega = 2*np.pi*freq
        signal = lambda t : A*np.cos(omega*t + np.pi*(1-lst[(t/Tb).astype(int)]))

        return signal

    @staticmethod
    def bpskDemod(signal,freq=10e9, Tb=1e-7):

        omega = 2*np.pi*freq
        refSig = lambda t : np.cos(omega*t)

        product = lambda t : signal(t) * refSig(t)

        bits = ""
        end = False
        i = 0

        while not end:
            try:
                print("Run no.", i)
                tPoints = np.arange(i*Tb,(i+1)*Tb, 1/(freq))
                integral = np.sum(product(tPoints))
                if integral > 0:
                    bits += "1"
                else:
                    bits += "0"
                i+=1
            except IndexError:
                end = True
                print("Reached end of data")

        return bits

