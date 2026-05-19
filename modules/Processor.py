# Signal Processing module

import numpy as np

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

    def setDataFromIntList(self, intList):
        self.__data = intList

    def dataIntList(self):
        return self.__data

    def dataBytes(self):
        return bytes(self.__data)

    def saveFile(self, fileName):
        with open(fileName, "wb") as file:
            file.write(self.dataBytes())

    def fftFunction(self):
        size = len(self.__data)
        fourierData = np.fft.fft(self.__data)
        amplitude = np.abs(fourierData)
        phase = np.angle(fourierData)
        freq = np.fft.fftfreq(size,1/self.__freq)

        return amplitude

