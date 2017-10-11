import numpy as np
import pandas as pd

def ExtractRawFeature(filePath):
    result = []
    startSamples = []
    with open(filePath) as f:
        listOfLines = f.readlines()

    for lines in listOfLines:
        linePart = lines.split()
        idxOfFirstPathword = linePart.index('firstPath') if 'firstPath' in linePart else -1
        if(len(linePart) > 150):
            idx = linePart.index('#')
            del linePart[0:idx+1]
            result.append(linePart)
        elif(idxOfFirstPathword >= 0):
            startSamples.append(linePart[idxOfFirstPathword+1])

    result2 = []
    for l in result:
        result2.append(list(map(int, l)))

        startSamples = list(map(int, startSamples))

    return np.array(startSamples), np.array(result2)

def ExtractAmplitude(arr):
    resultArray = np.empty([arr.shape[0], int(arr.shape[1]/2)])

    if(int(arr.shape[1])%2 != 0):
        return

    for i in range(0, arr.shape[1],2):
        resultArray[:,int(i/2)] = np.sqrt(arr[:,i]**2+arr[:,i+1]**2)

    return resultArray

def PrepareSample(Amps, Starts, NumPacket, label):
    result = np.empty([Amps.shape[0], NumPacket])
    for i in range(0, Amps.shape[0]):
        result[i,:] = Amps[i, Starts[i]:Starts[i]+NumPacket]
    sample = pd.DataFrame(result)
    sample['Pos'] = label
    return sample

def makeDataSet(numSample):
    startA, resA = ExtractRawFeature('/home/deeplearning/PycharmProjects/UWB/locA.txt')
    startB, resB = ExtractRawFeature('/home/deeplearning/PycharmProjects/UWB/locB.txt')
    AmpsA = ExtractAmplitude(resA)
    AmpsB = ExtractAmplitude(resB)
    SampleA = PrepareSample(AmpsA, startA, numSample, 'A')
    SampleB = PrepareSample(AmpsB, startB, numSample, 'B')
    df = pd.concat([SampleA, SampleB])
    #df = df.reset_index()

    return df

#print(makeDataSet(10).shape)