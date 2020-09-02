from math import log2
from pandas import read_csv, DataFrame
import numpy as np


def unit_I(var_1, var_2):
    if (var_1 > 0) & (var_2 > 0):
        sum_1 = var_1 + var_2
        pro_1, pro_2 = var_1 / sum_1, var_2 / sum_1
        return -pro_1 * log2(pro_1) - pro_2 * log2(pro_2)
    else:
        return 0


def info(D):  # D = [class entries vector]
    return unit_I(D[0], D[1])


def infoAttribute(S1, S2):
    Sum_1 = sum(S1)
    temp = 0
    for idx in range(len(S1)):
        temp = temp + (S1[idx] / Sum_1) * info(S2[idx])
    return temp


def gain(V1, V2):
    return V1 - V2


def splitInfo(Arr):
    temp = 0
    Sum = sum(Arr)
    for x in Arr:
        if x > 0:
            temp = temp + (-x / Sum) * log2(x / Sum)
        else:
            return 0
    return temp


def getUniqueAttributes(Array):
    Labels = []
    for x in Array:
        if x not in Labels:
            Labels.append(x)
    return Labels


def getScore(Array, Labels):
    score = np.zeros(len(Labels))
    for x_num, x_name in enumerate(Array):
        ind = Labels.index(x_name)
        score[ind] = score[ind] + 1
    return score


def getPosNegScore(Array, Labels, ClassLabels, ClassArray):
    score = np.zeros((len(Labels), len(ClassLabels)))
    for x_num, x_name in enumerate(Array, start=0):
        for y_num, y_name in enumerate(Labels):
            if y_name == x_name:
                ind = ClassLabels.index(ClassArray[x_num])
                score[y_num][ind] = score[y_num][ind] + 1
    return score


def splitData(data, tags, Attribute):
    temp_idx = np.zeros(len(tags))
    splittingArray = list(data[Attribute])
    data = data.drop(columns=Attribute)
    col = data.columns
    splitedData = {}
    for idxNum in range(len(tags)):
        splitedData[idxNum] = DataFrame(columns=col)
    for idxNum, idxName in enumerate(splittingArray, start=0):
        for tagNum, tagName in enumerate(tags):
            if idxName == tagName:
                splitedData[tagNum].loc[temp_idx[tagNum]] = data.iloc[idxNum]
                temp_idx[tagNum] = temp_idx[tagNum] + 1
    return splitedData


def main():
    Data = {0: read_csv('datasetHomework1.csv', sep=';')}
    no_of_Steps = 2
    tree_structure = []
    for pdx in range(no_of_Steps):
        print('Step no: ', pdx)
        for tdx in range(len(Data)):
            Attributes = Data[tdx].columns[:-1]
            Class = Data[tdx].columns[-1]
            ClassLabels = getUniqueAttributes(Data[tdx][Class])
            ClassScores = getScore(Data[tdx][Class], ClassLabels)
            if len(ClassLabels) == 1:
                a = 0
            else:
                UniqueAttributes = []
                UniqueAttributesScores = []
                UniqueAttributesPosNegScores = []
                for idxNum, idxName in enumerate(Attributes, start=0):
                    UniqueAttributes.append(getUniqueAttributes(Data[tdx][idxName]))
                    UniqueAttributesScores.append(getScore(Data[tdx][idxName], UniqueAttributes[idxNum][:]))
                    UniqueAttributesPosNegScores.append(
                        getPosNegScore(Data[tdx][idxName], UniqueAttributes[idxNum][:], ClassLabels, Data[tdx][Class]))
                GainRatio = []
                Entropy = info(ClassScores)
                for idx in range(len(Attributes)):
                    infoAge = infoAttribute(UniqueAttributesScores[idx], UniqueAttributesPosNegScores[idx])
                    Gain = gain(Entropy, infoAge)
                    SplitInfoAge = splitInfo(UniqueAttributesScores[idx])
                    GainRatio.append(Gain / SplitInfoAge)

                idx1stLayer = GainRatio.index(max(GainRatio))
                for gtx in range(len(GainRatio)):
                    print('Attribute: ', Attributes[gtx], '\tGainRatio: ', GainRatio[gtx])
                # print(Attributes[idx1stLayer])
                # print(UniqueAttributes[idx1stLayer])
        Data = splitData(Data[tdx], UniqueAttributes[idx1stLayer], Attributes[idx1stLayer])
        # print(len(Data))


if __name__ == '__main__':
    main()
