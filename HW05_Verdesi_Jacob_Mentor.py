import math
from sys import argv

import numpy as np
import seaborn
import pandas as pd
from joblib.numpy_pickle_utils import xrange
from matplotlib import pyplot as plt

"""
    File:   HW02_Verdesi_Jacob_Mentor.py
    Author: Jacob Verdesi
    Email:  jxv3386@rit.edu
    Description:
    This is a Mentor program for Classifying Abominable Data

"""


class Tree(object):
    def __init__(self,data,depth):
        self.data=data
        self.depth = depth
        self.leaf = False
        self.type= None
    def split(self):
        self.bestAttribute = self.data.columns[0]
        self.bestSplit = (pd.DataFrame(), pd.DataFrame())
        self.bestThresh = min(self.data.columns[0])
        self.bestGini=math.inf
        for attribute in self.data:
            if attribute == "Class":
                pass
            else:
                attThresh, gini, split, giniList, thresh = bestThreshold(self.data, attribute)
                #step = (1000 - 0) / float(len(giniList))
                #rang = [0 + i * step for i in xrange(len(giniList))]
                #plt.plot(rang, giniList, label=attribute)
                if gini < self.bestGini:
                    self.bestAttribute = attribute
                    self.bestSplit = split
                    self.bestThresh = attThresh
                    self.bestGini = gini
                #space = "\t" * self.depth
                #print(f"{space}Attribute: {attribute}, Threshold: {attThresh}, Gini: {gini}")
        self.left=decisionTree(self.bestSplit[0],self.depth+1)
        self.right=decisionTree(self.bestSplit[1],self.depth+1)

    def __str__(self):
        if self.leaf:
            return ""
        return "\n"+'|\t' * self.depth + self.bestAttribute+" "+str(self.bestThresh)+" "+str(self.bestGini)+" "+str(self.right)+str(self.left)

def comment(str):
    return "\"\"\"{" + str + "}\"\"\";"


def indent(train):
    tab = 0
    out = ""
    for c in train:
        if (c == '{' or c == '}' or c == ';'):
            if (c == '{'):
                tab += 1
            if (c == '}'):
                tab -= 1
            out += "\n" + (tab * '\t')
        else:
            out += c
    return out


def header(file):
    return "import pandas as pd;" \
           + comment("File:" + file.name + ";Author: Jacob Verdesi;Email:jxv3386@rit.edu;"
                                           "Description:This is a Trained program for Classifying Abominable Data")


def body():
    return "def printClassified(data,bestAttribute,bestThreshold):{" \
           "for i in data[bestAttribute]:{" \
           "if (bestAttribute==\"Height\" and i>bestThreshold) or (bestAttribute==\"Age\" and i<bestThreshold):{" \
           "print(-1);}else:{print(1);}}}"


def print_trailer(bestAttribute, bestThreshold):
    return "def main():{" + \
           comment("Main function") \
           + "fileName=\"the_validation_file.csv\";" \
           + "data=(pd.read_csv(fileName,sep=','));" \
           + "printClassified(data,\"" + str(bestAttribute) + "\"," + str(bestThreshold) + ");" \
           + "}if __name__ == '__main__':{main()}"


def quantize(data):
    # for attribute in data:
    #     if not attribute== "Class":
    #         Amax,Amin=math.ceil(max(data[attribute])),math.ceil(min(data[attribute]))
    #         bin=1+math.floor(math.sqrt(Amax-Amin)/2)
    #         data[attribute]=data[attribute].apply(lambda x:bin*(round(x/bin)))
    data['Age'] = data['Age'].apply(lambda x: 2 * (round(x / 2)))
    data['Ht'] = data['Ht'].apply(lambda x: 5 * (round(x / 5)))
    data['TailLn'] = data['TailLn'].apply(lambda x: 1 * (round(x / 1)))
    data['HairLn'] = data['HairLn'].apply(lambda x: 1 * (round(x / 1)))
    data['BangLn'] = data['BangLn'].apply(lambda x: 1 * (round(x / 1)))
    data['Reach'] = data['Reach'].apply(lambda x: 1 * (round(x / 1)))
    data['EarLobes'] = data['EarLobes'].apply(lambda x: 1 * (round(x / 1)))

    return data


def get_quantized_bin_size(data, attribute):
    """
    This function looks for the bin_size by sorting the values and then calculating
    the diffrence of 2 values that are next to eachother and are not the same
    :param data: DataFrame
    :param attribute: attribute to find bin_size
    :return: bin_size
    """
    list = sorted(data[attribute])
    for index in range(0, len(data[attribute])):
        if(index==len(list)-1):
            return 1
        if (list[index] != list[index + 1]):
            return list[index + 1] - list[index]

def bestThreshold(data,attribute):
    index = get_quantized_bin_size(data, attribute)
    bestThresh=0
    bestGini=math.inf
    bestSplit=(pd.DataFrame(),pd.DataFrame())
    thresh=[]
    giniList=[]
    for threshold in range(min(data[attribute]), max(data[attribute]) + index, index):
        left = data[data[attribute] <= threshold]
        right = data[data[attribute] > threshold]
        dataTotal=data.shape[0]
        leftA,leftB,leftTotal=left[left["Class"]==1].count()["Class"],left[left["Class"]==-1].count()["Class"],left.shape[0]
        rightA,rightB,rightTotal=right[right["Class"]==1].count()["Class"],right[right["Class"]==-1].count()["Class"],right.shape[0]
        #print(leftA,leftB,rightA,rightB,rightTotal,leftTotal)
        leftGini,rightGini=0,0
        if leftTotal>0:
            leftGini=1-math.pow(leftA/leftTotal,2)-math.pow(leftB/leftTotal,2)
        if rightTotal>0:
            rightGini=1-math.pow(rightA/rightTotal,2)-math.pow(rightB/rightTotal,2)
        weightedGini=leftGini*(leftTotal/dataTotal)+rightGini*(rightTotal/dataTotal)
        thresh.append(threshold)
        giniList.append(weightedGini)

        if weightedGini < bestGini:
            bestThresh = threshold
            bestGini=weightedGini
            bestSplit=(left,right)


    return bestThresh,bestGini,bestSplit,giniList,thresh
def decisionTree(data, depth):

    if depth > 3 or data.shape[0]<27:
        tree=Tree(data,depth)
        tree.leaf=True
        if data[data["Class"]==1].count()["Class"]>data[data["Class"]==-1].count()["Class"]:
            tree.type=1
        else:
            tree.type=-1
        return tree
    else:
        tree=Tree(data,depth)
        tree.split()

        return tree
def nFold(n,data):
    dataArray=np.array_split(data,n)
    avg=[]
    for testData in dataArray:
        decisionTreeData=dataArray.copy()
        decisionTreeData.remove(testData)
        print(decisionTreeData)
        #tree=decisionTree(data2,0)
        #avg.append(test(decisionTreeData,tree))

    print(sum(avg)/len(avg))
def test(data,mainTree):
    right = 0
    wrong = 0
    for index, row in data.iterrows():
        tree=mainTree
        while not tree.leaf:
            if row[tree.bestAttribute]<=tree.bestThresh:
                tree=tree.left
            else:
                tree=tree.right
        if(row.Class==tree.type):
            right+=1
        else:
            wrong+=1
    return right/(wrong+right)
def main():
    fileName = argv[1]
    writeFile = "HW05_Verdesi_Jacob_Trainer.py"
    file = open(writeFile, "w")

    data = pd.read_csv(fileName, sep=',')
    if(fileName=="Abominable_data_HW05_v420.csv"):
        data=quantize(data)
        data["Class"] = data['Class'].replace({"Assam": -1, "Bhuttan": 1})
    #seaborn.pairplot(data)
    nFold(3,data)
    #mainTree=decisionTree(data,0)
    #print(mainTree)

    #trainer = indent(header(file) + body() + print_trailer(bestAttribute, bestThreshold))
    #file.write(trainer)
    file.close()


if __name__ == '__main__':
    main()
