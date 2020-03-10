import math
from sys import argv
from types import SimpleNamespace
import numpy as np
import seaborn
import pandas as pd
from matplotlib import pyplot as plt

"""
    File:   HW05_Verdesi_Jacob_Mentor.py
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
        self.sign=None
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
                if gini < self.bestGini:
                    self.bestAttribute = attribute
                    self.bestSplit = split
                    self.bestThresh = attThresh
                    self.bestGini = gini
        self.left=decisionTree(self.bestSplit[0],self.depth+1)
        self.right=decisionTree(self.bestSplit[1],self.depth+1)

    def __str__(self):
        if self.leaf:
            return ""
        return "\n"+'|\t' * self.depth + self.bestAttribute+" "+str(self.bestThresh)+" "+str(self.bestGini)+" "+str(self.left)+str(self.right)

def comment(str):
    return "\"\"\"{" + str + "}\"\"\";"


def indent(train):
    tab = 0
    out = ""
    for c in train:

        if (c == '{'):
            tab += 1
        elif (c == '}'):
            tab -= 1
        elif(c == ';'):
            out += "\n"+ (tab*4 * ' ')
        else:
            out += c
    return out


def header(file):
    return "import pandas as pd;" \
           + comment("File:" + file.name + ";Author: Jacob Verdesi;Email:jxv3386@rit.edu;"
                                           "Description:This is a Trained program for Classifying Abominable Data")


def body(mainTree):

    return "def printClassified(data):{;" \
           "right=0;" \
           "data[\"Prediction\"]=\"\";" \
           "for index, row in data.iterrows():{;" \
           +printNode(mainTree) \
           +"}};"
def printNode(treeNode):
    if not treeNode.leaf:
        return f"if row[\"{treeNode.bestAttribute}\"] <= {str(treeNode.bestThresh)}:"+"{;"\
               +printNode(treeNode.left)+"};"\
               +"else:{;"\
               +printNode(treeNode.right)+"};"
    else:
        return f"data.loc[index,\"Prediction\"]={str(treeNode.type)};right+=(row[\"Class\"]=={str(treeNode.type)})"

def print_trailer(valdation_file):
    return "def main():{;" + \
           comment("Main function") \
           + f"fileName=\"{valdation_file}\";" \
           + "data=(pd.read_csv(fileName,sep=','));" \
           + "printClassified(data);" \
           + "};if __name__ == '__main__':{;main();}"


def quantize(data):
    # for attribute in data:
    #     if not attribute== "Class":
    #         Amax,Amin=math.ceil(max(data[attribute])),math.ceil(min(data[attribute]))
    #         bin=1+math.floor(math.sqrt(Amax-Amin)/2)
    #         data[attribute]=data[attribute].apply(lambda x:bin*(round(x/bin)))
    data.quantized=SimpleNamespace()
    data.quantized={"Age":2,"Ht":5,"TailLn":1,"HairLn":1,"BangLn":1,"Reach":1,"EarLobes":1}
    for attribute in data.quantized:
        closest=data.quantized.get(attribute)
        data[attribute] = data[attribute].apply(lambda x: closest * (round(x / closest)))
    return data


def get_quantized_bin_size(data, attribute):
    """
    This function looks for the bin_size by sorting the values and then calculating
    the diffrence of 2 values that are next to eachother and are not the same
    :param data: DataFrame
    :param attribute: attribute to find bin_size
    :return: bin_size
    """
    return data.quantized.get(attribute)
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
        left.quantized = SimpleNamespace()
        right.quantized = SimpleNamespace()

        left.quantized=data.quantized
        right.quantized=data.quantized

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

    if depth > 2 or data.shape[0]<27 or .95>=data[data["Class"]==1].count()["Class"]/data.shape[0]<.05:
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
        decisionTreeData=data.copy()
        for index, row in testData.iterrows():
            decisionTreeData=decisionTreeData.drop(index)

        tree=decisionTree(decisionTreeData,0)
        print(tree)
        avg.append(test(testData,tree))
    print(avg)
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
    #nFold(5,data)
    mainTree=decisionTree(data,0)
    print(mainTree)
    validationData="VALIDATION_DATA_TO_RELEASE.csv"
    trainer = indent(header(file) + body(mainTree) + print_trailer(validationData)).replace("\n\n","\n")
    file.write(trainer)
    file.close()


if __name__ == '__main__':
    main()
