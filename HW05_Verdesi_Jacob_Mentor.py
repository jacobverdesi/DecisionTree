import math
from sys import argv
import seaborn
import pandas as pd
from matplotlib import pyplot as plt

"""
    File:   HW02_Verdesi_Jacob_Mentor.py
    Author: Jacob Verdesi
    Email:  jxv3386@rit.edu
    Description:
    This is a Mentor program for Classifying Abominable Data

"""


class Tree(object):
    def __init__(self,left,right,attribute,depth,thresh,gini):
        self.left = left
        self.right = right
        self.attribute=attribute
        self.depth = depth
        self.thresh=thresh
        self.gini = gini
        self.rightNode=False
        self.leaf = False

    def __str__(self):
        if self.leaf:
            return ""
        return "\n"+'|\t' * self.depth + self.attribute+" "+str(self.thresh)+" "+str(self.gini)+" "+str(self.right)+str(self.left)

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
    for index in range(0, len(data[attribute])):
        list = sorted(data[attribute])
        if (list[index] != list[index + 1]):
            return list[index + 1] - list[index]


def decisionTreeHelper(data):
    return decisionTree(data, [att for att in data], 0)
def bestThreshold(data,attribute,lessThanThresh=False):
    index = get_quantized_bin_size(data, attribute)
    bestThresh=0
    bestGini=0
    thresh=[]
    giniList=[]
    for threshold in range(min(data[attribute]), max(data[attribute]) + index, index):
        lessThan=data[attribute][data[attribute] <threshold].count()
        greaterEqualTo=data[attribute][data[attribute]>=threshold].count()
        gini=1-math.pow(lessThan/data.shape[0],2)-math.pow(greaterEqualTo/data.shape[0],2)
        thresh.append(threshold)
        giniList.append(gini)
        if gini > bestGini:
            bestThresh = threshold
            bestGini=gini
    # plt.scatter(thresh, giniList)
    # plt.show()
    return bestThresh,bestGini
def decisionTree(data, attributes, depth):
    if depth > 5 :
        tree=Tree(None,None,"leaf",depth,0,0)
        tree.leaf=True
        return tree
    else:
        bestSplit = attributes[0]
        bestThresh = min(attributes[0])
        goodNess = 0
        #print(depth)
        for attribute in attributes:
            if attribute == "Class":
                pass
            else:
                attThresh,gini=bestThreshold(data,attribute)
                if gini>goodNess:
                    bestSplit=attribute
                    bestThresh=attThresh
                    goodNess=gini
                print(f"\tAttribute: {attribute}, Threshold: {attThresh}, Misses: {gini}")
        print(bestSplit,bestThresh,goodNess)


        left = data[data[bestSplit] < bestThresh]
        right = data[data[bestSplit] >= bestThresh]
        attributes.remove(bestSplit)
        leftList=attributes.copy()
        rightList=attributes.copy()
        left.drop(columns=[bestSplit])
        right.drop(columns=[bestSplit])
        leftT=decisionTree(left,leftList,depth+1)
        rightT=decisionTree(right,rightList,depth+1)
        tree=Tree(leftT,rightT,bestSplit,depth,bestThresh,goodNess)
        return tree

def main():
    fileName = argv[1]
    writeFile = "HW05_Verdesi_Jacob_Trainer.py"
    file = open(writeFile, "w")

    data = quantize(pd.read_csv(fileName, sep=','))
    if(fileName=="Abominable_data_HW05_v420.csv"):
        data["Class"] = data['Class'].replace({"Assam": -1, "Bhuttan": 1})
    #seaborn.plot(data)
    print(decisionTreeHelper(data))
    #trainer = indent(header(file) + body() + print_trailer(bestAttribute, bestThreshold))
    #file.write(trainer)
    file.close()


if __name__ == '__main__':
    main()
