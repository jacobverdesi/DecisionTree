import math
from sys import argv

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
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None
        self.depth = None


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
    data['Reach'] = data['Reach'].apply(lambda x: 5 * (round(x / 5)))
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
    return decisionTree(data, [att for att in data], Tree, 0)
def bestThreshold(data,attribute,lessThanThresh=False):
    index = get_quantized_bin_size(data, attribute)
    leastMisses=math.inf
    attThresh=0
    for threshold in range(min(data[attribute]), max(data[attribute]) + index, index):
        out = data.copy()
        if(lessThanThresh):
            out['left'] = (data[attribute] >= threshold)
        else:
            out['left'] = (data[attribute] < threshold)
        out['sum'] = out['left'] + out['Class']
        miss = ((out['sum']) == 1).sum()
        # print(attribute,threshold,"Misses",miss,"\n",out)
        if miss < leastMisses:
            leastMisses = miss
            attThresh = threshold
    return attThresh,leastMisses
def decisionTree(data, attributes, tree, depth):
    if depth >= 3:
        pass
    else:
        bestSplit = attributes[0]
        bestThresh = min(attributes[0])
        goodNess = math.inf
        for attribute in attributes:
            if attribute == "Class":
                pass
            else:
                attThresh,leastMisses=bestThreshold(data,attribute)
                if leastMisses>len(data[attribute])/2:
                    attThresh, leastMisses = bestThreshold(data, attribute,True)
                if leastMisses<goodNess:
                    bestSplit=attribute
                    bestThresh=attThresh
                    goodNess=leastMisses
                print(f"Attribute: {attribute}, Threshold: {attThresh}, Misses: {leastMisses}")
        print(bestSplit,bestThresh,goodNess)
        return

    # for attribute in data:
    #     if attribute=="Class" :
    #         pass
    #     else:
    #         index=get_quantized_bin_size(data,attribute)
    #         tprList = []
    #         fprList = []
    #         minMiss=math.inf
    #         minDistance=math.inf
    #         thresh=0
    #         misses=[]
    #         for threshold in range(min(data[attribute]),max(data[attribute]),index):
    #             out = data.copy()
    #             if (attribute=='Age'):
    #                 out['left']=(data[attribute]>threshold)
    #             else:
    #                 out['left']=(data[attribute]<=threshold)
    #             out['sum']=out['left']+out['Class']
    #             miss=((out['sum'])==1).sum()+((out['sum'])==0).sum()
    #             misses.append(miss)
    #             tpr=(out['sum']==1).sum()/((out['sum']==1).sum()+(out['sum']==2).sum())
    #             fpr=((out['sum']==-1).sum()/((out['sum']==0).sum()+(out['sum']==-1).sum()))
    #             tprList.append(tpr)
    #             fprList.append(fpr)
    #             distance=math.sqrt(pow(1-fpr,2)+pow(tpr,2))
    #             if(distance<minDistance):
    #                 minDistance=distance
    #                 minDistThresh=threshold
    #             if(miss<minMiss):
    #                 minMiss=miss
    #                 thresh=threshold
    #
    #         if(minMiss<bestMinMiss):
    #             bestThresh = thresh
    #             bestAttribute = attribute
    #             bestAttributeIndex = index
    #             bestMisses = misses
    #         print(attribute,minMiss)
    #
    #
    # print(bestMinMiss,bestThresh)
    # print(bestAttribute)
    # print(minDistance,minDistThresh)
    # bins=[x for x in range(min(data[bestAttribute]),max(data[bestAttribute]),bestAttributeIndex)]
    # plt.scatter(bins,bestMisses)
    # # plt.hist(data[bestAttribute],bins=bins,label="total",color='purple',alpha=.5,histtype='stepfilled')
    # # plt.hist(tp[bestAttribute],bins=bins,label="tp",color='b',alpha=.5,histtype='stepfilled')
    # # plt.hist(fp[bestAttribute],bins=bins, label="fp", color='y', alpha=.5, histtype='stepfilled')
    # # plt.hist(tn[bestAttribute],bins=bins,label="tn",color='g',alpha=.5,histtype='stepfilled')
    # # plt.hist(fn[bestAttribute],bins=bins,label="fn",color='r',alpha=.5,histtype='stepfilled')
    # # plt.legend(prop={'size': 10})
    #
    # #plt.scatter(bestTpr,bestFpr)
    # #rocIndex=(minDistThresh - min(data[bestAttribute])) // bestAttributeIndex
    # #plt.annotate(minDistThresh,(bestTpr[rocIndex],bestFpr[rocIndex]))
    # plt.show()
    # return bestAttribute,bestThresh


def main():
    fileName = argv[1]
    writeFile = "HW05_Verdesi_Jacob_Trainer.py"
    file = open(writeFile, "w")

    data = quantize(pd.read_csv(fileName, sep=','))
    if(fileName=="Abominable_data_HW05_v420.csv"):
        data["Class"] = data['Class'].replace({"Assam": 0, "Bhuttan": 1})
    print(data)
    decisionTreeHelper(data)
    for i in range(0,2):
        print(i)
    #trainer = indent(header(file) + body() + print_trailer(bestAttribute, bestThreshold))
    #file.write(trainer)
    file.close()


if __name__ == '__main__':
    main()
