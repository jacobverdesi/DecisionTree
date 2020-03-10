import pandas as pd
"""File:HW05_Verdesi_Jacob_Trainer.py
    Author: Jacob Verdesi
    Email:jxv3386@rit.edu
    Description:This is a Trained program for Classifying Abominable Data"""
def printClassified(data):
    right=0
    data["Prediction"]=""
    for index, row in data.iterrows():
        if row["BangLn"] <= 5:
            if row["Age"] <= 30:
                data.loc[index,"Prediction"]=-1
                right+=(row["Class"]==-1)
            else:
                if row["EarLobes"] <= 0:
                    data.loc[index,"Prediction"]=1
                    right+=(row["Class"]==1)
                else:
                    data.loc[index,"Prediction"]=1
                    right+=(row["Class"]==1)
                
            
        else:
            if row["Age"] <= 48:
                if row["Ht"] <= 125:
                    data.loc[index,"Prediction"]=1
                    right+=(row["Class"]==1)
                else:
                    data.loc[index,"Prediction"]=-1
                    right+=(row["Class"]==-1)
                
            else:
                if row["BangLn"] <= 6:
                    data.loc[index,"Prediction"]=1
                    right+=(row["Class"]==1)
                else:
                    data.loc[index,"Prediction"]=-1
                    right+=(row["Class"]==-1)
                
    tp=data[(data["Class"]==1) & (data["Prediction"]==1)].count()["Class"]
    tn=data[(data["Class"]==-1) & (data["Prediction"]==-1)].count()["Class"]
    fp=data[(data["Class"]==-1) & (data["Prediction"]==1)].count()["Class"]
    fn=data[(data["Class"]==1) & (data["Prediction"]==-1)].count()["Class"]
    print(tp,tn,fp,fn,(tp+tn)/data.shape[0])
def main():
    """Main function"""
    fileName="VALIDATION_DATA_TO_RELEASE.csv"
    data=(pd.read_csv(fileName,sep=','))
    printClassified(data)
    
if __name__ == '__main__':
    main()
    