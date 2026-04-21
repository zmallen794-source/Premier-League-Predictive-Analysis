import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, confusion_matrix

#load data
df = pd.read_csv("results.csv", encoding="latin1")
#convert target variable to int for convenience
df["FTR"] = df["FTR"].replace({"H": 0, "D": 1, "A":2}).astype("Int64")
#clean data, attempt to convert String data to integers
df = df.dropna()
df["HTR"] = df["HTR"].replace({"H": 0, "D": 1, "A":2}).astype("Int64")
years = {}
for date in df["Season"]:
    if date not in years:
        years[date]=[int(date[0:4])]
df["Season"] = df["Season"].replace(years).astype("Int64")
teams={}
i=0
for team in df["HomeTeam"]:
    if team not in teams:
        teams[team]=i
        i+=1
df["HomeTeam"] = df["HomeTeam"].replace(teams).astype("Int64")
df["AwayTeam"] = df["AwayTeam"].replace(teams).astype("Int64")
refs = {}
i=0
for ref in df["Referee"]:
    if ref not in refs:
        refs[ref]=i
        i+=1
df["Referee"] = df["Referee"].replace(refs).astype("Int64")
df['Month'] = df['DateTime'].astype(str).str[5:7].copy()
df['Month'] = df['Month'].astype("Int64")
#set X and y variables
y = df["FTR"].copy()
df=df.drop(columns=["FTR","FTHG","FTAG"])
df_pre = df[["HomeTeam","AwayTeam","Referee", "Season", "Month"]].copy()
df_half = df[["HTHG","HTAG","HTR"]].copy()
df_post = df[["HS","AS","HST","AST","HC","AC","HF","AF","HY","AY","AR","HR"]].copy()
frames = [df_pre,df_half,df_post]
for dataframe in frames:
    max_1 = 0
    best1_1 = "none"
    for column in dataframe:
        X = dataframe[[column]]
        #set test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        #scale test and training data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        #logreg
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        #get and compare results
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred,average="micro")
        if precision > max_1:
            max_1 = precision
            best1_1 = column
    if dataframe.equals(df_pre):
        print("The highest precision for pre-match is "+str(max_1)+" using variable "+best1_1+".")
    elif dataframe.equals(df_half):
        print("The highest precision for half time is "+str(max_1)+" using variable "+best1_1+".")
    else:
        print("The highest precision for post-match is "+str(max_1)+" using variable "+best1_1+".")
print(" ")

for dataframe in frames:
    max_2 = 0
    best1_2 = "none"
    best2_2 = "none"
    for column in dataframe:
        for column2 in dataframe:
            if column!=column2:
                X = dataframe[[column,column2]]
                #set test data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                #scale test and training data
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                #logreg
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                #get and compare results
                y_pred = model.predict(X_test)
                precision = precision_score(y_test, y_pred,average="micro")
                if precision > max_2:
                    max_2 = precision
                    best1_2 = column
                    best2_2 = column2
    if dataframe.equals(df_pre):
        print("The highest precision for pre-match is "+str(max_2)+" using variables "+best1_2+" and "+best2_2+".")
    elif dataframe.equals(df_half):
        print("The highest precision for half time is "+str(max_2)+" using variables "+best1_2+" and "+best2_2+".")
    else:
        print("The highest precision for post-match is "+str(max_2)+" using variables "+best1_2+" and "+best2_2+".")
print(" ")

for dataframe in frames:
    max_3 = 0
    best1_3 = "none"
    best2_3 = "none"
    best3_3 = "none"
    for column in dataframe:
        for column2 in dataframe:
            if column!=column2:
                for column3 in dataframe:
                    if column3 not in [column,column2]:
                        X = dataframe[[column,column2,column3]]
                        #set test data
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                        #scale test and training data
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                        #logreg
                        model = LogisticRegression(max_iter=1000)
                        model.fit(X_train, y_train)
                        #get and compare results
                        y_pred = model.predict(X_test)
                        precision = precision_score(y_test, y_pred,average="micro")
                        if precision > max_3:
                            max_3 = precision
                            best1_3 = column
                            best2_3 = column2
                            best3_3 = column3
    if dataframe.equals(df_pre):
        print("The highest precision for pre-match is "+str(max_3)+" using variables "+best1_3+", "+best2_3+", and "+best3_3+".")
    elif dataframe.equals(df_half):
        print("The highest precision for half time is "+str(max_3)+" using variables "+best1_3+", "+best2_3+", and "+best3_3+".")
    else:
        print("The highest precision for post-match is "+str(max_3)+" using variables "+best1_3+", "+best2_3+", and "+best3_3+".")
print(" ")

X = df.drop(columns="DateTime").copy()
#set test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#scale test and training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#logreg
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
#get and compare results
y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred,average="micro")
cm = confusion_matrix(y_test,y_pred)
print("The precision for all variables is "+str(precision)+".")
print(cm)