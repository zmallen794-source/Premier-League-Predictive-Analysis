#For our codes, we used Pandas, NumPy, matplotlib.pyplot, mpl_toolkits.mplot3d, scikit-learn, xgboost, torch, joblib

#All data used was based on results.csv, which is every game in the English Premier League from 1993/94 to 2021/22 seasons

#FinalProject_Kmeans_andPCS.py reviewed how unsupervised learning could review each feature and help cluster them into determining if the Home Team Wins, Away Team Wins, or if the outcome was too close to tell/tie occurred.
#This was performed by using Principal Components Analysis, followed by K-Means Clustering

#FinalProject_LogReg.py looks at how well it is to predict the outcome of the games based at differing point in the game, whether it is the information prior to the game, half-time, or stats for the full game. It performs its
#predictions through Logisitic Regression

#XGBoost_NN.ipynb trains a treebased model XGBoost, and then a Neural Network to predict the outcome of the games. The experiment was done in champion challanger metod, where we interated through XGBoost paramters to find the best possible model this was our baseline or champion model. We then created and trained a Neural Netowrk model(challanger) and experimented with is hyperparmeters trying to beat the outcome of the XGBoost model(champion).
#predictions through XGBoost and Neural Network.
