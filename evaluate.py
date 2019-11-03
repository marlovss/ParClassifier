#!/usr/bin/python
import re,codecs,random,argparse,os
from sklearn import svm
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,mean_squared_error,explained_variance_score,r2_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDRegressor
from imblearn.over_sampling import SMOTE,ADASYN
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from sklearn import tree
import numpy as np

def defineClass(t):
   if t>=4.5:
      return 1
   else:
      return 0



def testRegressors(trainX,trainY,testX,testY): 
#SVR
   result = dict()
   regressor = svm.LinearSVR()
   regressor = regressor.fit(trainX,trainY2)
   predicted = regressor.predict(testX)
   MSE = mean_squared_error(testY, predicted)
   r2Score = regressor.score(testX,testY)
   pearsonScore = pearsonr(testY,predicted)[0] 
   result.update({'svr':{'r2':r2Score,'mse':MSE,'pearson':pearsonScore}})
#SGDR
   regressor = SGDRegressor()
   regressor = regressor.fit(trainX,trainY2)
   predicted = regressor.predict(testX)
   MSE = mean_squared_error(testY, predicted)
   r2Score = regressor.score(testX,testY)
   pearsonScore = pearsonr(testY,predicted)[0] 
   result.update({'sgdr':{'r2':r2Score,'mse':MSE,'pearson':pearsonScore}})
#DT
   regressor = tree.DecisionTreeRegressor()
   regressor = regressor.fit(trainX,trainY2)
   predicted = regressor.predict(testX)
   MSE = mean_squared_error(testY, predicted)
   r2Score = regressor.score(testX,testY)
   pearsonScore = pearsonr(testY,predicted)[0] 
   result.update({'dtr':{'r2':r2Score,'mse':MSE,'pearson':pearsonScore}})
#Random Forest
   regresser = RandomForestRegressor()
   regressor = regressor.fit(trainX,trainY2)
   predicted = regressor.predict(testX)
   MSE = mean_squared_error(testY, predicted)
   r2Score = regressor.score(testX,testY)
   pearsonScore = pearsonr(testY,predicted)[0] 
   result.update({'rfr':{'r2':r2Score,'mse':MSE,'pearson':pearsonScore}})
# resultado
   return result


def testClassifiers(trainX,trainY,testX,testY):
   result = dict()
   
  
   # SVM
   classifier = svm.LinearSVC(max_iter=10000)
   classifier = classifier.fit(trainX,trainY)
   predicted = classifier.predict(testX)
   cfMatrix = confusion_matrix(testY,predicted)
   prec,rec,f1,sup = precision_recall_fscore_support(testY,predicted)
   result.update({'svm':{'cfMatrix':cfMatrix, 'prec':prec,'rec':rec,'f1':f1}})
   
   # Naive Bayes
   classifier = GaussianNB()
   classifier = classifier.fit(trainX,trainY)
   predicted = classifier.predict(testX)
   cfMatrix = confusion_matrix(testY,predicted)
   prec,rec,f1,sup = precision_recall_fscore_support(testY,predicted)
   result.update({'nb':{'cfMatrix':cfMatrix, 'prec':prec,'rec':rec,'f1':f1}})

   # Decision Tree
   classifier = tree.DecisionTreeClassifier()
   classifier = classifier.fit(trainX,trainY)
   predicted = classifier.predict(testX)
   cfMatrix = confusion_matrix(testY,predicted)
   prec,rec,f1,sup = precision_recall_fscore_support(testY,predicted)
   result.update({'dtc':{'cfMatrix':cfMatrix, 'prec':prec,'rec':rec,'f1':f1}})

   # Random Forest

   classifier = RandomForestClassifier()
   classifier = classifier.fit(trainX,trainY)
   predicted = classifier.predict(testX)
   cfMatrix = confusion_matrix(testY,predicted)
   prec,rec,f1,sup = precision_recall_fscore_support(testY,predicted)
   result.update({'rfc':{'cfMatrix':cfMatrix, 'prec':prec,'rec':rec,'f1':f1}})
 
   return result


import pickle


parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, default="data/vectors",
  help="Path to the directory containing the text files.")

parser.add_argument('--train', type=str, default="train",
  help="Path to the directory containing the train data files.")
parser.add_argument('--test', type=str, default="test",
  help="Path to the directory containing the test data files.")
parser.add_argument('--oversample', type=str, default="none",
  help="Path to the directory containing the test data files.")
parser.add_argument('--total', type=bool, default=False,
  help="Whether the classifier should test a dataset with combined representations (memory expensive and low performance).")
parser.add_argument('--sim', type=bool, default=False,
  help="If True evaluates Sentence Similarity Estimation, otherwise evaluates Paraphrase Classification")



FLAGS = parser.parse_args()


#define the methods
reps = set([re.findall("{}\.([^.]+)\.X[^.]*".format(FLAGS.train),name)[0] for name in os.listdir(FLAGS.input) if  re.findall("{}\.([^.]+)\.X[^.]*".format(FLAGS.train),name)])
data = dict([(rep,{}) for rep in reps])
data.update({'total':{}})


# load the experimental data
# train and test must have all the same sentence representation 
# and data format as pickled vectors
for rep in reps:
   data_forms = set([re.findall("{}\.{}\.(X[^.]*)".format(FLAGS.train,rep),name)[0] for name in os.listdir(FLAGS.input) if  re.findall("{}\.{}\.(X[^.]*)".format(FLAGS.train,rep),name)])
   for data_form in data_forms:
      data[rep][data_form]={}
      data[rep][data_form]['trainX']=pickle.load(open(os.path.join(FLAGS.input,FLAGS.train+'.'+rep+'.'+data_form),"rb"))
      data[rep][data_form]['testX']=pickle.load(open(os.path.join(FLAGS.input,FLAGS.test+'.'+rep+'.'+data_form),"rb"))
   
if FLAGS.total:
   data_forms = set([data_form for rep in reps for data_form in data[rep]])
   data['total']={}
   for data_form in data_forms:
     data['total'][data_form]={}
     data['total'][data_form]['trainX']= np.concatenate([data[rep][data_form]['trainX'] for rep in reps if data_form in data[rep]],1)  
     data['total'][data_form]['testX']= np.concatenate([data[rep][data_form]['testX'] for rep in reps if data_form in data[rep]],1)
trainY = pickle.load(open(os.path.join(FLAGS.input,FLAGS.train+'.y'),"rb"))
trainY2 = pickle.load(open(os.path.join(FLAGS.input,FLAGS.train+'.y2'),"rb"))
testY = pickle.load(open(os.path.join(FLAGS.input,FLAGS.test+'.y'),"rb"))
testY2 = pickle.load(open(os.path.join(FLAGS.input,FLAGS.test+'.y2'),"rb"))

for rep in reps:
    for data_form in data[rep]:
         trainX = data[rep][data_form]['trainX'].copy()
         testX = data[rep][data_form]['testX'].copy()
         if FLAGS.sim:
            result = testRegressors(trainX,trainY2,testX,testY2)
            for method in result:
               print(rep,data_form,method,result[method]['r2'],result[method]['mse'],result[method]['pearson'])
         else:
            # random oversampling
            if FLAGS.oversample=="random":
               pos = [trainX[i] for i in range(len(trainX)) if trainY[i]==1]
               neg = [trainX[i] for i in range(len(trainX)) if trainY[i]==0]
               while len(pos)<len(neg):
                  pos.extend(pos)
               pos=random.sample(pos,len(neg))
               trainX = np.concatenate([pos,neg])
               trainY = [1 for _ in range(len(pos))]
               trainY.extend([0 for _ in range(len(neg))])
            elif FLAGS.oversample=="smote":
               trainX,trainY = SMOTE().fit_sample(trainX,trainY)
            elif FLAGS.oversample=="adasyn":
               trainX,trainY = ADASYN().fit_sample(trainX,trainY)
            try:
                result = testClassifiers(trainX,trainY,testX,testY)
                for method in result:
                   print(rep,data_form,method,result[method]['prec'][1],result[method]['rec'][1],result[method]['f1'][1])
            except Exception:
                print("Exception",rep,data_form)
             
