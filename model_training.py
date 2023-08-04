import findspark
findspark.init()
findspark.find()
import warnings

# Load all the required libraries
import pyspark
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession	
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
import numpy as np
import pandas as pd
import boto
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings('ignore')

##### START SPARK #####
conf = pyspark.SparkConf().setAppName('winequality').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

##### LOAD TRAINING DATASET AND PRINT #####
df = spark.read.format("csv").load("file:///home/hadoop/TrainingDataset.csv" , header = True ,sep =";")
df.printSchema()
df.show()

##### CHANGE COLUMN NAME --> 'quality' TO 'label'
for col_name in df.columns[1:-1]+['""""quality"""""']:
    df = df.withColumn(col_name, col(col_name).cast('float'))
df = df.withColumnRenamed('""""quality"""""', "label")

##### GET THE FEATURES AND LABELS SEPERATE AND CONVERT TO NUMPY ARRAY ##### 
features =np.array(df.select(df.columns[1:-1]).collect())
label = np.array(df.select('label').collect())

##### CREATE FEATURE VECTOR ######
VectorAssembler = VectorAssembler(inputCols = df.columns[1:-1] , outputCol = 'features')
df_tr = VectorAssembler.transform(df)
df_tr = df_tr.select(['features','label'])

##### FUNCTION TO CREATE THE LABELPOINT AND PARALLEL TO CONVERT TO RDD #####
def to_labeled_point(sc, features, labels, categorical=False):
    labeled_points = []
    for x, y in zip(features, labels):        
        lp = LabeledPoint(y, x)
        labeled_points.append(lp)
    return sc.parallelize(labeled_points) 

##### RDD CONVERTED DATASET ##### 
dataset = to_labeled_point(sc, features, label)

##### SPLIT DATASET INTO TRAIN AND TEST ##### 
training, test = dataset.randomSplit([0.7, 0.3],seed =11)

##### CREATE A RANDOM FOREST TRAINING CLASSIFIER ##### 
RFmodel = RandomForest.trainClassifier(training, numClasses=10, categoricalFeaturesInfo={},
                                     numTrees=21, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=30, maxBins=32)

##### PREDICTIONS ##### 
predictions = RFmodel.predict(test.map(lambda x: x.features))
#predictionAndLabels = test.map(lambda x: (float(model.predict(x.features)), x.label))

##### GET RDD LABELS AND PREDICTIONS ##### 
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)

labelsAndPredictions_df = labelsAndPredictions.toDF()

##### CONVERTING RDD - SPARK DATAFRAME -- PANDAS DATAFRAME #####
print()
print('===== CONVERTING RDD - SPARK DATAFRAME -- PANDAS DATAFRAME =====') 
labelpred = labelsAndPredictions.toDF(["label", "Prediction"])
labelpred.show()
labelpred_df = labelpred.toPandas()

##### CALCULATE F1SCORE #####
print()
print('===== CALCULATE F1-SCORE =====')
F1score = f1_score(labelpred_df['label'], labelpred_df['Prediction'], average='micro')
print()
print("F1- score: ", F1score)
print()

print('===== CONFUSION MATRIX =====')
print(confusion_matrix(labelpred_df['label'],labelpred_df['Prediction']))
print()

print('===== CLASSIFICATION REPORT =====')
print(classification_report(labelpred_df['label'],labelpred_df['Prediction']))
print()

print('===== ACCURACY SCORE =====')
print("Accuracy ==> " , accuracy_score(labelpred_df['label'], labelpred_df['Prediction']))
print()

##### CALCULATE TEST ERRORS ##### 
print('===== CALCULATE TEST ERRORS =====')
testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test.count())  
print('Test Error ==> ' + str(testErr))
print()

##### Bucket S3 #####
RFmodel.save(sc, 's3://my-bucket-assignment2/trainingmodel.model')


