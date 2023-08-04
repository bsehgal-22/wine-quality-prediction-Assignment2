#import findspark
#findspark.init()
#findspark.find()

#LOAD LIBRARIES
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
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


##### START SPARK SESSION #####
conf = pyspark.SparkConf().setAppName('winequality').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

#path  = sys.argv[1]

##### LOAD VALIDATION DATASET AND PRINT #####
val = spark.read.format("csv").load("file:///home/hadoop/ValidationDataset.csv", header = True , sep=";")
val.printSchema()
val.show()


##### CHANGE COLUMN NAME --> 'quality' TO 'label' #####
for col_name in val.columns[1:-1]+['""""quality"""""']:
    val = val.withColumn(col_name, col(col_name).cast('float'))
val = val.withColumnRenamed('""""quality"""""', "label")

##### GET THE FEATURES AND LABELS SEPERATE AND CONVERT TO NUMPY ARRAY ##### 
features =np.array(val.select(val.columns[1:-1]).collect())
label = np.array(val.select('label').collect())

##### CREATE FEATURE VECTOR ######
VectorAssembler = VectorAssembler(inputCols = val.columns[1:-1] , outputCol = 'features')
df_tr = VectorAssembler.transform(val)
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

##### LOAD THE S3 MODEL ##### 
#RFModel = RandomForestModel.load(sc, "/winepredict/trainingmodel.model/") 
#RFModel = RandomForestModel.load(sc, 's3://mybucket-assignment2') 
RFModel = RandomForestModel.load(sc, "/my-bucket-assignment2/trainingmodel.model/")


print("model loaded successfully")
predictions = RFModel.predict(dataset.map(lambda x: x.features))

##### GET A RDD OF LABLE AND PREDICTIONS ##### 
labelsAndPredictions = dataset.map(lambda lp: lp.label).zip(predictions)
 
labelsAndPredictions_df = labelsAndPredictions.toDF()

##### CONVERTING RDD - SPARK DATAFRAME -- PANDAS DATAFRAME #####  
labelpred = labelsAndPredictions.toDF(["label", "Prediction"])
labelpred.show()
labelpred_df = labelpred.toPandas()

##### CALCULATE F1SCORE #####
F1score = f1_score(labelpred_df['label'], labelpred_df['Prediction'], average='micro')
print("F1- score: ", F1score)
print(confusion_matrix(labelpred_df['label'],labelpred_df['Prediction']))
print(classification_report(labelpred_df['label'],labelpred_df['Prediction']))
print("Accuracy" , accuracy_score(labelpred_df['label'], labelpred_df['Prediction']))

##### CALCULATE TEST ERRORS #####
testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(dataset.count())    
print('Test Error = ' + str(testErr))
