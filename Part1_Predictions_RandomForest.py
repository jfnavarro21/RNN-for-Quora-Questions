
# coding: utf-8

# In[17]:


from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType


# In[18]:


from math import log
from sklearn.metrics import log_loss
from random import seed
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline


# In[19]:


spark = SparkSession.builder.appName("Quora1_model").getOrCreate()
spark


# In[20]:


# Project Description
# Read features
trainFeaturesSmallPath = "data/train_features_1000.csv" # small training data
trainFeaturesPath = "data/train_features2.csv"           # full training data
testFeaturesPath = "data/test_features2.csv"             # full test data (for cluster job only)
outPath = "data/predictions.csv"  


# In[ ]:


# REad in train and test
train_df = spark.read.csv(trainFeaturesPath, header=True, inferSchema=True)
test_df = spark.read.csv(testFeaturesPath, header=True, inferSchema=True)


# In[22]:


# Look at train and test data
train_df.printSchema()
test_df.printSchema()


# In[23]:


# Get features' names from the training data
featuresNames=train_df.columns[1:-1]


# In[24]:


# Create features column
assembler=VectorAssembler(inputCols=featuresNames, outputCol='features')
train_df=assembler.transform(train_df)
test_df=assembler.transform(test_df)


# In[25]:


# Remove unnecessary columns
train_df=train_df.select('id','features','is_duplicate')
train_df.show(3)


# In[27]:


# Random Forest model
# Train a Random Forest model :RF_model

rf = RandomForestClassifier(labelCol="is_duplicate", featuresCol="features",                             maxMemoryInMB=16000, minInstancesPerNode=1, maxDepth=25, numTrees=60)
RF_model = rf.fit(train_df)
# Make predictions: RF_predictions
predictions = RF_model.transform(test_df)
#RF_predictions.show(5)
# Select (prediction, true label) and compute AUC of the test
#evaluator = BinaryClassificationEvaluator(labelCol="is_duplicate", metricName='areaUnderROC')
#RF_AUC = evaluator.evaluate(RF_predictions)
#print("Random Forest AUC = %g" % RF_AUC, tree, depth)


# In[ ]:


# def pro of one func
prob_of_one_udf = func.udf(lambda v: float(v[1]), FloatType())


# In[ ]:


outdf = predictions.withColumn('predict', func.round(prob_of_one_udf('probability'),6)).select('id','predict')
outdf.cache()
outdf.show(6)


# In[ ]:


# write csv
outdf.orderBy('id').coalesce(1).write.csv(outPath,header=True,mode='overwrite',quote="")

