# -*- coding: utf-8 -*-
"""
Created on Thu May 05 11:05:51 2016

@author: noodlefrenzy
"""

import json
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

def load_sentiwordnet(sc):
    lines = sc.textFile('../Data/SentiWordNet_3.0.0_20130122.txt')
    filtered = lines.filter(lambda x: x.find(u'#') != 0)
    parsed = filtered.map(lambda x: x.split('\t')).map(lambda x: ((x[0], x[1]), 
      { 'pos': float(x[2]), 'neg': float(x[3]), 'words': [word.split('#')[0] for word in x[4].split(' ')] }))
    inverted = parsed.flatMap(lambda x: [(word, { 'pos': x[1]['pos'], 'neg': x[1]['neg'] }) for word in x[1]['words']]).collectAsMap()
    # print inverted['scorned']
    # {'neg': 0.25, 'pos': 0.25}
    return inverted

lines = sc.textFile("../Data/sentences/*.json")
parsed = lines.map(json.loads)
parsed.persist()

sentences = parsed.map(lambda x: x['Sentence'])

def scorer(inverted):
    def score(x):
        x['pos'] = 0.0
        x['neg'] = 0.0
        for word in x['words']:
            if word in inverted:
                x['pos'] += inverted[word]['pos']
                x['neg'] += inverted[word]['neg']
        return x
    return score
    
inverted = load_sentiwordnet(sc)
words = sentences.map(lambda x: { 'sentence': x, 'words': x.split() }).map(scorer(inverted))
print words.first()
# {'neg': 0.5, 'pos': 1.25, 'words': [u'rt', u'@theresamechele:', u'benghazi', u'hero', u'tig', …

from pyspark.ml.feature import HashingTF

wordsDF = sqlContext.createDataFrame(words)
pos = wordsDF.withColumn('label', wordsDF.pos)
posTrain, posTest = pos.randomSplit([0.8, 0.2], seed=17)
hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=20)

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrPipeline = Pipeline(stages=[hashingTF, lr])
dt = DecisionTreeRegressor(maxDepth=10, maxBins=50)
dtPipeline = Pipeline(stages=[hashingTF, dt])
rf = RandomForestRegressor(maxDepth=10, maxBins=50, numTrees=50)
rfPipeline = Pipeline(stages=[hashingTF, rf])

posLR = lrPipeline.fit(posTrain)
lrPred = posLR.transform(posTest)
posDT = dtPipeline.fit(posTrain)
dtPred = posDT.transform(posTest)
posRF = rfPipeline.fit(posTrain)
rfPred = posRF.transform(posTest)

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
lr_rmse = evaluator.evaluate(lrPred)
dt_rmse = evaluator.evaluate(dtPred)
rf_rmse = evaluator.evaluate(rfPred)
print("LR RMSE %g, DT RMSE %g, RF RMSE %g" % (lr_rmse, dt_rmse, rf_rmse))

# LR RMSE 0.44829, DT RMSE 0.312846, RF RMSE 0.300322
