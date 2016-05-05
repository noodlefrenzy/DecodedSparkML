# -*- coding: utf-8 -*-
"""
Created on Thu May 05 10:26:47 2016

@author: noodlefrenzy
"""
import json
import re

from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

valid_languages = ['en', 'ar']
def is_valid(x):
    slen = lambda a, b: len(a[b]) if b in a else 0
    total_sections = slen(x, 'Keywords') + slen(x, 'Sectors') + slen(x, 'Groups') + slen(x, 'Statuses')
    has_fields = 'Sentence' in x and 'Language' in x and x['Sentence'] != None and x['Language'] != None
    return total_sections > 0 and has_fields and x['Language'] in valid_languages

def normalize_sentence(sentence):
    # http://stackoverflow.com/questions/26568722/remove-unicode-emoji-using-re-in-python
    try:
        # Wide UCS-4 build
        emoji_pattern = re.compile(u'['
            u'\U0001F300-\U0001F64F'
            u'\U0001F680-\U0001F6FF'
            u'\u2600-\u26FF\u2700-\u27BF]+', 
            re.UNICODE)
    except re.error:
        # Narrow UCS-2 build
        emoji_pattern = re.compile(u'('
            u'\ud83c[\udf00-\udfff]|'
            u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
            u'[\u2600-\u26FF\u2700-\u27BF])+', 
            re.UNICODE)
    if not is_valid(sentence):
        return ('invalid', sentence)
    try:
        normalized = emoji_pattern.sub(lambda x: u' %s ' % x.group(0), sentence['Sentence'])
        return ('valid', normalized.replace(u'#',u' # ').lower())
    except:
        return ('error', sentence)

lines = sc.textFile("../Data/sentences/*.json")
parsed = lines.map(json.loads)
parsed.persist()
kws = parsed.flatMap(lambda x: [] if 'Keywords' not in x else [(kw, 1) for kw in x['Keywords']])
histo = kws.reduceByKey(lambda x, y: x + y)
for kv in histo.takeOrdered(10, key=lambda x: -x[1]):
    print 'Keyword "%s": %i' % kv

from pyspark.ml.feature import HashingTF, IDF, Tokenizer

all_sentences = parsed.map(normalize_sentence)
sentences = all_sentences.filter(lambda x: x[0] == 'valid').map(lambda x: x[1])
labeled = sentences.map(lambda x: (1.0 if x.find('hillary') >= 0 and x.find('benghazi') >= 0 else 0.0, x))
sentenceData = sqlContext.createDataFrame(labeled, ['label', 'sentence'])

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=2000)
featurizedData = hashingTF.transform(wordsData)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# Split the data into training, test using the handy randomSplit. 
# It takes a seed if you want to make it repeatable.
trainData, testData = sentenceData.randomSplit([0.8, 0.2])

lr = LogisticRegression(maxIter=10, regParam=0.01, elasticNetParam=0.8)
# Let's create a pipeline that tokenizes, hashes, 
#  rescales using the TF/IDF we learned, and runs a logistic regressor
lrPipeline = Pipeline(stages=[tokenizer, hashingTF, idfModel, lr])
# Fit it to the training data
lrModel = lrPipeline.fit(trainData)
# Now is the time in Scala where we go lrModel.save('wasb:///models/myLR.model')
# No luck (yet) in Python

lrPredictions = lrModel.transform(testData)

from pyspark.mllib.evaluation import BinaryClassificationMetrics

metrics = BinaryClassificationMetrics(lrPredictions.map(lambda x: (x['prediction'], x['label'])))
print("Area under PR = %s" % metrics.areaUnderPR)
print("Area under ROC = %s" % metrics.areaUnderROC)

# Area under PR = 0.888781915483
# Area under ROC = 0.912449199139

def cat(x):
    if x['label'] > 0.5:
        return 'TP' if x['prediction'] > 0.5 else 'FN'
    else:
        return 'TN' if x['prediction'] < 0.5 else 'FP'
posneg = lrPredictions.map(lambda x: (cat(x), x))
posneg.persist()
rates = posneg.countByKey()
precision = rates['TP'] / float(rates['TP'] + rates['FP'])
recall = rates['TP'] / float(rates['TP'] + rates['FN'])
print rates
print 'Precision (%f) and Recall (%f)' % (precision, recall)

falseVals = posneg.flatMap(lambda x: [ (x[0], x[1]['sentence']) ] if x[0] == 'FP' or x[0] == 'FN' else []).distinct()
for fr in falseVals.take(100):
	print fr[0] + ': ' + fr[1]

# FP: news: clinton commits benghazi gaffe, saying us ?didn?t lose a single person? in libya: "liby... https://t.co/ecb2imazjs via @thenewshype
# FP: remember the missing stinger missiles?... remember benghazi? well the pigs are about to fly and the sun is... https://t.co/pcfeujjuko

