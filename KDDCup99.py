#!/usr/bin/env python

# dataset: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

import sys
from collections import OrderedDict

from pyspark import SparkConf, SparkContext
from numpy import array
from math import sqrt
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.feature import StandardScaler


def parse_interaction(line):
    """
    Parses a network data interaction.
    """
    line_split = line.split(",")
    clean_line_split = [line_split[0]]+line_split[4:-1]
    return (line_split[-1], array([float(x) for x in clean_line_split]))


def distance(a, b):
    """
    Calculates the euclidean distnace between two numeric RDDs
    """
    return sqrt(
        a.zip(b)
        .map(lambda x: (x[0]-x[1]))
        .map(lambda x: x*x)
        .reduce(lambda a,b: a+b)
        )


def dist_to_centroid(datum, clusters):
    """
    Determines the distance of a point to its cluster centroid
    """
    cluster = clusters.predict(datum)
    centroid = clusters.centers[cluster]
    return sqrt(sum([x**2 for x in (centroid - datum)]))


def clustering_score(data, k):
    clusters = KMeans.train(data, k, maxIterations=10, runs=5, initializationMode="random")
    result = (k, clusters, data.map(lambda datum: dist_to_centroid(datum, clusters)).mean())
    print "Clustering score for k=%(k)d is %(score)f" % {"k": k, "score": result[2]}
    return result



if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print "Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
          "KDDCup99.py kddcup.data.file"
        sys.exit(1)

    # set up environment
    max_k = int(sys.argv[1])
    data_file = sys.argv[2]
    conf = SparkConf().setAppName("KDDCup99") \
      #.set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)
    
    # load raw data
    print "Loading RAW data..."
    raw_data = sc.textFile(data_file)

    # count by all different labels and print them decreasingly
    print "Counting all different labels"
    labels = raw_data.map(lambda line: line.strip().split(",")[-1])
    label_counts = labels.countByValue()
    sorted_labels = OrderedDict(sorted(label_counts.items(), key=lambda t: t[1], reverse=True))
    for label, count in sorted_labels.items():
        print label, count

    # Prepare data for clustering input
    # the data contains non-numeric features, we want to exclude them since
    # k-means works with numeric features. These are the first three and the last
    # column in each data row
    print "Parsing dataset..."
    parsed_data = raw_data.map(parse_interaction)
    parsed_data_values = parsed_data.values().cache()

    # Standardize data
    print "Standardizing data..."
    standardizer = StandardScaler(True, True)
    standardizer_model = standardizer.fit(parsed_data_values)
    standardized_data_values = standardizer_model.transform(parsed_data_values)

    # Evaluate values of k from 5 to 40
    print "Calculating total in within cluster distance for different k values (10 to %(max_k)d):" % {"max_k": max_k}
    scores = map(lambda k: clustering_score(standardized_data_values, k), range(10,max_k+1,10))

    # Obtain min score k
    min_k = min(scores, key=lambda x: x[2])[0]
    print "Best k value is %(best_k)d" % {"best_k": min_k}

    # Use the best model to assign a cluster to each datum
    # We use here standardized data - it is more appropriate for exploratory purposes
    print "Obtaining clustering result sample for k=%(min_k)d..." % {"min_k": min_k}
    best_model = min(scores, key=lambda x: x[2])[1]
    cluster_assignments_sample = standardized_data_values.map(lambda datum: str(best_model.predict(datum))+","+",".join(map(str,datum))).sample(False,0.05)

    # Save assignment sample to file
    print "Saving sample to file..."
    cluster_assignments_sample.saveAsTextFile("sample_standardized")
    
    print "DONE!"
