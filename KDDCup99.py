#!/usr/bin/env python

# dataset: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

import sys
from collections import OrderedDict

from pyspark import SparkConf, SparkContext
from numpy import array
from math import sqrt
from pyspark.mllib.clustering import KMeans

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
    clusters = KMeans.train(data, k, maxIterations=10, runs=10, initializationMode="random")
    return data.map(lambda datum: dist_to_centroid(datum, clusters)).mean()


if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print "Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
          "KDDCup99.py kddcup.data.file"
        sys.exit(1)

    # set up environment
    conf = SparkConf().setAppName("KDDCup99") \
      #.set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)

    # load raw data
    raw_data = sc.textFile(sys.argv[1])

    # count by all different labels and print them decreasingly
    labels = raw_data.map(lambda line: line.strip().split(",")[-1])
    label_counts = labels.countByValue()
    sorted_labels = OrderedDict(sorted(label_counts.items(), key=lambda t: t[1], reverse=True))
    print "Different labels and their interaction counts: "
    for label, count in sorted_labels.items():
    	print label, count

    # Prepare data for clustering input
    # the data contains non-numeric features, we want to exclude them since
    # k-means works with numeric features. These are the first three and the last
    # column in each data row
    parsed_data = raw_data.map(parse_interaction)

    parsed_data_values = parsed_data.values().cache()

    # Evaluate values of k from 5 to 40
    print "Calculate scores for different k values (5 to 40)"
    scores = map(lambda k: (k, clustering_score(parsed_data_values, k)), range(5,41,5))

    # print scores
    for score in scores:
        print score

    min_k = min(scores, key=lambda x: x[1])[0]
    print "Best k value is ", min_k

