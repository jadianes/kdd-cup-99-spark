#!/usr/bin/env python

# dataset: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

import sys
from collections import OrderedDict

from pyspark import SparkConf, SparkContext
from numpy import array
from math import sqrt
from pyspark.mllib.clustering import KMeans

def parseInteraction(line):
    """
    Parses a network data interaction.
    """
    line_split = line.split(",")
    clean_line_split = [line_split[0]]+line_split[4:-1]
    return (line_split[-1], array([float(x) for x in clean_line_split]))

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
    rawData = sc.textFile(sys.argv[1])

    # count by all different labels and print them decreasingly
    users = rawData.map(lambda line: line.strip().split(",")[-1])
    user_counts = users.countByValue()
    sorted_users = OrderedDict(sorted(user_counts.items(), key=lambda t: t[1], reverse=True))
    print "Different users and their interaction counts: "
    for user, count in sorted_users.items():
    	print user, count

    # Prepare data for clustering input
    # the data contains non-numeric features, we want to exclude them since
    # k-means works with numeric features. These are the first three and the last
    # column in each data row
    parsedData = rawData.map(parseInteraction)

    # Build the model (cluster the data)
    clusters = KMeans.train(parsedData.values(), 10, maxIterations=10,
        runs=10, initializationMode="random")

    # Evaluate clustering by computing Within Set Sum of Squared Errors
    def error(point):
        center = clusters.centers[clusters.predict(point)]
        return sqrt(sum([x**2 for x in (point - center)]))

    WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("Within Set Sum of Squared Error = " + str(WSSSE))


    # print centroids
    for center in clusters.centers:
        print center

    # count labels in each cluster
    def label_point(data):
        cluster = clusters.predict(data[1])
        return (cluster, data[0])
    cluster_label_count = parsedData.map( label_point ).countByValue()
    sorted_cluster_label_count = OrderedDict(sorted(cluster_label_count.items(), key=lambda t: t[0], reverse=True))

    # print label counts
    print("Lebel to cluster assignments:")
    for (cluster,count) in sorted_cluster_label_count.items():
        print cluster[0], cluster[1], count

