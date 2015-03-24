#!/usr/bin/env python

# dataset: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

import sys
from collections import OrderedDict

from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import Vector

def parseInteraction(line):
    """
    Parses a network data interaction.
    """
    fields = line.strip().split(",")
    label = fields[-1]
    
    return fields[-1]

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






