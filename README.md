
# KDD Cup 99 - PySpark

This is my try with the *KDD Cup of 1999* using Python, Scikit-learn, and Spark.
The dataset for this data mining competition can be found
[here](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html).

## Taks description summary

You can find the complete description of the task
[here](http://kdd.ics.uci.edu/databases/kddcup99/task.html).

Software to detect network intrusions protects a computer network from
unauthorized users, including perhaps insiders.  The intrusion detector learning
task is to build a predictive model (i.e. a classifier) capable of
distinguishing between *bad connections*, called intrusions or attacks, and
*good normal connections*.

A connection is a sequence of TCP packets starting and ending at some well
defined times, between which data flows to and from a source IP address to a
target IP address under some well defined protocol.  Each connection is labeled
as either normal, or as an attack, with exactly one specific attack type.  Each
connection record consists of about 100 bytes.

Attacks fall into four main categories:

- DOS: denial-of-service, e.g. syn flood;
- R2L: unauthorized access from a remote machine, e.g. guessing password;
- U2R:  unauthorized access to local superuser (root) privileges, e.g., various
``buffer overflow'' attacks;
- probing: surveillance and other probing, e.g., port scanning.

It is important to note that the test data is not from the same probability
distribution as the training data, and it includes specific attack types not in
the training data. This makes the task more realistic.  Some intrusion experts
believe that most novel attacks are variants of known attacks and the
"signature" of known attacks can be sufficient to catch novel variants.

The datasets contain a total of 24 training attack types, with an additional 14
types in the test data only.

## Approach

We will start by working on a reduced dataset (the 10 percent dataset provided).

There we will do some exploratory data analysis and build a classifier.

However, our final approach will use clustering and anomality detection. We want
our model to be able to work well with unknown attack types.




    
