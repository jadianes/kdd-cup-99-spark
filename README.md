
# KDD Cup 99 - PySpark

This is my try with the *KDD Cup of 1999* using Python, Scikit-learn, and Spark.
The dataset for this data mining competition can be found
[here](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html).

## Task description summary

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
the training data. This makes the task more realistic. The datasets contain a
total of 24 training attack types, with an additional 14 types in the test data
only.

Some intrusion experts believe that most novel attacks are variants of known
attacks and the "signature" of known attacks can be sufficient to catch novel
variants. Based on this idea, we will experiment with different machine learning
approaches.

## Approach

We will start by working on a reduced dataset (the 10 percent dataset provided).

There we will do some exploratory data analysis using `Pandas`. Then we will
build a classifier using `Scikit-learn`. Our classifier will just classify
entries into `normal` or `attack`. By doing so, we can generalise the model to
new attack types.

However, in our final approach we want to use clustering and anomality
detection. We want our model to be able to work well with unknown attack types
and also to give an approchimation of the closest attack type. Initially we will
do clustering using `Scikit-learn` again and see if we can beat our previous
classifier.

Finally, we will use `Spark` to implement the clustering approach on the
complete dataset containing around 5 million interactions.

## Loading the data


    import pandas
    from time import time
    col_names = ["duration","protocol_type","service","flag","src_bytes",
        "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
    kdd_data_10percent = pandas.read_csv("/nfs/data/KDD99/kddcup.data_10_percent", header=None, names = col_names)
    kdd_data_10percent.describe()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration</th>
      <th>src_bytes</th>
      <th>dst_bytes</th>
      <th>land</th>
      <th>wrong_fragment</th>
      <th>urgent</th>
      <th>hot</th>
      <th>num_failed_logins</th>
      <th>logged_in</th>
      <th>num_compromised</th>
      <th>...</th>
      <th>dst_host_count</th>
      <th>dst_host_srv_count</th>
      <th>dst_host_same_srv_rate</th>
      <th>dst_host_diff_srv_rate</th>
      <th>dst_host_same_src_port_rate</th>
      <th>dst_host_srv_diff_host_rate</th>
      <th>dst_host_serror_rate</th>
      <th>dst_host_srv_serror_rate</th>
      <th>dst_host_rerror_rate</th>
      <th>dst_host_srv_rerror_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td> 494021.000000</td>
      <td> 4.940210e+05</td>
      <td>  494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td>...</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>     47.979302</td>
      <td> 3.025610e+03</td>
      <td>     868.532425</td>
      <td>      0.000045</td>
      <td>      0.006433</td>
      <td>      0.000014</td>
      <td>      0.034519</td>
      <td>      0.000152</td>
      <td>      0.148247</td>
      <td>      0.010212</td>
      <td>...</td>
      <td>    232.470778</td>
      <td>    188.665670</td>
      <td>      0.753780</td>
      <td>      0.030906</td>
      <td>      0.601935</td>
      <td>      0.006684</td>
      <td>      0.176754</td>
      <td>      0.176443</td>
      <td>      0.058118</td>
      <td>      0.057412</td>
    </tr>
    <tr>
      <th>std</th>
      <td>    707.746472</td>
      <td> 9.882181e+05</td>
      <td>   33040.001252</td>
      <td>      0.006673</td>
      <td>      0.134805</td>
      <td>      0.005510</td>
      <td>      0.782103</td>
      <td>      0.015520</td>
      <td>      0.355345</td>
      <td>      1.798326</td>
      <td>...</td>
      <td>     64.745380</td>
      <td>    106.040437</td>
      <td>      0.410781</td>
      <td>      0.109259</td>
      <td>      0.481309</td>
      <td>      0.042133</td>
      <td>      0.380593</td>
      <td>      0.380919</td>
      <td>      0.230590</td>
      <td>      0.230140</td>
    </tr>
    <tr>
      <th>min</th>
      <td>      0.000000</td>
      <td> 0.000000e+00</td>
      <td>       0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>...</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>      0.000000</td>
      <td> 4.500000e+01</td>
      <td>       0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>...</td>
      <td>    255.000000</td>
      <td>     46.000000</td>
      <td>      0.410000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>      0.000000</td>
      <td> 5.200000e+02</td>
      <td>       0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>...</td>
      <td>    255.000000</td>
      <td>    255.000000</td>
      <td>      1.000000</td>
      <td>      0.000000</td>
      <td>      1.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>      0.000000</td>
      <td> 1.032000e+03</td>
      <td>       0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>...</td>
      <td>    255.000000</td>
      <td>    255.000000</td>
      <td>      1.000000</td>
      <td>      0.040000</td>
      <td>      1.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>  58329.000000</td>
      <td> 6.933756e+08</td>
      <td> 5155468.000000</td>
      <td>      1.000000</td>
      <td>      3.000000</td>
      <td>      3.000000</td>
      <td>     30.000000</td>
      <td>      5.000000</td>
      <td>      1.000000</td>
      <td>    884.000000</td>
      <td>...</td>
      <td>    255.000000</td>
      <td>    255.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 38 columns</p>
</div>



Now we have our data loaded into a `Pandas` data frame. In order to get familiar
with our data, let's have a look at how the labels are distributed.


    kdd_data_10percent['label'].value_counts()




    smurf.              280790
    neptune.            107201
    normal.              97278
    back.                 2203
    satan.                1589
    ipsweep.              1247
    portsweep.            1040
    warezclient.          1020
    teardrop.              979
    pod.                   264
    nmap.                  231
    guess_passwd.           53
    buffer_overflow.        30
    land.                   21
    warezmaster.            20
    imap.                   12
    rootkit.                10
    loadmodule.              9
    ftp_write.               8
    multihop.                7
    phf.                     4
    perl.                    3
    spy.                     2
    dtype: int64



## Feature selection

Initially, we will use all features. We need to do something with our
categorical variables. For now, we will not include them in the training
features.


    num_features = [
        "duration","src_bytes",
        "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate"
    ]
    features = kdd_data_10percent[num_features].astype(float)
    features.describe()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration</th>
      <th>src_bytes</th>
      <th>dst_bytes</th>
      <th>land</th>
      <th>wrong_fragment</th>
      <th>urgent</th>
      <th>hot</th>
      <th>num_failed_logins</th>
      <th>logged_in</th>
      <th>num_compromised</th>
      <th>...</th>
      <th>dst_host_count</th>
      <th>dst_host_srv_count</th>
      <th>dst_host_same_srv_rate</th>
      <th>dst_host_diff_srv_rate</th>
      <th>dst_host_same_src_port_rate</th>
      <th>dst_host_srv_diff_host_rate</th>
      <th>dst_host_serror_rate</th>
      <th>dst_host_srv_serror_rate</th>
      <th>dst_host_rerror_rate</th>
      <th>dst_host_srv_rerror_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td> 494021.000000</td>
      <td> 4.940210e+05</td>
      <td>  494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td>...</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>     47.979302</td>
      <td> 3.025610e+03</td>
      <td>     868.532425</td>
      <td>      0.000045</td>
      <td>      0.006433</td>
      <td>      0.000014</td>
      <td>      0.034519</td>
      <td>      0.000152</td>
      <td>      0.148247</td>
      <td>      0.010212</td>
      <td>...</td>
      <td>    232.470778</td>
      <td>    188.665670</td>
      <td>      0.753780</td>
      <td>      0.030906</td>
      <td>      0.601935</td>
      <td>      0.006684</td>
      <td>      0.176754</td>
      <td>      0.176443</td>
      <td>      0.058118</td>
      <td>      0.057412</td>
    </tr>
    <tr>
      <th>std</th>
      <td>    707.746472</td>
      <td> 9.882181e+05</td>
      <td>   33040.001252</td>
      <td>      0.006673</td>
      <td>      0.134805</td>
      <td>      0.005510</td>
      <td>      0.782103</td>
      <td>      0.015520</td>
      <td>      0.355345</td>
      <td>      1.798326</td>
      <td>...</td>
      <td>     64.745380</td>
      <td>    106.040437</td>
      <td>      0.410781</td>
      <td>      0.109259</td>
      <td>      0.481309</td>
      <td>      0.042133</td>
      <td>      0.380593</td>
      <td>      0.380919</td>
      <td>      0.230590</td>
      <td>      0.230140</td>
    </tr>
    <tr>
      <th>min</th>
      <td>      0.000000</td>
      <td> 0.000000e+00</td>
      <td>       0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>...</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>      0.000000</td>
      <td> 4.500000e+01</td>
      <td>       0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>...</td>
      <td>    255.000000</td>
      <td>     46.000000</td>
      <td>      0.410000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>      0.000000</td>
      <td> 5.200000e+02</td>
      <td>       0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>...</td>
      <td>    255.000000</td>
      <td>    255.000000</td>
      <td>      1.000000</td>
      <td>      0.000000</td>
      <td>      1.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>      0.000000</td>
      <td> 1.032000e+03</td>
      <td>       0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>...</td>
      <td>    255.000000</td>
      <td>    255.000000</td>
      <td>      1.000000</td>
      <td>      0.040000</td>
      <td>      1.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>  58329.000000</td>
      <td> 6.933756e+08</td>
      <td> 5155468.000000</td>
      <td>      1.000000</td>
      <td>      3.000000</td>
      <td>      3.000000</td>
      <td>     30.000000</td>
      <td>      5.000000</td>
      <td>      1.000000</td>
      <td>    884.000000</td>
      <td>...</td>
      <td>    255.000000</td>
      <td>    255.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 38 columns</p>
</div>



As we mentioned, we are going to reduce the outputs to `normal` and `attack`.


    from sklearn.neighbors import KNeighborsClassifier
    labels = kdd_data_10percent['label'].copy()
    labels[labels!='normal.'] = 'attack.'
    labels.value_counts()




    attack.    396743
    normal.     97278
    dtype: int64



## Feature scaling

We are going to use a lot of distance-based methods here. In order to avoid some
features distances dominate others, we need to scale all of them.


    from sklearn.preprocessing import MinMaxScaler
    features.apply(lambda x: MinMaxScaler().fit_transform(x))
    features.describe()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration</th>
      <th>src_bytes</th>
      <th>dst_bytes</th>
      <th>land</th>
      <th>wrong_fragment</th>
      <th>urgent</th>
      <th>hot</th>
      <th>num_failed_logins</th>
      <th>logged_in</th>
      <th>num_compromised</th>
      <th>...</th>
      <th>dst_host_count</th>
      <th>dst_host_srv_count</th>
      <th>dst_host_same_srv_rate</th>
      <th>dst_host_diff_srv_rate</th>
      <th>dst_host_same_src_port_rate</th>
      <th>dst_host_srv_diff_host_rate</th>
      <th>dst_host_serror_rate</th>
      <th>dst_host_srv_serror_rate</th>
      <th>dst_host_rerror_rate</th>
      <th>dst_host_srv_rerror_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td> 494021.000000</td>
      <td> 4.940210e+05</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td>...</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
      <td> 494021.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>      0.000823</td>
      <td> 4.363595e-06</td>
      <td>      0.000168</td>
      <td>      0.000045</td>
      <td>      0.002144</td>
      <td>      0.000005</td>
      <td>      0.001151</td>
      <td>      0.000030</td>
      <td>      0.148247</td>
      <td>      0.000012</td>
      <td>...</td>
      <td>      0.911650</td>
      <td>      0.739865</td>
      <td>      0.753780</td>
      <td>      0.030906</td>
      <td>      0.601935</td>
      <td>      0.006684</td>
      <td>      0.176754</td>
      <td>      0.176443</td>
      <td>      0.058118</td>
      <td>      0.057412</td>
    </tr>
    <tr>
      <th>std</th>
      <td>      0.012134</td>
      <td> 1.425228e-03</td>
      <td>      0.006409</td>
      <td>      0.006673</td>
      <td>      0.044935</td>
      <td>      0.001837</td>
      <td>      0.026070</td>
      <td>      0.003104</td>
      <td>      0.355345</td>
      <td>      0.002034</td>
      <td>...</td>
      <td>      0.253903</td>
      <td>      0.415845</td>
      <td>      0.410781</td>
      <td>      0.109259</td>
      <td>      0.481309</td>
      <td>      0.042133</td>
      <td>      0.380593</td>
      <td>      0.380919</td>
      <td>      0.230590</td>
      <td>      0.230140</td>
    </tr>
    <tr>
      <th>min</th>
      <td>      0.000000</td>
      <td> 0.000000e+00</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>...</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>      0.000000</td>
      <td> 6.489989e-08</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>...</td>
      <td>      1.000000</td>
      <td>      0.180392</td>
      <td>      0.410000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>      0.000000</td>
      <td> 7.499542e-07</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>...</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      0.000000</td>
      <td>      1.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>      0.000000</td>
      <td> 1.488371e-06</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>...</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      0.040000</td>
      <td>      1.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
      <td>      0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>      1.000000</td>
      <td> 1.000000e+00</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>...</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
      <td>      1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 38 columns</p>
</div>



## Visualising data using Principal Components

By using Principal Component Analysis, we can reduce the dimensionality of our
data and plot it into a two-dimensional space. The PCA will capture those
dimensions with the maximum variance, reducing the information loss.


    # TODO

## Training a classifier

Following the idea that new attack types will be similar to known types, let's
start by trying a k-nearest neighbours classifier. We must to avoid brute force
comparisons in the Nxd space at all costs. Being N the number of samples in our
data more than 400K, and d the number of features 38 features, we will end up
with an unfeasible modeling process. For this reason we pass `algorithm =
'ball_tree'`. For more on this, se [here](http://scikit-
learn.org/stable/modules/neighbors.html#choice-of-nearest-neighbors-algorithm).


    clf = KNeighborsClassifier(n_neighbors = 5, algorithm = 'ball_tree', leaf_size=500)
    t0 = time()
    clf.fit(features,labels)
    tt = time()-t0
    print "Classifier trained in {} seconds".format(round(tt,3))

Now let's try the classifier with the testing data. First we need to load the
labeled test data. We wil also sample 10 percent of the entries. For that. we
will take advantage of the `train_test_split` function in `sklearn`.


    kdd_data_corrected = pandas.read_csv("/nfs/data/KDD99/corrected", header=None, names = col_names)
    kdd_data_corrected['label'].value_counts()




    smurf.              164091
    normal.              60593
    neptune.             58001
    snmpgetattack.        7741
    mailbomb.             5000
    guess_passwd.         4367
    snmpguess.            2406
    satan.                1633
    warezmaster.          1602
    back.                 1098
    mscan.                1053
    apache2.               794
    processtable.          759
    saint.                 736
    portsweep.             354
    ipsweep.               306
    httptunnel.            158
    pod.                    87
    nmap.                   84
    buffer_overflow.        22
    multihop.               18
    named.                  17
    sendmail.               17
    ps.                     16
    xterm.                  13
    rootkit.                13
    teardrop.               12
    xlock.                   9
    land.                    9
    xsnoop.                  4
    ftp_write.               3
    sqlattack.               2
    loadmodule.              2
    worm.                    2
    perl.                    2
    phf.                     2
    udpstorm.                2
    imap.                    1
    dtype: int64



We can see that we have new attack labels. In any case, we will convert all of
the to the `attack.` label.


    kdd_data_corrected['label'][kdd_data_corrected['label']!='normal.'] = 'attack.'
    kdd_data_corrected['label'].value_counts()




    attack.    250436
    normal.     60593
    dtype: int64



Again we select features and scale.


    from sklearn.cross_validation import train_test_split
    kdd_data_corrected[num_features] = kdd_data_corrected[num_features].astype(float)
    kdd_data_corrected[num_features].apply(lambda x: MinMaxScaler().fit_transform(x))

Now we can sample the 10 percent of the test data (after we scale it). Although
we also get training data, we don't need it in this case.


    features_train, features_test, labels_train, labels_test = train_test_split(
        kdd_data_corrected[num_features], 
        kdd_data_corrected['label'], 
        test_size=0.1, 
        random_state=42)

Now, do predictions using our classifier and the test data. kNN classifiers are
slow compared to other methods due to all the comparisons required in order to
make predictions.


    t0 = time()
    pred = clf.predict(features_test)
    tt = time() - t0
    print "Predicted in {} seconds".format(round(tt,3))

    Predicted in 282.116 seconds


That took a lot of time. Actually, the more training data we use with a k-means
classifier, the slower it gets to predict. It needs to compare the new data with
all the points. Definitively we want some centroid-based classifier if we plan
to use it in real-time detection.

And finally, calculate the R squared value using the test labels.


    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    print "R squared is {}.".format(round(acc,4))

    R squared is 0.801.


## Clustering

So finally, let's try our anomaly detection approach in the reduced dataset. We
will start by doing **k-means clustering**. Once we have the cluster centers, we
will use them to determine the labels of the test data (unlabeled).

Based on the assumption that new attack types will resemble old type, we will be
able to detect those. Moreover, anything that falls too far from any cluster,
will be considered anomalous and therefore a possible attack.

### KMeans clustering


    from sklearn.cluster import KMeans
    k = 30
    km = KMeans(n_clusters = k)


    t0 = time()
    km.fit(features)
    tt = time()-t0
    print "Clustered in {} seconds".format(round(tt,3))

    Clustered in 315.355 seconds


Check cluster sizes.


    pandas.Series(km.labels_).value_counts()




    0     280476
    1      48557
    11     38322
    4      23822
    13     19492
    7      11466
    3      10467
    29     10083
    26      9320
    19      5030
    5       4277
    2       3933
    9       3577
    17      3341
    8       2923
    25      2583
    22      2407
    20      1886
    10      1728
    21      1375
    18      1297
    12      1221
    23      1184
    15       995
    6        978
    16       970
    14       874
    24       680
    27       581
    28       176
    dtype: int64



Get labels for each cluster. Here, we go back to use the complete set of labels.


    labels = kdd_data_10percent['label']
    label_names = map(
        lambda x: pandas.Series([labels[i] for i in range(len(km.labels_)) if km.labels_[i]==x]), 
        range(k))

Print labels for each cluster.


    for i in range(k):
        print "Cluster {} labels:".format(i)
        print label_names[i].value_counts()

    Cluster 0 labels:
    smurf.     280455
    normal.        21
    dtype: int64
    Cluster 1 labels:
    neptune.      48553
    portsweep.        4
    dtype: int64
    Cluster 2 labels:
    normal.             3453
    back.                460
    buffer_overflow.      14
    loadmodule.            2
    ftp_write.             2
    guess_passwd.          1
    warezclient.           1
    dtype: int64
    Cluster 3 labels:
    neptune.      10397
    portsweep.       54
    satan.           13
    normal.           3
    dtype: int64
    Cluster 4 labels:
    normal.    22570
    back.       1248
    phf.           3
    satan.         1
    dtype: int64
    Cluster 5 labels:
    normal.       4243
    satan.          25
    portsweep.       9
    dtype: int64
    Cluster 6 labels:
    normal.     975
    ipsweep.      3
    dtype: int64
    Cluster 7 labels:
    normal.    11466
    dtype: int64
    Cluster 8 labels:
    normal.    2923
    dtype: int64
    Cluster 9 labels:
    normal.         3562
    back.             14
    warezclient.       1
    dtype: int64
    Cluster 10 labels:
    normal.          1347
    ipsweep.          255
    pod.               65
    warezmaster.       18
    nmap.              17
    imap.              10
    smurf.              3
    land.               3
    ftp_write.          2
    rootkit.            2
    multihop.           2
    neptune.            1
    loadmodule.         1
    guess_passwd.       1
    portsweep.          1
    dtype: int64
    Cluster 11 labels:
    neptune.         38189
    nmap.              103
    portsweep.          19
    normal.              7
    land.                2
    guess_passwd.        1
    imap.                1
    dtype: int64
    Cluster 12 labels:
    satan.        1219
    portsweep.       2
    dtype: int64
    Cluster 13 labels:
    normal.    19492
    dtype: int64
    Cluster 14 labels:
    normal.          803
    guess_passwd.     50
    portsweep.        10
    back.              8
    ipsweep.           2
    neptune.           1
    dtype: int64
    Cluster 15 labels:
    portsweep.    913
    ipsweep.       81
    normal.         1
    dtype: int64
    Cluster 16 labels:
    teardrop.    970
    dtype: int64
    Cluster 17 labels:
    normal.      3172
    pod.           83
    smurf.         39
    satan.         23
    teardrop.       8
    ipsweep.        6
    nmap.           4
    rootkit.        3
    land.           1
    spy.            1
    neptune.        1
    dtype: int64
    Cluster 18 labels:
    ipsweep.     898
    normal.      169
    pod.         116
    nmap.         99
    land.         14
    multihop.      1
    dtype: int64
    Cluster 19 labels:
    normal.         4962
    warezclient.      52
    rootkit.           4
    satan.             4
    perl.              3
    ipsweep.           2
    loadmodule.        1
    spy.               1
    imap.              1
    dtype: int64
    Cluster 20 labels:
    normal.             1195
    warezclient.         660
    buffer_overflow.      16
    loadmodule.            5
    ftp_write.             4
    back.                  3
    multihop.              2
    rootkit.               1
    dtype: int64
    Cluster 21 labels:
    normal.       1262
    satan.         111
    teardrop.        1
    portsweep.       1
    dtype: int64
    Cluster 22 labels:
    normal.    2407
    dtype: int64
    Cluster 23 labels:
    normal.    1184
    dtype: int64
    Cluster 24 labels:
    normal.         370
    warezclient.    306
    multihop.         2
    warezmaster.      2
    dtype: int64
    Cluster 25 labels:
    normal.    2187
    back.       395
    phf.          1
    dtype: int64
    Cluster 26 labels:
    normal.    9245
    back.        75
    dtype: int64
    Cluster 27 labels:
    smurf.     293
    normal.    259
    satan.      21
    nmap.        8
    dtype: int64
    Cluster 28 labels:
    satan.        172
    portsweep.      3
    land.           1
    dtype: int64
    Cluster 29 labels:
    neptune.      10059
    portsweep.       24
    dtype: int64


We can see how, in most of them, there is a dominant label. It would be
interesting to go cluster by cluster and analyise mayority labels, or how labels
are split between different clusters (some with more dominance than others). All
that would help us understand each type of attack! This is also a benefit of
using a clustering-based approach.

### Predictions

We can now predict using our test data.


    t0 = time()
    pred = km.predict(kdd_data_corrected[num_features])
    tt = time() - t0
    print "Assigned clusters in {} seconds".format(round(tt,3))

    Assigned clusters in 0.698 seconds


We can see that the assignment process is much faster than the prediction
process with our kNN. But we still need to assign labels.


    # TODO: get mayority label for each cluster assignment (we have labels from the previous step)


    # TODO: check these labels with those in the corrected test data in order to calculate accuracy


    

## Using the complete dataset with Spark

The script [KDDCup99.py](kdd-cup-99-spark/KDDCup99.py) runds thorugh a series of steps to perform
k-means clustering over the complete dataset using `PySpark`.

The clustering results are stored in a `CSV` file. This file is very convenient
for visualisation purposes. It would be very hard to cluster and visualise
results of the complete dataset using `Scikit-learn`.

The following chart depicts the **first two pincipal components** for the
clustering results.

![](https://raw.githubusercontent.com/jadianes/kdd-cup-99-spark/master/clusters.png)

Remember that we have up to 24 different labels in our complete dataset. However
we have generated up to 80 different clusters. As a result of this, some of the
clusters appear very close in the first principal component. This is due to the
variability of interactions for a given type of attack (or label).


    # TODO: follow the same approach for label assignment in the test data as before


    
