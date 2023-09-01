## Setup

Run experiments merging linear quantisation parameters from the root directory.
The functionality can be imported as follows: 
```
>>> from src.linear_merge import *
```

## Quantiles

First of all we show that for 25k random samples there is almost no difference in the
quantiles computed.

We use E5-small (embeddings of quora passages) and Cohere (embeddings of wiki passages)
for our tests. The full data sets are ~500k and ~900k, respectively.

We generate 100 random samples of 25k vectors then plot relative error in the lower
and upper ends of the 99th percentile central confidence. The maximum error is 0.15%
for E5-small and 0.2% for Cohere. The error distributions are as follows:

![alt text](./E5-small-CI-Errors.png)

![alt text](./Cohere-CI-Errors.png)

In the following, we always use at most 25k samples to compute the percentiles used
to clip components for linear quantisation.

## Quantisation

Linear quantisation to 8 bits is computed as follows:
$$
  \vec{x}_q = q(\vec{x};l,u) = \left[\frac{256 (clip(\vec{x}, l, u) - l)}{u - l}\right]
$$
where the subtraction is broadcast over the vector $\vec{x}$, $l$ and $u$ are upper and
lower quantiles (in the following we use central confidence intervals), the $clip$
function truncates componentwise to the interval $[l, u]$ and $[\cdot]$ denotes
round to the nearest integer. The inverse operation, or dequantise, is
$$
  \vec{x}_d = d(\vec{x};l,u) = l + \frac{(u - l) \vec{x}_q}{256}
$$
It is important to choose the confidence interval large enough such that no single
outlying component is clipped if they are not close in magnitude. A sufficient
condition for this is to use $CI > 1 - 1/d$ where $d$ is the vector dimension.
The Cohere embeddings have exactly this property as per the figure below.

![alt text](./Cohere-components.png )

In the following we explore how best to merge segments which contain different 
quantisation parameters $\{(l_i, u_i)\}$. The basic requirements is to minimise the
number of times we need to
1. Recompute quantiles ($l$ and $u$)
2. Requantise the vectors

Requantising vectors requires us to load every vector dequantise it using the old
quantiles and then requantise using the new quantiles. In general, if segments
contain random samples of the full dataset then, as per the discussion above, we
expect their quantiles to be very similar. Specifically, $l_i \thickapprox l_j$ 
and $u_i \thickapprox u_j$ for $i \neq j$. However, in adversarial cases, such as
if different segments contain disjoint regions vector space, we need to be able to
detect a problem and requantise. We explore two criteria to achieve this based on
the definition of quantisation operation.

Provided $|l_n - l_o| < \epsilon$ and $|u_n - u_o| < \epsilon$ for some small
$\epsilon$ then
$$
   d(q(\vec{x};l_o,u_o);l_n,u_n) \thickapprox d(q(\vec{x};l_o,u_o);l_o,u_o)
$$

In such cases there is no point in requantising since the result will be no more
accurate than retaining the current quantised vectors. We can deduce the largest
$\epsilon$ for which this is the case based on the definition of quantisation.
In particular, if $\epsilon \ll \frac{u - l}{256}$ then we do not expect
$q(d(\vec{x};l_o,u_o);l_n,u_n)$ to change many values in the quantised vector. Roughly
speaking we expect the dequantised values to be uniformly distributed on any of
the 256 subdivisions of $[l_n,u_n]$ which implies the probability that a component
will change in requantisation is $\epsilon / \frac{u - l}{256}$. In practice, we
found $\epsilon = \frac{0.2 (u - l)}{256}$, was sufficient to ensure that the
error introduced by retaining the original quantised vectors and only updating
the quantiles had almost no effect.

In order to compute the quantiles efficiently we use a weighted mean of the values
from each segment. The weight is proportional to the count of vectors in the segment.
This is to ensure that if any segment is very small the estimate is close to the
large segments (which will be accurate) and so we will not requantise them. Specifically,
the new quantiles are defined as
$$
  l_m = \frac{\sum_i{ |\{\vec{x}_i\}| l_i }}{\sum_i{ |\{\vec{x}_i\}| }}
$$
and
$$
  u_m = \frac{\sum_i{ |\{\vec{x}_i\}| u_i }}{\sum_i{ |\{\vec{x}_i\}| }}
$$
The criterion to choose to retain the original quantised vectors for a given
segment is then
$$
  |l_i - l_m| < \frac{0.2 (u_m - l_m)}{256} \text{ and } |u_i - u_m| < \frac{0.2 (u_m - l_m)}{256}
$$

The figures below show the RMSE error distributions between the raw vectors
and quantised vectors for a merge of four random segments.

![alt text](./E5-small-quantisation-RMSE.png)

![alt text](./Cohere-quantisation-RMSE.png)

The data for these were generated as follows:
```python
>>> import numpy as np
>>> from src.linear_merge import *
>>> x = read_fvecs("data/corpus-quora-E5-small.fvec")
>>> partition = [0] + [i for i in np.random.choice(x.shape[0], 3)] + [x.shape[0]]
>>> partition.sort()
>>> x_p = random_partition(x, partition)
>>> x_ = np.concatenate(x_p, axis=0)
>>> q_p = [central_confidence_interval(sample(x), 0.99) for x in x_p]
>>> x_p_q = quantise_all(x_p, q_p)
>>> x_m_q, q_m, r = merge_quantisation(x_p_q, q_p)
>>> x_m = dequantise(x_m_q, q_m[0], q_m[1])
>>> x_p = np.concatenate(dequantise_all(x_p_q, q_p), axis=0)
>>> c_m, e_m = np.histogram(compute_quantisation_rmse(x_, x_m), bins=100)
>>> c_p, e_p = np.histogram(compute_quantisation_rmse(x_, x_p), bins=100)
```
Observe that the baseline uses the per segment quantiles to dequantise while
the merged uses the weighted average of the segment quantiles. The decision
for whether to requantise using the vectors was based on the criterion above.
In this example no segments were requantised.

In the following we repeated this 100 times and tracked the relative difference
between the merged dequantised vectors and original dequantised vectors and
compared it to their difference from the raw vectors. Specifically, we computed
$$
  \frac{\sum_{i,j}{\|d(q(\vec{x}_{i,j};l_i,u_i);l_m,u_m)-d(q(\vec{x}_{i,j};l_i,u_i);l_i,u_i)\|}}{\sum_{i,j}{\|\vec{x}_{i,j}-d(q(\vec{x}_{i,j};l_i,u_i);l_i,u_i)\|}}
$$
The maximum relative error for E5-small
introduced by merge was 4%. In practice, this is essentially no different to the
error introduced by requantising. On average only 1% of vectors were requantised
and worst case only 15% of vectors were requantised.

![alt text](./E5-merge-relative-RMSE.png)

The decision to recompute quantiles rather than use the weighted average uses the
same form but a different value for $\epsilon$. We found it was sufficient to only
recompute quantiles in the case that $\epsilon>\frac{u_m-l_m}{32}$, this dealt with
all adversarial cases we discuss below. If this was the case for any segment then
we recomputed quantiles using 25k random samples. We sample each segment in proportion
to its count. In particular we sample a segment
$$
\left\lceil\frac{25000 |\{\vec{x}_i\}|}{\sum_i|\{\vec{x}_i\}|}\right\rceil
$$

Two distinct adversarial cases were explored:
1. For dot product (Cohere) vectors were sorted prior to partitioning,
2. For both dot and cosine similarity the vectors were clustered by k-means to
   generate partitions.
