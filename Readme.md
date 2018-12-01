# Group Normalization

**2018/12/1**: 

* For details, please reference the [blog post](https://www.jarvis73.cn/2018/04/10/Group-Normalizatioin/).
* Other implementation in tensorflow contributions: [Group Norm](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/group_norm)

**Original:**

Here is a simple group normalization [http://arxiv.org/abs/1803.08494](http://arxiv.org/abs/1803.08494) with tensorflow.

If you want to use the simplest implementation of group normalization, then just use the code given in the paper and remember to define the trainable variables `gamma` and `beta`. And these two variables should have the same shape of `[N, G, 1, 1, 1]` (tensorflow will broadcast it to `[N, G, C//G, H, W]`).

But maybe someone want to use a moving average version of group norm, just like batch norm implemented in tensorflow. So I write a simple wrapper function to use moving average. If you have read the paper of Wu and He, you should know that LN (Layer Norm) and IN (Instance Norm) are two extreme cases of GN. Inspired by this conclusion, actually we can implement group norm directly in tensorflow using `tf.layers.batch_normalization`. Actually, in the doc of `tf.layers.BatchNormalization`, parameter `axis` is explained detailedly. 

For an input tensor with shape `[N, H, W, C]`:

* If `axis=-1`, then standard `Batch norm` is used.
* If `axis=0`, then it becomes `Layer norm`.
* If `axis=[0,-1]`, then it becomes `Instance norm`.

For an input tensor with shape `[N, C, C//G, H, W]`:

* If `axis=[0, 1]`, then it becomes `Group norm`.

## Requirements

* tensorflow >= 1.5
