Soft DTW for PyTorch in CUDA
===
Fast CUDA implementation of [soft-DTW](https://github.com/mblondel/soft-dtw) for PyTorch. 
Based on [pytorch-softdtw](https://github.com/Sleepwalking/pytorch-softdtw) but can run up to 100x faster!
Both `forward()` and `backward()` passes are implemented using CUDA.

My implementation is partly inspired by 
[_"Developing a pattern discovery method in time series data and its GPU acceleration"_](https://ieeexplore.ieee.org/document/8400444)
wherein a diagonal-based implementation of the Belman recursion is proposed. 

How to use
---
This code depends on [PyTorch](https://pytorch.org/) and [Numba](http://numba.pydata.org/). 
Just include `soft_dtw_cuda.py` in your projects, and you should be good to go!

You can also run the included profiler/test (tested with Python v3.6), and see the speedups you'd get:

```
git clone https://github.com/Maghoumi/pytorch-softdtw-cuda
cd pytorch-softdtw-cuda
python soft_dtw_cuda.py
```

### Sample code
A sample code is already provided in the script. Here's a quick example:

```python
from soft_dtw_cuda import SoftDTW
...
criterion = SoftDTW(use_cuda=True, gamma=0.1)
...
loss = criterion(out, target)  # Just like any torch.nn.xxxLoss()
# Aggregate and call backward()
loss.mean().backward()
```

FAQ:
---

### This is awesome! What can I do to help?
Consider starring this repository if you find it helpful. Also, don't forget to thank the author of 
[pytorch-softdtw](https://github.com/Sleepwalking/pytorch-softdtw) for his CPU implementation.

Also, please consider contributing to this project by improving the performance, addressing existing 
limitations, etc. PRs are greatly welcome!

### Does it support pruning?
Yes! Use the `bandwitdh` argument to specify the Sakoe-Chiba bandwidth to use for pruning.

### How fast does it run?
It depends on your batch size and sequence length. The longer the sequences and the larger the batch size,
the faster this code runs.
 
Here's what I get on with Ryzen 9 3900x and Titan RTX:

```
Profiling forward() + backward() times for batch_size=128, seq_len=16, dims=2...
	CPU:      0.008581484199385158
	GPU:      0.001516376400104491
	Speedup:  5.659204534437374

Profiling forward() + backward() times for batch_size=512, seq_len=64, dims=2...
	CPU:      0.23573041420022492
	GPU:      0.0035874296001566107
	Speedup:  65.71011573019692

Profiling forward() + backward() times for batch_size=512, seq_len=256, dims=2...
	CPU:      3.729714207601501
	GPU:      0.03254888820010819
	Speedup:  114.58806779117891
```

Note that there are tons of opportunities for optimizing this code further (e.g. various 
CUDA optimizations such as the use shared memory, etc.). Contributions/improvements are greatly appreciated!

### How accurate are the results?
Depends on the length of your inputs. Because of the sequential nature of this code, the longer your input
sequences are, the higher numerical errors become due to accumulation. Especially in the `backward()` call,
you could see floating point errors of up to `1e-3` in the resulting derivative tensor.

The unit tests included in `soft_dtw_cuda.py` verify the results against the CPU implementation.

### What are the limitations?
Some limitations are:

1. All sequences in the batch should have the same length / number of features.
2. Both `x` and `y` should have the same length (time steps). Otherwise the code will use the CPU implementation.
3. Inputs cannot have lengths longer than 1024 (due to CUDA limitations on the maximum block size). 
   The code will warn if your sequence length is too long, and will fall-back to the CPU implementation. 
4. You may run out of CUDA resources if your inputs are long (but still less than 1024). See below.

### I'm seeing `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`. Help!
This means the length of your sequences is too long, and your GPU cannot spawn a sufficient number of threads.
This is related to point 4 above in the "limitations". I'm not sure if it's possible to query the CUDA device
in Numba to see if launching the kernel is possible given the number of necessary threads. In these cases
consider using the CPU implementation.  

License
---
This project is licensed under the MIT License.
