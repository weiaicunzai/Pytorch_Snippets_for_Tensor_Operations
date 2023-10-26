# Pytorch Code Snippets For Complex Tensor Operations

This repository contains the snippets for Various Complex Pytorch Tensor Operations.
We have often encountered a situation that requires complex parallel tensor operations to speed up the program, instead of for loops.
However, these operations are often composed of many simple operations, and there are no APIs we can directly use.
Memorizing these simple operations can be hard and unnecessary sometimes.
And documents on Complex Pytorch Tensor Operations are insufficent at the moment.
The Pytorch Code Snippets for Complex Tensor Operations are therefore included in this repository to address the issue of a lack of documentation.


## Symmetric Matrix Mask of Same Values in a 1D Tensor
Snippet Source: [InforNCE loss at SimCLR](https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/simclr.py#L26)

Examples:
```python
# we have
tensor([1, 2, 3, 3])

# we want:
# it is a symmetric matrix
# the output[i, j] is True if input[i] == input[j], otherwise False
# the first element of input is 1, 1 is unique in the list, so output[0, 0] is True
tensor([[ True, False, False, False],
        [False,  True, False, False],
        [False, False,  True,  True],
        [False, False,  True,  True]])
```

Code:
```python
x = torch.tensor([1, 2, 3, 3])

# x.unsqueeze(0) has a shape of [1, 4]
# x.unsqueeze(1) has a shape of [4, 1]
# == operations will make them both broadcast to shape[4, 4]

# x.unsqueeze(0) will broadcast to
# tensor([[1, 2, 3, 3],
#         [1, 2, 3, 3],
#         [1, 2, 3, 3],
#         [1, 2, 3, 3]])
#
# x.unsqueeze(1) will broadcast to
# tensor([[1, 1, 1, 1],
#         [2, 2, 2, 2],
#         [3, 3, 3, 3],
#         [3, 3, 3, 3]])

mask = x.unsqueeze(0) == x.unsqueeze(1)
print(mask)
```
Output:
```python
tensor([[ True, False, False, False],
        [False,  True, False, False],
        [False, False,  True,  True],
        [False, False,  True,  True]])

# mask[0, 0] = True means x[0] == x[0];
# mask[1, 1] = True means x[1] == x[1];
# mask[2, 3] = True means x[2] == x[3];
# mask[3, 2] = True means x[3] == x[2];
# mask[3, 3] = True means x[3] == x[3];
```

## Distance Matrix of two tokens
Snippet Source: [Relative Position Embeddings](https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/fd1163eb72d74538932f4f2d62e9c2e876f232e5/relative_position.py#L13C11-L13C11)

Examples:
```python
# we have
q_len=5
v_len=5

# we want:
# Distance between two tokens in a seq
# in matrix M,  M[i, j] represent distance between the ith and jth token in the same sequence
tensor([[ 0,  1,  2,  3,  4],
        [-1,  0,  1,  2,  3],
        [-2, -1,  0,  1,  2],
        [-3, -2, -1,  0,  1],
        [-4, -3, -2, -1,  0]])
```

Code:
```python
a = torch.arange(5)
b = torch.arange(5)

# a[None, :] broad cast to
# tensor([[ 0,  1,  2,  3,  4],
#        [0,  1,  2,  3,  4],
#        [0,  1,  2,  3,  4],
#        [0,  1,  2,  3,  4],
#        [0,  1,  2,  3,  4]]
# b[:, None] broad cast to
# tensor([[0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1],
#         [2, 2, 2, 2, 2],
#         [3, 3, 3, 3，3],
#         [4, 4, 4, 4, 4]])


output = a[None,:] - b[:,None]

print(output)
```
Output:
```python
tensor([[ 0,  1,  2,  3,  4],
        [-1,  0,  1,  2,  3],
        [-2, -1,  0,  1,  2],
        [-3, -2, -1,  0,  1],
        [-4, -3, -2, -1,  0]])

```


## Get the Index (start from 1) of Elements Which Are Different From the Last Element in a List
Snippet Source: [SWAV](https://github.com/facebookresearch/swav/blob/5e073db0cc69dea22aa75e92bfdd75011e888f28/src/resnet50.py#L308)

Examples:
```python
# we have
tensor([22, 22, 33, 11, 11, 44])

# we want:
tensor([2, 3, 5, 6]) # 33 is different from 22, 11 is different from 33, 44 is different from 11
```

Code:
```python
x = [22, 22, 33, 11, 11, 44]
idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor(x),
            return_counts=True,
        )[1], 0)


# torch.unique_consecutive(torch.tensor(x), return_corners=True)
# (tensor([22, 33, 11, 44]), tensor([2, 1, 2, 1]))

#  torch.cumsum(torch.tensor([22, 33, 11, 44]))
#  tensor([ 22,  55,  66, 110])
print(idx_crops)
```
Output:
```python
tensor([2, 3, 5, 6])

# the 2nd element of input 22
# the 3rd element of input 33
# the fifth element of input 11
# the sixth element of input 44

```
## Explain of nonzero() method

Pytorch build-in method
1: always return a 2D tensor
1: returns the indices of the non zero values
2: the first dimension of the 2D tensor represents the number of the non zero values
3: the second dimension of the 2D tensor represents the indices of each dimension of the non zero values.

example:
returned 2D index:
```python
# shape [5, 3] five non zero values(each row), each row represent a 3 dimension index of a non zero value

# the first column represent the indices of all non zero values in the first dimension
# the second column represent the indices of all non zero values in the second dimension
# the 3rd column represent the indices of all non zero values in the 3rd dimension

indices = tensor([[0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1]])

img[indices[0]] # get the first non zero values ([0, 0, 0] means the first non zero value is in the 0th of first dim, 0th of the 2nd dim, 0th of the 3rd dim)
```

```python
a = torch.tensor([1, 0, 3, 0])
print(a.nonzero())
```

Output:
```python
# shape: [2, 1]    tensor a contains two non zero values, the first element is at
tensor([[0],
        [2]])
```

```python
a = torch.tensor([[1, 0], [3, 0]])
print(a.nonzero())
```

Output:
```python
tensor([[0, 0],
        [1, 0]])
```

```python
a = torch.tensor([[[1, 0], [3, 0]], [[3, 0], [4, 5]]])
print(a.nonzero())
```

Output:
```python
# shape [5, 3] : 5 non zero values, each are represented in a 3D index
tensor([[0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1]])
```




## Random sample N elements of a tensor (numpy.random.choice Pytorch equivalent)
Snippet Source: [ContrastiveSeg](https://github.com/tfzhou/ContrastiveSeg/blob/2ab84d8ec679adc7f7be1853c8684b44bf899273/lib/loss/loss_contrast_mem.py#L30)

Examples:
```python
# we have
img = torch.Tensor(3, 16, 224, 224)

# we want
samples = img[K_indices]
samples.shape # [K, 16]
```


Code:
```python
# tensor: example
img = torch.Tensor(3, 16, 224, 224)

# move the unsampled channel to the last
img = img.permute(0, 2, 3, 1)
# flatten the sampled dimension
img = img.contiguous().view(-1, 16)

# the population
# here we set all the elements in the tensor as population for simplicity
mask = torch.ones(3, 224, 224)
# flatten the mask accordingly
mask = mask.contiguous().view(-1)


K = 100 # number of samples

# sample K indices
indices = mask.nonzero().squeeze(-1)
perm = torch.randperm(mask.numel())
random_indices = perm[:K]


print(img[random_indices].shape)

```

Output:
```python
torch.Size([100, 16])
```

## Explain How Masked Index Works in Pytorch

Explain how masked index works in multi dimension tensor, mask is bool type.


```python
a = torch.randn((2, 2, 4, 4))
#
mask = a.sum(dim=(2, 3)) > 0

#the shape of mask is torch.Size([2, 2])

print(mask, mask.shape)

# the shape of a is [2, 2, 4, 4]

# matching process........................................
# first compare the shape of two tensors (mask and a)
# the mask index start to match with the index of a start from index 0 (the first dimension)
# e.g. the shape of mask is 2, 2, and the first two dimension of a is 2, 2
# therefore the mask and tensor a are matched with each other in the first and second dimension
print("a.shape:", a.shape, "a[mask].shape:", a[mask].shape)
```


Output of multiple runs
```python
# the shape of mask is [2, 2], the shape of a is [2, 2, 4, 4]
# therefore the first two dimensions of a are masked
# there are 4 elements in mask,  4 elements (each element is a 4x4 tensor in this case) in a's first two dimension
# the shape of a[mask] is  [N, 4, 4], N is number of selected elements in mask
# if no element is selected by the mask, the output tensor is in shape [0, 4, 4]


tensor([[ True,  True],
        [False, False]]) torch.Size([2, 2])
a.shape: torch.Size([2, 2, 4, 4]) a[mask].shape: torch.Size([2, 4, 4])

tensor([[False, False],
        [False, False]]) torch.Size([2, 2])
a.shape: torch.Size([2, 2, 4, 4]) a[mask].shape: torch.Size([0, 4, 4])
```


**if mask is long type, this may not be the case**
```python
a.shape : [a1, a2, a3]
mask.shape : [m1, m2, m3, m4] # mask.dtype is int64

# it is basically like:
# mask = mask.view(m1 * m2 * m3 * m4)
# out = a[mask]
# out = out.view(m1, m2, m3, m4, a2, a3)
# only indexing from the first dimension

out = a[mask].shape :[m1, m2, m3, m4, a2, a3]
```

## Explain How gather() Works in Pytorch

gather() replaces one dimension with index values. For example, if we have a sentence with shape [batch_size, seq_length, hidden_size], we want to translate this sentence into shape [batch_size, labels_length, hidden_size], we could use gather() to help us. In the official Pytorch document, we have
```python

# input: 3D tensor, out: 3D tensor
# out and index will have the same shape

# replace the ith element of dim 0 in out with (index[i][j][k])-th element of input in dim 0
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0

# replace the ith element of dim 1 in out with (index[i][j][k])-th element of input in dim 1
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1


# replace the ith element of dim 2 in out with (index[i][j][k])-th element of input in dim 2
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
```

Code:
```python
x = torch.arange(3 * 3 * 4 * 4)
x = x.reshape(3, 3, 4, 4)
x = x.view(3, -1)
values, indices = x.topk(3, dim=1)
# indices and x should have the same number of dimension
# e.g.  x: [4, 4], index[4, 3] :ok
# e.g.  x: [4, 4], index[4, 3, 3] :not ok
c = torch.gather(x, index=indices, dim=1)

print(x)
print(indices)

# the value of c[0, 0] is the same as x[0, indices[0, 0]]
# the value of c[0, 1] is the same as x[0, indices[0, 1]]
print(c)
```

Output:
```python
# shape[3, 48]
tensor([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
          14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
          28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
          42,  43,  44,  45,  46,  47],
        [ 48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
          62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,
          76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
          90,  91,  92,  93,  94,  95],
        [ 96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
         110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
         124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
         138, 139, 140, 141, 142, 143]])

# shape[3, 3]
tensor([[47, 46, 45],
        [47, 46, 45],
        [47, 46, 45]])

# shape[3, 3]
tensor([[ 47,  46,  45],
        [ 95,  94,  93],
        [143, 142, 141]])
```


However, for most cases, index and input does not have the same dimension, this will raise an error. Inorder to use ``gather()``, we can use ``squeeze()`` or ``expand()`` to make sure they have the same dimension.

Examples:

```python
# input tensor
a = torch.randn(2, 3, 4)

# index tensor
b = torch.randn(2, 3)
_, index = b.topk(dim=1, k=2) # shape: [2, 2]

# we have input shape: [2, 3, 4], and index shape: [2, 2]
# we want to extract feats from input tensor according to the index
# the output shape we want is out: [2, 2, 4]
# inorder to use gather, we first unsqueeze the index shape from [2, 2] to [2, 2, 1]
# index shape of [2, 2, 1] will result the shape of out become: [2, 2, 1], but we want
# to extract all 4 channels of the input tensor with shape [2, 2, 4]
# so we replicate the index 4 times to [2, 2, 4], expand function can achieve this goal
# expand function do not consume extra memory, only create more views

index = index.unsqueeze(-1).expand(-1, -1, a.shape[-1])
out = torch.gather(input=a, index=index, dim=1)


print('------')
print(a)
print(index[:, :, 0])
print('------')

# as we can see, a and out have the same result
print(a[0, index[0, 0, 0], :])
print(a[0, index[0, 1, 0], :])
print(a[1, index[1, 0, 0], :])
print(a[1, index[1, 1, 0], :])
print()
print(out)
```