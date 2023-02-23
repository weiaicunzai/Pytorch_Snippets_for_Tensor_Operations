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
