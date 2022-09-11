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
tensor([2, 3, 5, 6])
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