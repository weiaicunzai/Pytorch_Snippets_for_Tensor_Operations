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

# mask[0, 0] means x[0] == x[0]; 
# mask[1, 1] means x[1] == x[1]; 
# mask[2, 3] means x[2] == x[3]; 
# mask[3, 2] means x[3] == x[2]; 
# mask[3, 3] means x[3] == x[3];
```