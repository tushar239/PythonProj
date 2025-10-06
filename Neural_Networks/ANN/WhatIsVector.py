'''
A vector is simply an ordered list of numbers — like a one-dimensional array.
It represents a quantity that has both magnitude and direction.

import numpy as np
v = np.array([2, 3])

2. Types of Vectors
Type	            Shape	        Example	            Meaning
Row Vector	        (1, n)	        [1, 2, 3]	        1 row, n columns
Column Vector	    (n, 1)          \[1, 2, 3]          n rows, 1 column
1D Vector (in code)	(n,)	        np.array([1,2,3])	No explicit row/column orientation

4. In Machine Learning

Vectors are used everywhere:

A feature vector represents one data sample:
e.g., [height, weight, age] = [180, 75, 25]

In NLP, a word vector (embedding) represents a word as a list of numbers capturing meaning:
e.g., “king” → [0.25, 0.81, 0.63, …]

In neural networks, inputs, weights, and outputs are often vectors or matrices.

5. Relationship to Matrices and Scalars

Concept	    Example	        Shape	    Description
Scalar	    5	            ()	        Single number
Vector	    [1, 2, 3]	    (3,)	    1D array
Matrix	    [[1,2],[3,4]]	(2,2)	    2D array
Tensor	    3D+ array	    (3,3,3,…)	Higher-dim structure
'''