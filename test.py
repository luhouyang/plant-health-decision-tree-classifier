# add '#%%' to make it run like Jupyter Notebook

# has shape of (5,)
array_1 = [1, 2, 3, 4, 5]

# has shape of (3, 5)
array_2 = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15]
]

# has shape of (3, 3, 2)
multi_dimension_array = [
  [
    [1, 2], [1, 2], [1, 2]
  ],
  [
    [1, 2], [1, 2], [1, 2]
  ],
  [
    [1, 2], [1, 2], [1, 2]
  ]
]

import numpy as np

np_arr_1 = np.asarray(array_1)
np_arr_2 = np.asarray(array_2)
np_arr_3 = np.asarray(multi_dimension_array)

print(np_arr_1.shape)
print(np_arr_2.shape)
print(np_arr_3.shape)
