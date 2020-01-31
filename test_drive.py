# test_drive.py
# I use this file to test things, 


import torch 
import numpy as np




if __name__ is '__main__':
    a = np.array([[ 0.12358109, -0.9923345 ,  0.22797763],
       [-0.6561437 ,  0.75463593,  1.9027297 ],
       [-0.72913426,  0.6843707 , -3.3165429 ],
       [-0.7543768 ,  0.6564416 ,  0.9942609 ],
       [ 0.02537189, -0.9996781 , -1.3757801 ]], dtype=np.float32)

    b = np.array([-0.5227407 ,  1.7110571 ,  1.8942035 ,  2.437542  ,  0.18703152], dtype=np.float32)




    c = np.concatenate((a,np.expand_dims(b, axis=1)), axis=1)
    print(c)
