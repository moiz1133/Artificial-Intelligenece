#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x=2*np.random.rand(100,1)
y=4+3*x+np.random.randn(100,1)


# In[17]:


lin_reg=LinearRegression()
lin_reg.fit(x,y)
lin_reg.intercept_,lin_reg.coef_
x_new=np.array([[0],[2]])
lin_reg.predict(x_new)


# In[ ]:




