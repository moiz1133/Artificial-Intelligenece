#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
x=2*np.random.rand(100,1)
y=4+3*x+np.random.randn(100,1)
print(x)
print(y)


# In[9]:


x_new=np.c_[np.ones((100,1)),x]
thetaBest=np.linalg.inv(x_new.T.dot(x_new)).dot(x_new.T).dot(y)
print(thetaBest)


# In[10]:


x_n=np.array([[0],[2]])
x_n_new=np.c_[np.ones((2,1)),x_n]
ypredicted=x_n_new.dot(thetaBest)
ypredicted


# In[14]:


plt.plot(x_n,ypredicted,"r-")
plt.plot(x,y,"b.")
plt.axis([0,2,0,15])
plt.show()


# In[ ]:




