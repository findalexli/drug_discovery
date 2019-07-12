
# coding: utf-8

# In[1]:


from fastai.vision import *


# In[2]:


tfms = get_transforms(do_flip=True)


# In[3]:


np.random.seed(42)
data = ImageDataBunch.from_csv('.',ds_tfms=tfms, size = 400)


# In[5]:


data.show_batch(row=3)


# In[6]:


learn = cnn_learner(data, models.resnet18, metrics=accuracy)


# In[8]:


learn.fit(20)


# OK. This performance is not good. Let us try another model. Restnet34 this time.

# In[11]:


learn_restnet34 = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[12]:


learn_restnet34.fit(20)


# Slightly better than Resnet 18. let us see if we are looking at the right learning rate by using the Learning rate finder

# In[13]:


lr_find(learn_restnet34)


# In[14]:


learn_restnet34.recorder.plot()


# While patially from this plot, we know that Resnet 34 is probably not a good model for our dataset, we still note that 1e-2 might be the best learning rate to choose 

# In[16]:


lr = 1e-2


# In[17]:


learn_restnet34.fit_one_cycle(20,slice(lr),pct_start=0.8)


# ## Let us try Resnet 50 

# We are having memory issues, so set a smaller batch size from the start. I found a intersting twitter about practical tips from simple tricks to multi-GPU code and distributed setups. https://twitter.com/Thom_Wolf/status/1051771906255454208

# In[1]:


from fastai.vision import *


# In[2]:


tfms = get_transforms(do_flip=True)


# In[6]:


np.random.seed(42)
bs = 32
data = ImageDataBunch.from_csv('.',ds_tfms=tfms, size = 400, bs=bs)


# Note: Resnet50 takes more GPU memory than Resnet 18 or Resnet34, so starting from bs = 256, I tried 256, 128, 64, 32, the last of which works. 

# In[10]:


learn_resnet50 = cnn_learner(data, models.resnet50, metrics=error_rate)


# In[11]:


lr_find(learn_resnet50)


# In[12]:


learn_resnet50.recorder.plot()


# In[13]:


lr= 2e-3
learn_resnet50.fit_one_cycle(20,slice(lr),pct_start=0.8)


# ## Let us try Seueeze Net  

# From now on, let us reset our cuda memory each time. Note that si

# In[ ]:


get_ipython().run_line_magic('reset', '')


# from faastai.vision import *
# tfms = get_transforms(do_flip=True)
# np.random.seed(42)
# bs = 32
# data = ImageDataBunch.from_csv('.',ds_tfms=tfms, size = 400, bs=bs)
# learn_squeezenet1_0 = cnn_learner(data, models.squeezenet1_0, metrics=error_rate)

# In[ ]:


lr_find(learn_squeezenet1_0)
learn_squeezenet1_0.recorder.plot()


# In[ ]:


lr= 2e-3
learn_resnet50.fit_one_cycle(20,slice(lr),pct_start=0.8)


# We can try squeezenet1-1

# In[14]:


get_ipython().run_line_magic('reset', '')


# In[ ]:


from faastai.vision import *
tfms = get_transforms(do_flip=True)
np.random.seed(42)
bs = 32
data = ImageDataBunch.from_csv('.',ds_tfms=tfms, size = 400, bs=bs)
learn_squeezenet1_1 = cnn_learner(data, models.squeezenet1_1, metrics=error_rate)


# In[ ]:


lr_find(learn_squeezenet1_0)
learn_squeezenet1_0.recorder.plot()

Remember to change learning rate each time based on the plot coming out of the learning rate finder
# In[ ]:


lr= 2e-3 
learn_resnet50.fit_one_cycle(20,slice(lr),pct_start=0.8)

