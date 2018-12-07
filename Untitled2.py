
# coding: utf-8

# In[1]:


import xgboost


# In[2]:


import shap


# In[3]:


from IPython.display import display, HTML, Image


# In[4]:


shap.initjs()


# In[5]:


X,y = shap.datasets.boston()


# In[6]:


model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)


# In[7]:


explainer = shap.TreeExplainer(model)


# In[8]:


shap_values = explainer.shap_values(X)


# In[9]:


shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])


# In[10]:


import xgboost
import shap
from IPython.display import display, HTML, Image

# load JS visualization code to notebook
shap.initjs()

# train XGBoost model
X,y = shap.datasets.boston()
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])


# In[11]:


shap.force_plot(explainer.expected_value, shap_values, X)


# In[12]:


import xgboost
import shap

# load JS visualization code to notebook
shap.initjs()

# train XGBoost model
X,y = shap.datasets.boston()
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])

