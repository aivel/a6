#!/usr/bin/env python
# coding: utf-8

# <a id='SU' name="SU"></a>
# ## [Set up](#P0)

# In[1]:


#get_ipython().system('sudo apt-get install build-essential swig')
#get_ipython().system('curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install')
#get_ipython().system('pip install auto-sklearn')
#get_ipython().system('pip install shap')
#get_ipython().system('pip install pipelineprofiler # visualize the pipelines created by auto-sklearn')
#get_ipython().system('pip install scipy==1.7.3')
#get_ipython().system('pip install gdown')
#get_ipython().system('pip install dvc')
#get_ipython().system("pip install 'dvc[gdrive]'")


# In[2]:


from google.colab import drive
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from joblib import dump

import autosklearn.regression
import PipelineProfiler
import shap
import datetime
import logging


#drive.mount('/content/drive', force_remount=True)


# In[3]:


project_path = '/content/drive/MyDrive/GColab/A6/a6'
raw_data_path = f'{project_path}/data/raw'
models_path = f'{project_path}/models'
timesstr = str(datetime.datetime.now()).replace(' ', '_')


# In[4]:


log_config = {
    "version":1,
    "root":{
        "handlers" : ["console"],
        "level": "DEBUG"
    },
    "handlers":{
        "console":{
            "formatter": "std_out",
            "class": "logging.StreamHandler",
            "level": "DEBUG"
        }
    },
    "formatters":{
        "std_out": {
            "format": "%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(lineno)d : (Process Details : (%(process)d, %(processName)s), Thread Details : (%(thread)d, %(threadName)s))\nLog : %(message)s",
            "datefmt":"%d-%m-%Y %I:%M:%S"
        }
    },
}


# In[5]:


logging.config.dictConfig(log_config)


# In[6]:


df = pd.read_csv(f"{raw_data_path}/winequality-red.csv", delimiter=';')


# In[7]:


random_state = 320
test_size = 0.2

logging.info('Splitting the data: random_state = %s, test_size = %s' % (random_state, test_size))

x = df.loc[:, :'alcohol']
y = df.loc[:, 'quality']

logging.debug('x, y shape = %s, %s' % (x.shape, y.shape))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state, test_size=test_size)

x_train.to_csv(f'{raw_data_path}/winequality-red-x-train.csv', index=False)
x_test.to_csv(f'{raw_data_path}/winequality-red-x-test.csv', index=False)
y_train.to_csv(f'{raw_data_path}/winequality-red-y-train.csv', index=False)
y_test.to_csv(f'{raw_data_path}/winequality-red-y-test.csv', index=False)

logging.debug('x_train, y_train shape = %s, %s' % (x_train.shape, y_train.shape))
logging.debug('x_test, y_test shape = %s, %s' % (x_test.shape, y_test.shape))

x_train = x_train.copy()
x_test = x_test.copy()
y_train = y_train.copy()
y_test = y_test.copy()

xy_test_df = pd.DataFrame(pd.concat([x_test, y_test], axis=1))


# <a id='P2' name="P2"></a>
# ## [Modelling](#P0)

# ### Pipeline Definition

# In[8]:


logging.info('Defining pipeline')

regr = LinearRegression()
model = Pipeline(steps=[
                        ('regressor', regr)
                        ])


# ### Model Training

# In[9]:


cvs = cross_val_score(model, x_train, y_train)

logging.info('Cross val score = %s' % cvs)

model.fit(x_train, y_train)


# ### Auto ML
# 

# In[10]:


time_left_for_this_task = 60
per_run_time_limit = 30

logging.info('AutoML: time_left_for_this_task = %s, per_run_time_limit = %s' % (time_left_for_this_task, per_run_time_limit))

automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=time_left_for_this_task,
    per_run_time_limit=per_run_time_limit,
)
automl.fit(x_train, y_train)


# In[11]:


dump(automl, f'{models_path}/model_{timesstr}.pkl')
logging.info(f'Saved regressor model at {models_path}/model_{timesstr}.pkl ')


# In[12]:


logging.info(f'Autosklearn model statistics:')
logging.info(automl.sprint_statistics())


# In[13]:


profiler_data = PipelineProfiler.import_autosklearn(automl)
PipelineProfiler.plot_pipeline_matrix(profiler_data)


# ### Model Evaluation

# In[14]:


logging.info("Let's try to predict the target value with the use of our model.")

y_test_pred = model.predict(x_test)

logging.info(f"Mean Squared Error is {mean_squared_error(y_test, y_test_pred)}, \n R2 score is {automl.score(x_test, y_test)}")


# Let's also plot y_test against y_test_pred on a scatter for both models
# 
# 

# In[15]:


test_pred_df = pd.DataFrame(np.concatenate((x_test, y_test.to_numpy().reshape(-1, 1), y_test_pred.reshape(-1, 1)),  axis=1))
test_pred_df.columns = [*xy_test_df.columns, 'quality-p']

column = 'alcohol'

fig = px.scatter(test_pred_df, x=column, y=test_pred_df.loc[:, 'quality'])
fig.add_trace(go.Scatter(x=test_pred_df[column], y=test_pred_df.loc[:, 'quality-p'],
                    mode='markers',
                    name='model'))

fig.write_html(f"{models_path}/test-vs-pred-main_{timesstr}.html")


# In[16]:


fig = px.scatter(test_pred_df, x='quality', y='quality-p')
fig.write_html(f"{models_path}/test-vs-pred-scatter_{timesstr}.html")


# ## Model explainability

# In[17]:


explainer = shap.KernelExplainer(model = automl.predict, data = x_test.iloc[:50, :], link = "identity")


# In[18]:


# Set the index of the specific example to explain
X_idx = 0

shap_value_single = explainer.shap_values(X = x_test.iloc[X_idx : X_idx + 1, :], nsamples = 100)

x_test.iloc[X_idx : X_idx + 1, :]


# In[19]:


# print the JS visualization code to the notebook

shap.initjs()
shap.force_plot(base_value = explainer.expected_value,
                shap_values = shap_value_single,
                features = x_test.iloc[X_idx:X_idx+1,:], 
                show=False,
                matplotlib=True
)

plt.savefig(f"{models_path}/shap_example_{timesstr}.png")

logging.info(f"Shapley example saved as {models_path}/shap_example_{timesstr}.png")


# In[20]:


shap_values = explainer.shap_values(X = x_test.iloc[0:50,:], nsamples = 100)


# In[21]:


# print the JS visualization code to the notebook

shap.initjs()

fig = shap.summary_plot(shap_values = shap_values,
                  features = x_test.iloc[0:50,:],
                  show=False)

plt.savefig(f"{models_path}/shap_summary_{timesstr}.png")

logging.info(f"Shapley summary saved as {models_path}/shap_summary_{timesstr}.png")


# In[21]:




