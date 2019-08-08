# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython


#%%
import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import pylab as plt
plt.style.use('fivethirtyeight')


#%%
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from io import StringIO
from IPython.display import Image  
import pydotplus

from sklearn.model_selection import cross_validate, cross_val_score


#%%
data = pd.read_csv('./data/EF_battles_corrected.csv', parse_dates=['start', 'end'])


#%%
data['result_num'] = data['result'].map({'axis':-1, 'allies':1}).fillna(0) # * data['result_prop']


#%%
mask = data[['allies_infantry', 'axis_infantry']].isnull().any(1)


#%%
data=data[~mask]


#%%
cols_to_fill = ['allies_planes', 'axis_planes', 'axis_tanks', 'allies_tanks', 'axis_guns', 'allies_guns']
mask_null = data[cols_to_fill].isnull()
data[cols_to_fill] = data[cols_to_fill].fillna(0)


#%%
data['end_num'] = (data['end'].dt.year -1938) * 12 +  data['end'].dt.month
data['start_num'] = (data['start'].dt.year -1938) * 12 +  data['start'].dt.month
data['duration'] = (data['end_num'] - data['start_num']).clip(lower=1)


#%%
cols = [
    'allies_infantry', 'axis_infantry', 'allies_tanks', 'axis_tanks', 
    'allies_planes', 'axis_planes', 'duration'
]


#%%
# ?classification_report

#%% [markdown]
# ## Models

#%%
model1 = DecisionTreeClassifier(random_state=2019, max_depth=10)


#%%
data[cols].head(10)


#%%



#%%
cv = cross_validate(model1,
                    data[cols], 
                    data['result_num'],
                    cv=4)

cv = pd.DataFrame(cv)
print(cv.to_string())


#%%
cv['test_score'].mean()

#%% [markdown]
# # Feature Engineering
#%% [markdown]
# #### Army size comparison

#%%
data['infantry_ratio'] = data['allies_infantry'] / data['axis_infantry']
cols.append('infantry_ratio')

for tp in 'infantry', 'planes', 'tanks', 'guns':
    data[f'{tp}_diff'] = data[f'allies_{tp}'] - data[f'axis_{tp}']
    cols.append(f'{tp}_diff')


#%%
scores = cross_val_score(model1, 
                         data[cols], 
                         data['result_num'],
                         cv=4)
pd.np.mean(scores)


#%%
scores

#%% [markdown]
# # leaders

#%%
model1.__str__().replace('\n                       ', ' ')


#%%
from collections import Counter
# lea


#%%
alead = data['axis_leaders'].str.split(',')


#%%
# alead[alead.notnull()].apply(lambda x: 'Erich von Manstein' in x)


#%%
def _generate_binary_most_common(col, N=10):
    mask = col.notnull()
    lead_list = [ el.strip() for _, cell in col[mask].iteritems() for el in cell if el != '']
    c = Counter(lead_list)
    
    mc = c.most_common(N)
    df = pd.DataFrame(index=col.index, columns=[name[0] for name in mc])
    
    for name in df.columns:
        df.loc[mask, name] = col[mask].apply(lambda x: name in x).astype(int)
    return df.fillna(0)
    
    
    


#%%
axis_pop = _generate_binary_most_common(data['axis_leaders'].str.split(','), N=2)
allies_pop = _generate_binary_most_common(data['allies_leaders'].str.split(','), N=2)


#%%
data2 = pd.concat([data, axis_pop, allies_pop], axis=1)


#%%
cols2 = cols + axis_pop.columns.tolist() + allies_pop.columns.tolist()


#%%
model2 = DecisionTreeClassifier(random_state=2019, max_depth=15)


#%%
scores = cross_val_score(model1, 
                         data2[cols2], 
                         data2['result_num'],
                         cv=4)
pd.np.mean(scores)

#%% [markdown]
# ### Belligerents

#%%
data2['belligerents.allies']


#%%
scores = cross_val_score(model1, 
                         data2[cols2], 
                         data2['result_num'],
                         cv=4)
pd.np.mean(scores)

#%% [markdown]
# ## Hyper-parameter optimisation

#%%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint


#%%
# specify parameters and distributions to sample from
param_dist = {"max_depth": sp_randint(5, 20),
              "max_features": sp_randint(1, len(cols2)),
              "min_samples_split": sp_randint(2, 11),
              "criterion": ["gini", "entropy"]}


#%%
rs = RandomizedSearchCV(
    model1,
    param_distributions=param_dist,
    cv=4, iid=False,
    random_state=2019,
    n_iter=50
)


#%%
rs.fit(data2[cols2], data2['result_num'])


#%%
rs.best_score_


#%%
rs.best_estimator_

#%% [markdown]
# # Visualize the graph

#%%
dot_data = StringIO()

export_graphviz(rs.best_estimator_, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=cols2)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

#%% [markdown]
# ### Storing tree chart to a file

#%%
# export_graphviz(rs.best_estimator_, out_file='./assets/graph.dot',  
#                 filled=True, rounded=True, special_characters=True, feature_names=cols2)

#%% [markdown]
# # Random Forest

#%%
from sklearn.ensemble import RandomForestClassifier


#%%
rf = RandomForestClassifier(random_state=2019)


#%%
scores = cross_val_score(rf, 
                         data2[cols2], 
                         data2['result_num'],
                         cv=4)
pd.np.mean(scores)


#%%
param_dist2 = {
              'n_estimators': sp_randint(50, 2000),
              "max_depth": sp_randint(5, 25),
              "max_features": sp_randint(1, len(cols2)),
              "min_samples_split": sp_randint(2, 11),
              "criterion": ["gini", "entropy"]
}


#%%
rs2 = RandomizedSearchCV(
    rf,
    param_distributions=param_dist2,
    cv=4, iid=False,
    random_state=2019,
    n_iter=50
)


#%%
rs2.fit(data2[cols2], data2['result_num'])


#%%
rs2.best_score_


#%%
data2[cols2]


#%%



