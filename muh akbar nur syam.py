#!/usr/bin/env python
# coding: utf-8

# In[7]:


MENAMPILKAN SUMMARY DAN VISUAL 3D
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
data = {
    'X1': [50, 40, 60, 55, 45, 65, 70, 75, 80, 90, 89, 87, 20, 30, 15, 55, 45, 65, 50, 40],
    'X2': [20, 25, 30, 35, 40, 50, 55, 60, 65, 70, 66, 78, 77, 55, 66, 25, 20, 35, 30, 45],
    'Y': [60, 55, 65, 70, 62, 75, 80, 85, 90, 95, 77, 87, 10, 20, 30, 60, 50, 60, 65, 67],
}

df = pd.DataFrame(data)
X = df[['X1', 'X2']]  # Variabel independen
X = sm.add_constant(X)  # Tambahkan konstanta
Y = df['Y']  # Variabel dependen

model = sm.OLS(Y, X).fit()  # Membuat model regresi OLS
summary = model.summary()  # Tampilkan ringkasan model
print(summary)

# Scatter plot untuk melihat hubungan antara X1, X2, dan Y
fig = px.scatter_3d(df, x='X1', y='X2', z='Y', title='Scatter Plot 3D')

# Regresi berganda sebagai bidang
xx, yy = np.meshgrid(df['X1'], df['X2'])
zz = model.params['const'] + model.params['X1'] * xx + model.params['X2'] * yy
fig.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=0.8, colorscale='Viridis'))

fig.show()

''' from statsmodels.stats.diagnostic import het_breuschpagan

_, p_value, _, _ = het_breuschpagan(model.resid, X)
if p_value < 0.05:
    print("Terdapat bukti heteroskedastisitas.")
else:
    print("Tidak terdapat bukti heteroskedastisitas.") '''


# In[6]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
data = {
    'X1': [50, 40, 60, 55, 45, 65, 70, 75, 80, 90, 89, 87, 20, 30, 15, 55, 45, 65, 50, 40, 20, 35, 45, 55, 65],
    'X2': [20, 25, 30, 35, 40, 50, 55, 60, 65, 70, 66, 78, 77, 55, 66, 25, 20, 35, 30, 45, 25, 35, 45, 60, 65],
    'Y': [60, 55, 65, 70, 62, 75, 80, 85, 90, 95, 77, 87, 10, 20, 30, 60, 50, 60, 65, 67, 35, 40 , 55, 60, 60],
}


# In[8]:


df = pd.DataFrame(data)
X = df[['X1', 'X2']]  # Variabel independen
X = sm.add_constant(X)  # Tambahkan konstanta
Y = df['Y']  # Variabel dependen

model = sm.OLS(Y, X).fit()  # Membuat model regresi OLS
summary = model.summary()  # Tampilkan ringkasan model
print(summary)


# In[10]:


# Scatter plot untuk melihat hubungan antara X1, X2, dan Y
fig = px.scatter_3d(df, x='X1', y='X2', z='Y', title='Scatter Plot 3D')

# Regresi berganda sebagai bidang
xx, yy = np.meshgrid(df['X1'], df['X2'])
zz = model.params['const'] + model.params['X1'] * xx + model.params['X2'] * yy
fig.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=0.8, colorscale='Viridis'))

fig.show()


# In[5]:





# In[ ]:




