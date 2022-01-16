# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 23:52:24 2022

@author: Mtime
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.sidebar.header('EDA: Cleveland Heart Disease, clean data') 
st.sidebar.subheader('Visualization using streamlit, Matplotlib and seaborn')

# Get the data
df0 = pd.read_csv(r'C:\Users\Mtime\OneDrive\Bureau\heart.csv')


def zero2median(df0):
    columns_with_zero = df.columns[(df==0).sum() > 0][1:-1]
    # Index(['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'], dtype='object')
    df[columns_with_zero]=df[columns_with_zero].replace(0,np.nan)
    for feature in columns_with_zero:
        df[feature].fillna(df[feature].median(),inplace=True)
    
    return df

# Start with cleaned dataframe (이미 전처리 되었음.)
df = df0  # zero2median(df0)


df['target'] = df['target'].apply(lambda x: 'HD' if x == 1 else 'noHD')

classes=df.target
noDB,DB=classes.value_counts()
st.sidebar.write('non-Heart-Disease(noHD):',noDB)
st.sidebar.write('Heart-Disease(HD):',DB)

# plotly batchart
df1 = df.groupby(["target"]).count().reset_index()
fig0 = px.bar(df1, 
              y=df.groupby(["target"]).size(),
              x="target", 
              width=350, height=300, 
              labels={'y':'Number'}, 
              color='target', 
              color_discrete_map=dict(noHD = 'green', HD = 'red'))
st.sidebar.write(fig0)
# pi-chart
fig1=px.pie(df, values=df.target.value_counts(), 
            width=350, height=300, 
            names=['noHD','HD'], color=['noHD','HD'], 
            color_discrete_map=dict(noHD = 'green', HD = 'red')) 
st.sidebar.write(fig1)

st.subheader(f'Clean Data Information: shape = {df.shape}')
st.subheader('features and classes')
st.write(f'##### features = {list(df.columns)[:-1]}')
st.write(f'##### classes = {pd.unique(classes)}')
st.write('***')
# Show the data as a table (you can also use st.write(df))
st.dataframe(df)
# Get statistics on the data
# st.write(df.describe())

# histogram with plotly
st.header("Histogram")
## multi-column layput
row0_1, row0_space2, row0_2 = st.beta_columns(
    (1, .1, 1))

with row0_1: 
    hist_x = st.selectbox("Select a feature", options=df.columns, index=df.columns.get_loc("age"))
        
with row0_2: 
    bar_mode = st.selectbox("Select barmode", ["relative", "group"], 0)

hist_bins = st.slider(label="Histogram bins", min_value=5, max_value=50, value=25, step=1, key='h1')
# hist_cats = df['target'].sort_values().unique()
hist_cats = df[hist_x].sort_values().unique()
hist_fig1 = px.histogram(df, x=hist_x, nbins=hist_bins, 
                         title="Histogram of " + hist_x,
                         template="plotly_white", 
                         color='target', 
                         barmode=bar_mode, 
                         color_discrete_map=dict(noHD = 'green', HD = 'red'),  
                         category_orders={hist_x: hist_cats}) 
st.write(hist_fig1)


# boxplots
st.header("Boxplot")
st.subheader("With a categorical variable - target [noHD, HD]")
## multi-column layput
row1_1, row1_space2, row1_2 = st.beta_columns(
    (1, .1, 1))

with row1_1: 
    box_x = st.selectbox("Boxplot variable", options=df.columns, index=df.columns.get_loc("age"))
        
with row1_2: 
    box_cat = st.selectbox("Categorical variable", ["target"], 0)

st.write("Hint - try comparison w.r.t Categories")
box_fig = px.box(df, x=box_cat, y=box_x, title="Box plot of " + box_x, color='target', 
                        color_discrete_map=dict(noHD = 'green', HD = 'red'), 
                        template="plotly_white", 
                        points="all") 
st.write(box_fig)
# violin

st.header("Violin Plot")
row2_1, row2_space2, row2_2 = st.beta_columns((1, .1, 1))

with row2_1: 
    corr_x = st.selectbox("violin - X variable", options=df.columns, index=df.columns.get_loc("age"))
        
with row2_2: 
    corr_y = st.selectbox("violin - Y variable", options=df.columns, index=df.columns.get_loc("cp"))


fig = px.violin(df, y=corr_y, 
                            color='target', 
                            color_discrete_map=dict(noHD = 'green', HD = 'red'), 
                            template="plotly_white", # can be 'outliers', or False
               )
st.write(fig)
###heatmap
st.subheader('Heatmap of selected parameters')
fig5 = plt.figure(figsize=(10,8))
hmap_params = st.multiselect("Select parameters to include on heatmap", options=list(df.columns), default=[p for p in df.columns if "target" not in p])
sns.heatmap(df[hmap_params].corr(), annot=True, vmin=-1, vmax=1, cmap='coolwarm')
st.pyplot(fig5)

st.header("Area Plot")
row2_1, row2_space2, row2_2 = st.beta_columns((1, .1, 1))

with row2_1: 
    area_x = st.selectbox("area - X variable", options=df.columns, index=df.columns.get_loc("age"))
        
with row2_2: 
    area_y = st.selectbox("area - Y variable", options=df.columns, index=df.columns.get_loc("cp"))


fig5 = px.area(df,area_x, y=area_y, 
                            color='target', 
                            color_discrete_map=dict(noHD = 'green', HD = 'red'), 
                            template="plotly_white", # can be 'outliers', or False
               )
st.write(fig5)
######ecdfPlot#######

st.header("Ecdf Plot" )
row2_1, row2_space2, row2_2 = st.beta_columns((1, .1, 1))

with row2_1: 
    ecdf_x = st.selectbox("ecdf - X variable", options=df.columns, index=df.columns.get_loc("age"))
        
with row2_2: 
    ecdf_y = st.selectbox("ecdf - Y variable", options=df.columns, index=df.columns.get_loc("cp"))


fig6 = px.ecdf(df,x=ecdf_x, y=ecdf_y, 
                            color='target', 
                            color_discrete_map=dict(noHD = 'green', HD = 'red'), 
                            template="plotly_white", # can be 'outliers', or False
               )
st.write(fig6)

########Strip########

st.header("Strip Plot" )
row2_1, row2_space2, row2_2 = st.beta_columns((1, .1, 1))

with row2_1: 
    Strip_x = st.selectbox("Strip - X variable", options=df.columns, index=df.columns.get_loc("age"))
        
with row2_2: 
    Strip_y = st.selectbox("Strip - Y variable", options=df.columns, index=df.columns.get_loc("cp"))


fig7 = px.strip(df,x=Strip_x, y=Strip_y, 
                            color='target', 
                            color_discrete_map=dict(noHD = 'green', HD = 'red'), 
                            template="plotly_white", # can be 'outliers', or False
               )
st.write(fig7)

#########Heatmap######
st.subheader('Heatmap of correlation')
fig8 = plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True, vmin=-1, vmax=1, cmap='coolwarm')
st.pyplot(fig8)
#st.write(fig8)

st.header("scatter and violin Plots" )
row2_1, row2_space2, row2_2 = st.beta_columns((1, .1, 1))

with row2_1: 
    Strip_x = st.selectbox("scatter - X variable", options=df.columns, index=df.columns.get_loc("age"))
        
with row2_2: 
    Strip_y = st.selectbox("scatter - Y variable", options=df.columns, index=df.columns.get_loc("cp"))


fig7 = px.scatter(df,x=Strip_x, y=Strip_y,marginal_y="violin",
           marginal_x="box", trendline="ols", template="simple_white", # can be 'outliers', or False
               )
st.write(fig7)

# Correlations
## multi-column layput

# st.write(fig4)

# correlation heatmap


#############################################################
