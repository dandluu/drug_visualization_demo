import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# from config import DBConfig
from env import SQLALCHEMY_DATABASE_URL
from sqlalchemy import create_engine
# from sqlalchemy.orm import scoped_session, sessionmaker



st.title(""" 
Drug Clearance Target Data - Javelin Assesement
""")
st.write("""Linear Regression model is just an example, this model can be replaced with a more complicated
 one with the addition of additional parameters and features to train on. The database has not yet been cleansed of outliers and missing data.
 The Upload CSV Funcationality has not yet been tested yet, but it should work if you submit your file as a CSV with column titles properly named.
 'drug_name', 'moke_logd74, 'vdss', 'CL', and 'ionstate'. 
 """)
st.write("Made by Dan Luu 2021 using Streamlit's Framework")


# First Sidebar ---------------------------------------------------------------------------------

with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file, Please Rename Human VDSS to vdss and MoKa_logD7.4 to moka_logd74 in your columns", type=["csv"])
    #  Use Database instead

# Visualizae ------------------------------------------------------------------------------------

def visualize(df):
    st.subheader('2. Visualizing Dataset')
    vdss_plot = alt.Chart(df).mark_point().encode(
        alt.Y('vdss', scale=alt.Scale(
            type='log', clamp=True
        )),
        x='moka_logd74', color='ionstate', 
        tooltip=['drug_name','moka_logd74', 'vdss', 'ionstate']
    ).interactive()

    cl_plot = alt.Chart(df).mark_point().encode(
        alt.Y('CL', scale=alt.Scale(
            type='log', clamp=True
        )),
        x='moka_logd74', color='ionstate', 
        tooltip=['drug_name','moka_logd74', 'vdss', 'ionstate']
    ).interactive()

    streamlit_cl = st.altair_chart(cl_plot, use_container_width=True)
    streamlit_vdss = st.altair_chart(vdss_plot, use_container_width=True)

    return streamlit_vdss, streamlit_cl


# Model -------------------------------------------------------------------------------------------
# print(f'before global model :{type(model)}')

# model = None

    
def build_model(df, random_state=0, test_size=0.2):
    st.subheader('3. Building a linear regression model from the data')
    
    df = df.copy()

    df.dropna(how='any', inplace=True)
    
    # global model
    # model = LinearRegression()

    x = np.array(df['moka_logd74']).reshape(-1,1)
    y = np.array(df['CL']).reshape(-1,1)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size, random_state=random_state)
    # random_state = 0 
    # test_size = 0.2
    # global model
    model = LinearRegression()
    print(f'model in  :{type(model)}')
        
    train = model.fit(x_train, y_train)
        
    score = model.score(x_test, y_test)
    # print(type(model))
    st.write(f'Model was built successfully with an accuracry score of {score}')
    
    
    
    # print(f'prediction code model: {type(model)}')
    
    prediction = model.predict(np.array(logD).reshape(-1,1))
    st.sidebar.write(f'Predicted Clearance from LogD of {logD} : {prediction[0]}')


    return model

    # def predict(logD):
    #     if model:
    #         print(type(model))

    #         prediction = model.predict(np.array(logD).reshape(-1,1))
        
    #         st.sidebar.write(f'Predicted Clearance from LogD of {logd} : {prediction[0]}')
    #     else:
    #         pass



st.subheader('1. Dataset')
# def main()
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    visualize(df)

    st.sidebar.header('Adjust your parameters and retrain the model')
    random_state = st.sidebar.number_input('Random State number', min_value=0)
    test_size = st.sidebar.slider('Sample Test Size', min_value=0.2, max_value=1.0, step=0.1, )
    st.sidebar.header('3. Making a prediction')    
    logD = st.sidebar.slider('LogD score', min_value=-9.0, max_value=9.0, step=0.1)
    
    visualize(df)
    
    model = build_model(df1, random_state, test_size)
    # model = build_model(df)
    # print(type(model))


else:
    st.info('Waiting for CSV file to be uploaded.')
    st.sidebar.header('2. Adjust your parameters and train the model')
    random_state = st.sidebar.number_input('Random State number', min_value=0)
    test_size = st.sidebar.slider('Sample Test Size', min_value=0.2, max_value=1.0, step=0.1)
    st.sidebar.header('3. Making a prediction')    
    logD = st.sidebar.slider('LogD score', min_value=-9.0, max_value=9.0, step=0.1)
    df1 = None
    
    if st.button('Use Data from Database instead or Retrain your model'):

        engine = create_engine(SQLALCHEMY_DATABASE_URL)
        ##### NEED TO FIX ENGINE ENV VARIABLES #####

        df = pd.read_sql('javelin', engine, columns=['drug_name', 'vdss', 'CL', 'moka_logd74', 'ionstate'])
        engine.dispose()
        # print(engine)

        st.markdown('The dataset from the database is used an the example.')
        st.write(df.head(5))
        st.write(f'Size of the data has {df.shape[0]} records')

        
        visualize(df)

        df1 = df.copy()


        model = build_model(df1, random_state, test_size)
        # dump model
        # print(f'built model?? {type(model)}')

    # open model from pickle open with, read as rb
    

    