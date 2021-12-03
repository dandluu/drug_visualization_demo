import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from env import SQLALCHEMY_DATABASE_URL
from sqlalchemy import create_engine

st.set_page_config(
	layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    
)

st.title(""" 
Drug Clearance Target Data Visualization Demo
Dan Luu 2021
""")

st.write("Modified Dataset from Lombardo et al 2018")
st.write("""

This is a demo site to show Altair visualizations from either a csv or database of pharmokinetic properties of drug clearance and steady state volume of distribution against LogD at 7.4pH.
Part of this Challenege was to take a drug dataset and perform EDA and outlier removal on the data and then the data must be transformed to either a SQL or noSQL database. Then a machine learning model must be built from the database 
 to demonstrate a complete data science workflow applied to a pharmocology domain. As an additional and bonus task, 
 a web application was created to host the model and be used by other users to either retrain the model based on basic parameters or use the 'default' model to make predictions. 
The model is not functional and this is only a basic demonstration and example of what this challenege can do. More precise models are protected by the owner.

The **model state is NOT stored**, therefore you must click a button each time you would like to change a parameter or prediction

""")

st.write("""A Linear Regression model is created from the dataset and allows the user to set the random_state, training sample size, and use the model to make a prediction. 
However, this is only a demonstration, this model can be replaced with a more complicated model with additional parameters and features to train on. 
The database has not yet been cleansed of outliers and missing data.
 
 The Upload CSV Funcationality only works if you submit your file as a CSV with appropiate column titles:
 'drug_name', 'moka_logd74, 'vdss', 'CL', and 'ionstate'. Other columns in the CSV will be ignored. More functionality can be added in terms of visualing the data and model building. 
 """)

st.write("For more details on the demo challenge please see: [Github](https://github.com/dandluu/drug_visualization_demo) ")
st.write("Connect with me on [LinkedIn](https://www.linkedin.com/in/dandluu/) ")
st.write("Other Projects: [Craigstimate](craigstimate-301619.uc.r.appspot.com/) ")







# First Sidebar ---------------------------------------------------------------------------------

with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file, Please Rename Human VDSS to vdss and MoKa_logD7.4 to moka_logd74 in your columns", type=["csv"])
    #  Use Database instead

# Visualize ------------------------------------------------------------------------------------

def visualize(df):
    st.subheader('**2. Visualizing Dataset**')
    st.markdown('2.1 Scatterplots')
    
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

def make_box(df, col):
    
    box = alt.Chart(df).mark_boxplot(size=30).encode(
    alt.Y(col, scale=alt.Scale(
            type='log')),
    x='ionstate', color='ionstate').interactive()
    
    st_box = st.altair_chart(box, use_container_width=True)
    
    return st_box

def make_facet(df, col):

    facet = alt.Chart(df).mark_point().encode(
        alt.Y(col, scale=alt.Scale(
            type='log',  
            clamp=True
        )),
        x='moka_logd74',
        color='ionstate', tooltip=['drug_name','moka_logd74', 'vdss', 'ionstate', 'CL']
    ).properties(
        width=200,
        height=200
    ).facet(
        column='ionstate'
    ).interactive()

    st_facet = st.altair_chart(facet, use_container_width=True)
    
    return st_facet


def make_regression(df, y_var, degree_list):
    degree_list = degree_list
    base = alt.Chart(df).mark_circle().encode(
            alt.X("moka_logd74"), alt.Y(y_var, scale=alt.Scale(
            type='log')), color='ionstate'
    )

    polynomial_fit = [
        base.transform_regression(
            "moka_logd74", y_var, method="poly", order=order, as_=["moka_logd74", str(order)]
        )
        .mark_line()
        .transform_fold([str(order)], as_=["degree", y_var])
        .encode(alt.Color("degree:N")).interactive()
        for order in degree_list
    ]

    return st.altair_chart(alt.layer(base, *polynomial_fit), use_container_width=True)

# Model -------------------------------------------------------------------------------------------
# print(f'before global model :{type(model)}')

# model = None
    
def build_model(df, random_state=0, test_size=0.2):
    st.subheader('**3. Building a linear regression model from the data**')
    
    df = df.copy()

    df.dropna(how='any', inplace=True)

    x = np.array(df['moka_logd74']).reshape(-1,1)
    y = np.array(df['CL']).reshape(-1,1)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size, random_state=random_state)
    # random_state = 0 
    # test_size = 0.2
    # global model
    model = LinearRegression()
    # print(f'model in  :{type(model)}')
        
    model.fit(x_train, y_train)
        
    score = model.score(x_test, y_test)
    # print(type(model))
    print('Model created')
    
    st.write(f'Model was built successfully with an accuracry score of {score}')
    
    # print(f'prediction code model: {type(model)}')
    
    prediction = model.predict(np.array(logD).reshape(-1,1))
    st.sidebar.write(f'Predicted Clearance from LogD of {logD} : {round(prediction[0][0], 2)}')

    return model

#Function that runs visualizations and model building
def start():
    
    visualize(df)

    st.write('2.2 Box Plots')
    make_box(df, 'CL')
    make_box(df,'vdss')
    
    st.write(f'2.3 Ionstate Scatter')
    make_facet(df, 'CL')
    make_facet(df,'vdss')
    
    st.write('2.4 Polynomial Regression Fitting')
    cl_degree_list = [1, 2, 3, 5]
    vd_degree_list = [2, 3, 5]
    make_regression(df, 'CL', cl_degree_list) 
    make_regression(df, 'vdss', vd_degree_list)

    build_model(df, random_state, test_size)
    
    st.write("Made by Dan Luu 2021 with Streamlit's Framework")

# ------ START OF PAGE Loading -----------------------------------
st.subheader('**1. Dataset**')

st.sidebar.header('Adjust your parameters and retrain the model')
random_state = st.sidebar.number_input('Random State number', min_value=0)
test_size = st.sidebar.slider('Sample Test Size', min_value=0.2, max_value=1.0, step=0.1, )
st.sidebar.header('**Make a prediction**')    
logD = st.sidebar.slider('LogD score', min_value=-9.0, max_value=9.0, step=0.1)


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('1.1. Glimpse of dataset')
    
    st.write(df)
    print('Using Uploaded CSV')
    
    start()


else:
    st.info('Waiting for CSV file to be uploaded.')

    if st.button('Use Data from database with no outliers removed'):

        engine = create_engine(SQLALCHEMY_DATABASE_URL)

        df = pd.read_sql('javelin', engine, columns=['drug_name', 'vdss', 'CL', 'moka_logd74', 'ionstate'])
        engine.dispose()
        # print(engine)

        st.markdown('The dataset from the database is used an the example.')
        st.markdown('1.1. Glimpse of dataset')

        st.write(df)
        print('using database data')

        start()

    elif st.button('Use the cleaned data with some outliers removed'):
        df= pd.read_csv("Drug_cleaned_data.csv")

        st.markdown('1.1. Glimpse of dataset')

        st.write(df)
        print('Using cleaned data')
        
        start()


# Notes to add: 
# st.sidebar.button('Start over')
# open model from pickle open with, read as rb
    


    