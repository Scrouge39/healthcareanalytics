import numpy as np
import pandas as pd
import streamlit as st
import chart_studio.plotly as py
import matplotlib.pyplot as plt
import plotly
import math 
import plotly.express as px
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go
import statsmodels.api as sm
import seaborn as sns
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, roc_curve


data=pd.read_csv("diabetes.csv")


st.set_page_config(
    page_title = 'Healthcare Analytics',
    page_icon = '✅',
    
)

#username password login page
logged = False
if not logged:
  
  username = st.sidebar.text_input("Please enter Your Username")
  pwd = st.sidebar.text_input("Please enter Your Password")


  if pwd:
    if username.lower() == "drsamarhajj" and (pwd) == "MS42":
      logged = True
      st.success('You are Logged In')
    else:
      st.warning('Invalid Credentials')



if logged:

  st.image(Image.open('diabetesprediction.png'),width= 1600)
  



  st.sidebar.title('App Navigation')


  navigation = st.sidebar.radio("Please Choose a Page ",["Introduction","Data Exploration","Dashboard","Predictive Model"])
  if navigation == "Introduction":
        st.image('healthcareml.png')
        st.write('# Main Benefits of Machine Learning and Predictive Models in Healthcare :\n * Improved diagnostics\n * High cost effectiveness\n * Enhanced operational efficiency\n * Decreased readmission rates\n * Personalized medical care')

        st.image('diabetes.jpg',width=700)
        st.write("# Diabetes is a disease that occurs when blood glucose, also called blood sugar, is too high.\n * Blood glucose is the main source of energy and comes from the food we eat. \n * Insulin, a hormone made by the pancreas, helps glucose from food get into cells to be used for energy.\n * Sometimes the body doesn’t make enough — or any — insulin or doesn’t use insulin well. Glucose then stays in the blood and doesn’t reach the cells")



  if navigation == "Data Exploration":
      st.image('diabetespic.png',width=700)


      st.header(" This App introduces you to the Dataset , showing relationship and correlation of features with outcome , and predicts if Patients might have Diabetes or Not based on entering certain Features.")
      st.subheader("**To begin, Let me Introduce You to the Dataset ** 👇")
      
      
      if st.checkbox("Show Data"):
          st.table(data)
      if st.checkbox("Data Explanation"):
         st.markdown("The following features have been provided to help us predict whether a person is diabetic or not: \n * Pregnancies: Number of times pregnant \n * Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test \n * BloodPressure: Diastolic blood pressure (mm Hg) \n * SkinThickness: Triceps skin fold thickness (mm) \n * Insulin: 2-Hour serum insulin (mu U/ml) \n * BMI: Body mass index (weight in kg/(height in m)2) \n * DiabetesPedigreeFunction: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history) \n * Age: Age (years)\n * Outcome: Class variable (0 if non-diabetic, 1 if diabetic)")

      if st.checkbox("Describe Data"):
         show= data.describe()
         show
      if st.checkbox("Show Data Size"):
         show= data.shape
         show

      if st.checkbox("Show Columns"):
          columns_show= sorted(data)
          st.table(columns_show)

      if st.checkbox("Select Columns"):
       column= st.selectbox("Please Select Column",
  ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetePedigreeFunction', 'Age',])
       if column == 'Pregnancies':
          data1=data['Pregnancies']
          data1
       if column == 'Glucose':
          data1=data['Glucose']
          data1
       if column == 'BloodPressure':
          data1=data['BloodPressure']
          data1
       if column == 'SkinThickness':
          data1=data['SkinThickness']
          data1
       if column == 'Insulin':
          data1=data['Insulin']
          data1
       if column == 'BMI':
          data1=data['BMI']
          data1
       if column=="DiabetePedigreeFunction":
          data1=data['DiabetePedigreeFunction']
          data1
       if column == 'Age':
          data1=data['Age']
          data1
       
      
  if navigation == "Dashboard":
      # corr = data.corr()
      # print(corr)
      # sns.heatmap(corr, 
      #    xticklabels=corr.columns, 
      #    yticklabels=corr.columns)
      
  
      fig, ax = plt.subplots()
      st.write('# Correlation between Different Features and the outcome')
      sns.heatmap(data.corr(), ax=ax)
      st.write(fig)

      st.markdown("### We can analyze through the heatmap, brighter colors indicate more correlation, the closer the correlation score is towards 1 , the more correlation there is.\n ### As we can see from the heatmap, glucose levels, age, BMI and number of pregnancies all have significant correlation with the outcome variable. We can also notice the correlation between pairs of features, like Age and pregnancies, or Insulin and skin thickness.")
      st.write('# Number of Diabetic (1) and Non-Diabetic people (0)')
      outcome_df= pd.DataFrame(
      data["Outcome"].dropna().value_counts()).reset_index()
      outcome_df = outcome_df.sort_values(by='index')
      fig = Figure()
      ax = fig.subplots()
      sns.barplot(x=outcome_df['Outcome'],
      y=outcome_df['Outcome'], color='Blue', ax=ax)
      ax.set_xlabel('Outcome')
      ax.set_ylabel('Count')
      st.pyplot(fig)
      st.write("### 0 = Non-diabetic = 268 cases")
      st.write("### 1 = Diabetic = 500 cases")

      data["Outcome"].dropna().value_counts().reset_index()
      outcome_df = outcome_df.sort_values(by='index')
      fig1 = Figure()
      ax1 = fig.subplots()
      sns.barplot(x=outcome_df['Outcome'],
      y=outcome_df['Outcome'], color='Blue', ax=ax)
      ax1.set_xlabel('Outcome')
      ax1.set_ylabel('Count')

       #pie chart that display BMI filtered by different age groups and lifestyle 
      st.write('# What is the BMI accross Age type & Lifestyle')
      st.sidebar.markdown('# Filter BMI by Age type or Lifestyle')
      chart_visual = st.sidebar.radio('Select what dimension you want',('Lifestyle','Age type'))

      df2 = data.groupby(['Lifestyle']).mean()['BMI']
      df3 = data.groupby(['Age type']).mean()['BMI']
      fig = go.Figure()

      if chart_visual =='Lifestyle':
          fig = go.Figure(data=[go.Pie(labels=df2.index, values=df2.values)])
          st.write(fig)


      elif chart_visual =='Age type':
           fig =go.Figure(data=[go.Pie(labels=df3.index, values=df3.values)])
           st.write(fig)

      

      

      
  if navigation == "Predictive Model":

#Step 1 : removing outliers
    def remove_outliers(df):
      Q1 = df.quantile(0.25)
      Q3 = df.quantile(0.75)
      IQR = Q3 - Q1
      return df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
#Step 2 : #replacing 0 values
    df = data
    df['Age'] = df['Age'].replace(0, df['Age'].median())
    df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].median())
    df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].median())
    df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
    df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())
    df['BMI'] = df['BMI'].replace(0, df['BMI'].median())
    df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].replace(0, df['DiabetesPedigreeFunction'].median())
 
    data_removed = remove_outliers(df)
#Machine Learning

#Step 3 :scaling 
    #scaling all the Features expect for outcome!

    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(df.drop(columns=['Outcome','Lifestyle','Age type']))
    y = df['Outcome']

#Step4: Dataframe

    #we re-put columns Xs in order in a DataFrame because after scaling the data looses its index!
    scaled_X = pd.DataFrame(scaled_X, columns = ['Pregnancies', 'Glucose', 'BloodPressure',"SkinThickness", 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    #Step 5 :training the model

    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.3, random_state=7)
    
    #Step6 fitting the model

    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    
    #Test accuracy
    number1 = clf.score(X_test, y_test) 
    #test AUC
    number2 = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    number_f = "{:,.2f}".format((number1*100))
    number_f2 = "{:,.2f}".format((number2*100))


    st.button((f'Model Accuracy: {number_f} %'))
    st.button((f'Model Perfomance: {number_f2} %'))

  #Features importance!! this calcualtes features importance , i've put hashtag because i only wanted the barchart display

   #importance = clf.coef_[0]
   #st.write(importance)
   #for i,v in enumerate(importance):
     #st.write(('Feature: %0d, Score: %.5f' % (i,v)))

    st.write('# Features Importance')
    features=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetePedigreeFunction', 'Age']
    fig1= go.Figure()
    st.write((fig1.add_trace(go.Bar(x=features, y=[1.4709,4.0711,0.3256,0.9806,-0.0977,2.2988,1.6701,0.7966],marker_color='crimson',
                name='expenses'))))




   


    import pickle

    # pickle.dump(clf, open(r'clf_model.sav', 'wb'))

    # /////////////////////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    Pregnancies = (st.text_input("Enter Pregnancies"))
    Glucose =  (st.text_input('Enter Glucose Levels'))
    BloodPressure = (st.text_input('Enter BloodPressure'))
    SkinThickness= st.slider('SkinThickness',7,99)
    Insulin= st.slider('Insulin rate',15,850)
    BMI = st.slider('BMI', 18, 70)
    DiabetesPedigreeFunction = (st.text_input('Enter DiabetesPedigreeFunction Number'))
    Age = (st.text_input('Enter Age'))
    
    
    
    
    if Pregnancies:
      Pregnancies = int (Pregnancies)
    if Glucose:
      Glucose = int(Glucose)
    if BloodPressure:
      BloodPressure = int(BloodPressure)

    if Pregnancies and Glucose and BloodPressure and SkinThickness and Insulin and BMI and DiabetesPedigreeFunction and Age:

      predicted_val = clf.predict([[Pregnancies, Glucose, BloodPressure,SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
      st.write(predicted_val)
