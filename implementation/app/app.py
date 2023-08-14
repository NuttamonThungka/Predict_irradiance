import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import pickle
import joblib
import numpy as np
from datetime import datetime
from PIL import Image
from datetime import datetime, timedelta
import time as time_lib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import gaussian_kde
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from PIL import Image


# Load your irradiance data DataFrame
# Replace this with your actual data loading code
FOLDER_TEST = '/Users/khunnoot/Desktop/predict_irradiance/implementation/data'
FOLDER_MODEL = '/Users/khunnoot/Desktop/predict_irradiance/trainning/model'
test = 'Synergy_testset.csv'
RF_model = 'RandomForest.joblib'
Synergy_test = pd.read_csv(os.path.join(FOLDER_TEST, test), parse_dates = ['Datetime'], index_col = ['Datetime'])
loaded_RF_model = joblib.load(os.path.join(FOLDER_MODEL,RF_model))
irradiance_df = Synergy_test.copy()
irradiance_df['RandomForest'] = loaded_RF_model.predict(Synergy_test)



st.write("""
# Predicting solar irradiance using satellite images

This product aims to provide ground irradiance estimation at Synergy Technology Co., Ltd. by using a cloud attenuation model that takes cloud cover index extracted from Himawari-8 satellite as an input.

""")


st.sidebar.title("Date period")

# Date range picker with calendar interface
start_date, end_date = st.sidebar.date_input("Date Range", [pd.to_datetime("2023-02-06"), pd.to_datetime("2023-06-30")])

# Time resolution selection based on available options
time_resolutions = [10, 30, 60]
time_resolution = st.sidebar.selectbox("Time Resolution (min)", time_resolutions)

# Checkbox for selecting models
st.sidebar.title("Parameter")
selected_models = []
for model_name in ['RandomForest','Iclr','CI0_cm','CI0_ov','CI1_cm','CI1_ov']:
    if st.sidebar.checkbox(f"Show {model_name}"):
        selected_models.append(model_name)

# Filter data based on user inputs
filtered_data = irradiance_df.loc[str(start_date) : str(end_date)]
if time_resolution == 30 :
    filtered_data = irradiance_df[(irradiance_df.index.minute == 30)|(irradiance_df.index.minute == 0)]
elif time_resolution == 60 :
    filtered_data = irradiance_df[(irradiance_df.index.minute == 0)]
else :
    filtered_data = filtered_data
    
filtered_data.sort_index(inplace = True)
filtered_data = filtered_data[selected_models]



# Plot time series
st.header("Predicted Irradiance Time Series in Synergy Technology Co., Ltd.")
st.line_chart(filtered_data)

st.write("""
## Chosen dataframe

### parameter
1. **RandomForest** is the predicted solar irradiance by using RandomForest model that the proposed model of the product.
2. **Iclr** is the clear-sky solar irradiance (the irradiance in clear-sky condition)
3. **CI** is the cloud index (CI = [0,1]) refer to the cloud opacity that can attenuate the irradiance
    - CI0 , CI1 are the CI(t), CI(t-1)  
        - CI0 = CI at  the current time
        - CI1 = CI at lag time 10 min 
    - CI_ov is the cloud index that extract from red-channel image from RGB 
    - CI_cm is the cloud index that extract from cloudmask image type
4. **Datetime** is time at  zone time UTC+7
""")

#show dataframe
filtered_data[selected_models]

# Allow user to download DataFrame as CSV
st.sidebar.title("Download Data")
if st.sidebar.button("Download Predicted Irradiance Data"):
    csv_file = filtered_data.to_csv()
    st.sidebar.download_button(label="Download CSV", data=csv_file, file_name="predicted_irradiance.csv")
    
st.write("""
# Data Analytic
## Error matric
Shown are error matric of all models. There are 5 models that cotain :

    - OLS is ordinary least squares or linear regression model
    - poly is polynomial regression model
    - yXGB is XGBoost model
    - yRF is RandomForest model
    - CNN is Convolutional Neural Network
""")




def eva_table(df, name_model, base = 'I',):
    data = []
    dfnoon = df[(df.index.hour > 10) & (df.index.hour <=14)]
    dfnonnoon = df[~((df.index.hour > 10) & (df.index.hour <=14))]
    for i in range(len(name_model)):
        
        MAE= mean_absolute_error(df[base],df[name_model[i]])
        RMSE = np.sqrt(mean_squared_error(df[base],df[name_model[i]]))
        noon_MAE = mean_absolute_error(dfnoon[base],dfnoon[name_model[i]])
        noon_RMSE = np.sqrt(mean_squared_error(dfnoon[base],dfnoon[name_model[i]]))
        non_noon_MAE = mean_absolute_error(dfnonnoon[base],dfnonnoon[name_model[i]])
        non_noon_RMSE = np.sqrt(mean_squared_error(dfnonnoon[base],dfnonnoon[name_model[i]]))
        
        data.append({'Overall MAE':MAE, 'Overall RMSE':RMSE, 'MAE at noon':noon_MAE, 'RMSE at noon':noon_RMSE,
                    'MAE at non-noon':non_noon_MAE, 'RMSE at non-noon':non_noon_RMSE})
    df = pd.DataFrame(data)
    df.index = name_model
    return df

def group_table(df, name_model, base = 'I', score = 'MAE', by = None):
    dfh = df[['site_name',base]].copy()
    if by == None:
        by = [dfh.index.hour]

    if score == 'MAE':
        for i in range(len(name_model)):
            dfh[name_model[i]] = abs(df[base]-df[name_model[i]])
        return dfh.groupby(by=by).mean().reset_index()
    else:
        for i in range(len(name_model)):
            dfh[name_model[i]] = abs(df[base]-df[name_model[i]])**2
        return np.sqrt(dfh.groupby(by=by).mean()).reset_index()
    
FOLDER_MODEL = '/Users/khunnoot/Desktop/predict_irradiance/trainning/model'
result_model = 'compare_model_result.csv'
sns.set(style="whitegrid")
compare = pd.read_csv(os.path.join(FOLDER_MODEL,result_model), parse_dates = ['Datetime'], index_col = ['Datetime'])


name_model = ['OLS','poly','yXGB', 'yRF', 'CNN']
eva_table(compare, name_model, base = 'I').round(3).iloc[:,[0,2,3,4,5]]

st.write("""
## Hourly MAE
""")

name_model = ['OLS','poly','yXGB', 'yRF', 'CNN']
q = group_table(compare, name_model, base = 'I').drop(columns=['I'])
q.rename(columns = {'Datetime':'Hour'}, inplace = True)
# q.set_index('Hour',inplace = True)


fig = px.bar(q, x="Hour", y=name_model, barmode='group', height=400)
st.plotly_chart(fig)



st.dataframe(q, use_container_width=True)
    