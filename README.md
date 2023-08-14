# Predicting solar irradiance using satellite images

This product aims to provide ground irradiance estimation at Synergy Technology Co., Ltd. by using a cloud attenuation model that takes cloud cover index extracted from Himawari-8 satellite as an input.
<p align="center">
  <img src="https://github.com/NuttamonThungka/Predict_irradiance/assets/113121308/3dc3441f-f308-4e45-b666-9ae9fa957884" width="800" height="400" />
</p>

## Web app

[Predicting solar irradiance using satellite images](http://192.168.1.68:8501/)
<p align="center">
  <img src="https://github.com/NuttamonThungka/Predict_irradiance/assets/113121308/844f631a-2840-4d1a-95fb-5134eef8038a" width="900" height="500" />
</p>




This repository is composed of the following folders
- **trainning** are utilized for downloading data, cleaning it, and generating datasets. And contains training code for the Cloud Attenuation model. There are three groups in total: Regression, Tree-based, and CNNs models.
- **Implementation** is used for implement code with contain app to deploy the deshboard.
