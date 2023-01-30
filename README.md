# Variational Inference for Neyman-Scott Processes
## Instructions
### Dependency
- Anaconda (Python >= 3.8)
### Data
The data for retweets can be downloaded from [Google Drive Link](https://drive.google.com/drive/folders/0BwqmV0EcoUc8UklIR1BKV25YR1U?resourcekey=0-OrlU87jyc1m-dVMmY5aC4w&usp=sharing), provided by Hongyuan Mei from the [Neural Hawkes Process](https://github.com/hongyuanmei/neurawkes).

The data for earthquakes and homicides can be downloaded from [Google Drive Link](https://drive.google.com/drive/folders/1ELuYM9qIj2hoSzJcYAs9UklcwO2WdTNu?usp=sharing).

The synthetic dataset can be downloaded from [Google Drive Link](https://drive.google.com/drive/folders/1zFXdvVeCHCtu9xnhRI_F1odhnNGdIrxS?usp=share_link).
### Train
The modules should be added to PYTHONPATH first, e.g.,
```
export PYTHONPATH='./'
```
Training of 1-hidden for earthquakes with USAPs
```
python earthquake_weibull_1_hidden_general_virtual.py
```
Traning of 1-hidden for earthquakes with UNSPs
```
python earthquake_weibull_1_hidden_nsp_virtual.py
```
Training of 2-hidden for earthquakes with USAPs
```
python earthquake_weibull_2_hidden_general_virtual.py
```
Traning of 2-hidden for earthquakes with UNSPs
```
python earthquake_weibull_2_hidden_nsp_virtual.py
```
You can do the same things for retweets and homicides.

### Prediction
predict with 1-hidden MCMC
```
python earthquake_weibull_1_hidden_posterior_prediction.py
```
predict with 1-hidden UNSPs
```
python earthquake_weibull_1_hidden_virtual_nsp_prediction.py
```
predict with 1-hidden USAPs
```
python earthquake_weibull_1_hidden_virtual_general_prediction.py
```
predict with 2-hidden MCMC
```
python earthquake_weibull_2_hidden_posterior_prediction.py
```
predict with 2-hidden UNSPs
```
python earthquake_weibull_2_hidden_virtual_nsp_prediction.py
```
predict with 2-hidden USAPs
```
python earthquake_weibull_2_hidden_virtual_general_prediction.py
```
You can do the same things for retweets and homicides.
## Disclaimer
This site provides applications using data that has been modified for use from its original source, www.cityofchicago.org, the official website of the City of Chicago.  The City of Chicago makes no claims as to the content, accuracy, timeliness, or completeness of any of the data provided at this site.  The data provided at this site is subject to change at any time.  It is understood that the data provided at this site is being used at one’s own risk.
