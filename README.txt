WATTWISE - ELECTRICITY CONSUMPTION PREDICTION

Files:
- data/electricity_consumption.csv: 600 synthetic practice records
- notebook/Electricity_Consumption_Prediction.ipynb: guided workflow notebook
- train_model.py: trains and compares three regression models
- app.py: Streamlit application
- requirements.txt: Python dependencies

Local run:
1. pip install -r requirements.txt
2. python train_model.py
3. streamlit run app.py

Online deployment:
Upload the contents of this folder to GitHub. On Streamlit Community Cloud, select the repository, branch main and app.py as the main file.

Dataset honesty:
The dataset is synthetically generated for learning and demonstration. It is not real smart-meter or utility-company data. Predictions are estimates, not actual meter readings or guaranteed bills.

Features: temperature, humidity, occupants, AC hours, other appliance hours, lighting hours, weekend indicator.
Target: Daily_Consumption_kWh.
