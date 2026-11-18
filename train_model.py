from pathlib import Path
import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
root=Path(__file__).resolve().parent
df=pd.read_csv(root/"data"/"electricity_consumption.csv")
features=["Temperature_C","Humidity_Percent","Occupants","AC_Hours","Appliance_Hours","Lighting_Hours","Weekend"]
X,y=df[features],df["Daily_Consumption_kWh"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
models={"Linear Regression":LinearRegression(),"Decision Tree":DecisionTreeRegressor(max_depth=6,random_state=42),"Random Forest":RandomForestRegressor(n_estimators=160,random_state=42)}
rows=[]; best=None; best_rmse=float("inf")
for name,m in models.items():
    m.fit(X_train,y_train); p=m.predict(X_test); mse=mean_squared_error(y_test,p); rmse=np.sqrt(mse)
    rows.append({"Model":name,"MAE":mean_absolute_error(y_test,p),"MSE":mse,"RMSE":rmse,"R2":r2_score(y_test,p)})
    if rmse<best_rmse: best,best_rmse=m,rmse; best_name=name
print(pd.DataFrame(rows).round(3).to_string(index=False))
out=root/"model"; out.mkdir(exist_ok=True); joblib.dump(best,out/"electricity_model.pkl")
print("Selected:",best_name,"| Saved model/electricity_model.pkl")
