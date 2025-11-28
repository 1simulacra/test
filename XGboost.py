import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics, model_selection
import xgboost as xgb
from pylab import mpl
import numpy as np

# 设置中文显示字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

df = pd.read_csv('XGBoost/pythonProject8/boston.csv') #读取数据

y = df['MEDV'].values
x = df.drop('MEDV', axis=1).values
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

ss_X = preprocessing.RobustScaler()# 离散标准化处理
ss_Y = preprocessing.RobustScaler()
X_train_scaled = ss_X.fit_transform(X_train)
y_train_scaled = ss_Y.fit_transform(y_train.reshape(-1, 1))
X_validation_scaled = ss_X.transform(X_validation)
y_validation_scaled = ss_Y.transform(y_validation.reshape(-1, 1))

xgb_model = xgb.XGBRegressor(max_depth=3,
                             learning_rate=0.1,
                             n_estimators=100,
                             objective='reg:squarederror',
                             booster='gbtree',
                             random_state=0)
# 拟合
xgb_model.fit(X_train_scaled, y_train_scaled)
y_validation_pred = xgb_model.predict(X_validation_scaled) # 在测试集上进行预测

plt.figure(figsize=(14, 7))
plt.plot(range(y_validation_scaled.shape[0]), y_validation_scaled, color="blue", linewidth=1.5, linestyle="-")
plt.plot(range(y_validation_pred.shape[0]), y_validation_pred, color="red", linewidth=1.5, linestyle="-.")
plt.legend(['真实值', '预测值'])
plt.title("真实值与预测值比对图")
plt.show()  #显示图片

mse = np.mean((y_validation_scaled - y_validation_pred)**2)
print('均方误差：', mse)
