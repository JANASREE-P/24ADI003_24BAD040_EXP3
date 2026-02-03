print("JANASREE 24BAD040")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv("C:\\Users\\janas\\Downloads\\auto-mpg.csv")
df['horsepower']=pd.to_numeric(df['horsepower'],errors='coerce')
df=df.dropna()

X,y=df[['horsepower']],df['mpg']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)

for d in [2,3,4]:
    p=PolynomialFeatures(d)
    sc=StandardScaler()
    Xtrp=sc.fit_transform(p.fit_transform(Xtr))
    Xtep=sc.transform(p.transform(Xte))
    ypred=LinearRegression().fit(Xtrp,ytr).predict(Xtep)
    print(d,mean_squared_error(yte,ypred),np.sqrt(mean_squared_error(yte,ypred)),r2_score(yte,ypred))

    ypred=Ridge(1).fit(Xtrp,ytr).predict(Xtep)
    print("Ridge",r2_score(yte,ypred))

xr=pd.DataFrame(np.linspace(X.min(),X.max(),300),columns=['horsepower'])
plt.scatter(X,y,s=10)

for d in [2,3,4]:
    p=PolynomialFeatures(d)
    Xp=StandardScaler().fit_transform(p.fit_transform(X))
    plt.plot(xr,LinearRegression().fit(Xp,y).predict(StandardScaler().fit_transform(p.transform(xr))))

plt.show()
