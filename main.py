import warnings
warnings.simplefilter("ignore")
import pandas as pd
from sklearn.linear_model import (RANSACRegressor,HuberRegressor,LinearRegression)

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__=="__main__":
    dataset =pd.read_csv("felicidad.csv")
    #print(dataset.head(5))

    x=dataset.drop(["country","score"], axis=1 )
    y=dataset[["score"]]

    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1,)
    estimadores={
        "SVR" : SVR(gamma="auto",C=1.0,epsilon=0.1),
        "RANSAC" : RANSACRegressor(),
        "HUBER" : HuberRegressor(epsilon=1.35),
        "LINEAR" : LinearRegression()
    }

    for name, estimador in estimadores.items():
        estimador.fit(X_train,y_train)
        """
        predictions=estimador.predict(X_test)
        print("-"*20)
        print(name)
        print("MSE" , mean_squared_error(y_test,predictions))
        """
        """
        predictions=estimador.predict(pd.DataFrame([[118,4,4,1,0,0,0,1,1,1]]))
        print(predictions)
        """

        predictions=estimador.predict(pd.DataFrame([[12,7.1,6.9,1.1,1.4,0.7,0.5,0.2,0.1,2.8]]))
        print(predictions)
       

        
