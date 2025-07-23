#Iris case study with KNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt


def IrisCaseStudy(Datapath):
    df=pd.read_csv(Datapath)

    print("Check the data")
    print(df.head())

    #check for null/missing values if any
    print(df.isnull())
    #no null values 

    x=df.drop(columns=['variety'])
    # Encode target variable 
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['variety']) 

    print("After encoding:")#to check
    print(y)

    scaler=StandardScaler()
    x_scale=scaler.fit_transform(x)

    #Split the dataset

    x_train,x_test,y_train,y_test=train_test_split(x_scale,y,test_size=0.3,random_state=42)

    accuracyscores=[]
    k_range=range(1,8)

    #use for loop for best value of k
    for k in k_range:
        model=KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        accuracy=accuracy_score(y_test,y_pred)
        print("value of k is: ",k)
        print("Accuracy score is:",accuracy)
        print()
        accuracyscores.append(accuracy)
    
    #Visualization
    plt.figure(figsize=(8,5))
    plt.plot(k_range,accuracyscores,marker='o',linestyle='--')
    plt.title("Accuracy vs K value")
    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(k_range)
    plt.show()

    #print the Best value of k
    best_k=k_range[accuracyscores.index(max(accuracyscores))]
    print("Best value of k is:",best_k)

    model=KNeighborsClassifier(n_neighbors=best_k)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)

    print("Final best accuracy is: ",accuracy*100)
    cm=confusion_matrix(y_test,y_pred)
    print(cm)
            

def main():
    IrisCaseStudy("iris.csv")

if __name__=="__main__":
    main()