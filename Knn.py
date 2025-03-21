import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv('iris.csv')

print(df.head())
print(df.info())

spl_len = df["SepalLengthCm"]
spl_wid = df["SepalWidthCm"]
ptl_len = df["PetalLengthCm"]
ptl_wid = df["PetalWidthCm"]

plt.figure(figsize=(10, 6))
sns.scatterplot(x=spl_len, y=spl_wid, hue=df["Species"], palette="Set1")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Sepal Width")


plt.figure(figsize=(10, 6))
sns.scatterplot(x=ptl_len, y=ptl_wid, hue=df["Species"], palette="Set1")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Petal Length vs Petal Width")

plt.show()

X = df.iloc[:, 1:5].to_numpy()  
y = df["Species"].to_numpy()    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

def dot_product(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

def sum_of_squares(a):
    result = 0
    for i in range(len(a)):
        result += a[i] ** 2
    return result

def cos_similarity(x, y):
    result = []
    for j in range(len(x)):
        dott = dot_product(x[j], y)
        mag1 = (sum_of_squares(x[j])) ** 0.5
        mag2 = (sum_of_squares(y)) ** 0.5
        similarity = dott / (mag1 * mag2)
        result.append(similarity)
    return result

def ecludian_distance(x, y):
    result = []
    for i in range(len(x)):
        temp = 0
        for j in range(len(y)):
            temp += (x[i][j] - y[j]) ** 2
        result.append(temp ** 0.5)
    return result

def manhattan_distance(x, y):
    result = []
    for i in range(len(x)):
        temp = 0
        for j in range(len(y)):
            temp += abs(x[i][j] - y[j])
        result.append(temp)
    return result

def knn(x_train, y_train, x_test, k, distance_metric):
    predictions = []
    
    for test_point in x_test:
        if distance_metric == "cosine_similarity":
            distances = cos_similarity(x_train, test_point)
            sorted_indices = np.argsort(-np.array(distances))  
        elif distance_metric == "euclidean_distance":
            distances = ecludian_distance(x_train, test_point)
            sorted_indices = np.argsort(np.array(distances))  
        elif distance_metric == "manhattan_distance":
            distances = manhattan_distance(x_train, test_point)
            sorted_indices = np.argsort(np.array(distances))  
       
        k_neighbors = sorted_indices[:k]  
        nearest_labels = [y_train[i] for i in k_neighbors]

        label_counts = {}
        for label in nearest_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        predicted_label = max(label_counts, key=label_counts.get)
        predictions.append(predicted_label)

    return np.array(predictions)

def accuracy(y_test, y_pred):
    correct = 0
    wrong = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            correct += 1
        else:
            wrong += 1
    return correct / len(y_test), correct,wrong

for metric in ["cosine_similarity", "euclidean_distance", "manhattan_distance"]:
    y_pred = knn(X_train, y_train, X_test, 5, metric)
    print(f"Accuracy with {metric}: {accuracy(y_test, y_pred)}")



