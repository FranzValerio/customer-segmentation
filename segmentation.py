import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

file_path = 'C:/Users/Francisco Valerio/Desktop/customer-segmentation/data/Train.csv'

data_train = pd.read_csv(file_path)

data_train.head()

categorical_columns = ['Ever_Married', 'Graduated', 'Profession', 'Var_1']

for column in categorical_columns:

    mode_val = data_train[column].mode()[0]

    data_train[column] = data_train[column].fillna(mode_val)

numerical_columns = ['Age', 'Work_Experience', 'Family_Size']

for column in numerical_columns:

    mean_val = data_train[column].mean()

    data_train[column]= data_train[column].fillna(mean_val)

categorical_columns = ['Ever_Married', 'Graduated', 'Profession', 'Var_1']

for column in categorical_columns:

    mode_val = data_train[column].mode()[0]

    data_train[column] = data_train[column].fillna(mode_val)

numerical_columns = ['Age', 'Work_Experience', 'Family_Size']

for column in numerical_columns:

    mean_val = data_train[column].mean()

    data_train[column]= data_train[column].fillna(mean_val)


data_train = data_train.drop(columns=['Var_1'])

categorical_variables = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Segmentation']

encoder = LabelEncoder()

for col in categorical_variables:

    if col in data_train.columns:

        data_train[col] = encoder.fit_transform(data_train[col])

    else:

        print(f"Column '{col}' not found in dataframe.")

standard_scaler = StandardScaler()

data_train[numerical_columns] = standard_scaler.fit_transform(data_train[numerical_columns])

sse = []

for k in range(1,20):

    kmeans = KMeans(n_clusters=k, random_state=42)

    kmeans.fit(data_train[numerical_columns])

    sse.append(kmeans.inertia_)

optimal_k = 4

kmeans = KMeans(n_clusters=optimal_k, random_state = 42)

kmeans.fit(data_train[numerical_columns])

data_train['Cluster'] = kmeans.labels_

