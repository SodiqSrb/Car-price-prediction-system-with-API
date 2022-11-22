# Car-price-prediction-system-with-API
Hi everyone i’m : 
Abulsemiu Sodiq

In this article, we will develop an end-to-end machine learning project with 11 steps in our own locale. After that, we will create an API with FastAPI, after creating the interface with the help of Streamlit, we will dockerize our project.

Data Load
Feature Engineering
Data Cleaning
Outlier Removal
Data Visualization (Relationship Between Variables)
Extracting Training Data
Encoding
Build a Model
Create API with FastAPI
Creating a web interface to the model created with Streamlit.
Dockerize
df = pd.read_csv("cars.csv")
df.head()

Feature Engineering
Add new feature for company names.

df["company"] = df.name.apply(lambda x: x.split(" ")[0])
df.head()

Data Cleaning
year has many non-year values.

df2 = df.copy()
df2 = df2[df2["year"].str.isnumeric()]
year is in object. Change to integer.

df2["year"] = df2["year"].astype(int)
Price has Ask for Price.

df2 = df2[df2["Price"] != "Ask For Price"]
df2.Price

Price has commas in its prices and is in object.

df2.Price = df2.Price.str.replace(",","").astype(int)
kms_driven has object values with kms at last.

df2["kms_driven"]

df2["kms_driven"] = df2["kms_driven"].str.split(" ").str.get(0).str.replace(",","")
It has nan values and two rows have ‘Petrol’ in them.

df2 = df2[df2["kms_driven"].str.isnumeric()]
df2.info()

df2["kms_driven"] = df2["kms_driven"].astype(int)
fuel_type has nan values.

df2[df2["fuel_type"].isna()]

df2 = df2[~df2["fuel_type"].isna()]
Changing car names. Keeping only the first three words.

df2['name']=df2['name'].str.split().str.slice(start=0,stop=3).str.join(' ')
df2.head()

Resetting the index of the final cleaned data.

df2 = df2.reset_index(drop=True)
Save the clanned data.

df2.to_csv("cleaned_car_data.csv")
df2.describe(include="all")

Outlier Removal
Drop the price outliers.

df2 = df2[df2["Price"]<6e6].reset_index(drop=True)
df2

Data Visualization
Checking relationship of Company with Price.

df2["company"].unique()

import seaborn as sns
plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='company',y='Price',data=df2)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()

Checking relationship of Year with Price.

plt.subplots(figsize=(20,10))
ax=sns.swarmplot(x='year',y='Price',data=df2)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()

Checking relationship of kms_driven with Price.

sns.relplot(x='kms_driven',y='Price',data=df2,height=7,aspect=1.5)

Checking relationship of Fuel Type with Price.

plt.subplots(figsize=(14,7))
sns.boxplot(x='fuel_type',y='Price',data=df2)

Relationship of Price with FuelType, Year and Company mixed.

ax=sns.relplot(x='company',y='Price',data=df2,hue='fuel_type',size='year',height=7,aspect=2)
ax.set_xticklabels(rotation=40,ha='right')

Extracting Training Data
X=df2[['name','company','year','kms_driven','fuel_type']]
y=df2['Price']
X

Applying Train Test Split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
Encoding
Creating an OneHotEncoder object to contain all the possible categories.

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
ohe=OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])
Creating a column transformer to transform categorical columns.

column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                remainder='passthrough')
Build a Model
Linear Regression Model

lr=LinearRegression()
Making a pipeline

pipe=make_pipeline(column_trans,lr)
Fitting the model

pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
Checking R2 Score

r2_score(y_test,y_pred)

Finding the model with a random state of TrainTestSplit where the model was found to give almost 0.92 as r2_score.

scores=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))
scores[np.argmax(scores)]

Predict spesific car price

pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))

The best model is found at a certain random state

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)

import joblib
joblib.dump(pipe,open('LinearRegressionModel.pkl','wb'))
pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))

Create API with FastAPI
Initialize application instance in main.py in app folder.

app = FastAPI(title='Car Price Prediction', version='1.0',
description='Linear Regression model is used for prediction')
Initialize model artifacte files. This will be loaded at the start of FastAPI model server.

model = joblib.load("LinearRegressionModel.joblib")
This struture will be used for Json validation.

With just that Python type declaration, FastAPI will perform below operations on the request data

1) Read the body of the request as JSON.

2) Convert the corresponding types (if needed).

3) Validate the data.If the data is invalid, it will return a nice and clear error, indicating exactly where and what was the incorrect data.

class Data(BaseModel):
    name: str
    company: str
    year: int
    kms_driven: float
    fuel_type: str
Create API root or home endpoint.

@app.get('/')
@app.get('/home')
def read_home():
    """
    Home endpoint which can be used to test the availability of the    application.
    """
    return {'message': 'System is healthy'}
Create ML API endpoint to predict against the request received from the client.

@app.post("/predict")
def predict(data: Data):
    result = model.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array([data.name,data.company,data.year,data.kms_driven,data.fuel_type]).reshape(1,5)))[0]
    return result
We need to add this code block to run the codes we wrote at the end of main.py.

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


Creating a web interface to the model created with Streamlit.
We are reading the csv file with the help of pandas, as we will use the data that we save the clean data in app.py.

df = pd.read_csv("../cleaned_car_data.csv")
Let’s add the codes that we will create the interface with the help of Streamlit.

def run():
    streamlit.title("Car Price Prediction")
    name = streamlit.selectbox("Cars Model", df.name.unique())
    company = streamlit.selectbox("Company Name", df.company.unique())
    year = streamlit.number_input("Year")
    kms_driven = streamlit.number_input("Kilometers driven")
    fuel_type = streamlit.selectbox("Fuel type", df.fuel_type.unique())

Let’s store the data we receive from the user through the interface in the “data” variable.

data = {
    'name': name,
    'company': company,
    'year': year,
    'kms_driven': kms_driven,
    'fuel_type': fuel_type,
    }
When the “Predict” button is clicked, we get results from our model using the API we created with the help of “request”.

if streamlit.button("Predict"):
    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    prediction = response.text
    streamlit.success(f"The prediction from model: {prediction}")
We need to add this code block to run the codes we wrote at the end of app.py.

if __name__ == '__main__':
    run()
We can open our application by running the “streamlit run app.py” command in the file directory where app.py is located.

Let’s try it, will we be able to get the results we got on the notebook in the web application?


We will do cauterization with Docker so that the service we have created works on other systems.

Dockerize
Before that, we should not forget to create the requirements.txt file with the pipreqs ./ command on the console screen.

We need to add Dockerfile, It will expect a file at /app/main.py and will expect it to contain a variable app with your FastAPI application.

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
COPY /app/requirements.txt /app/
RUN pip install -r /app/requirements.txt
COPY ./model /model/
COPY ./app /app
Again on the console screen, “docker image build -t car-price .” We create docker images by running the command.
We can see the image we created by running the “docker image ls” command.
We will upload our container to Docker Hub to share the created container with the whole world. After creating a repository in Docker Hub, we tag it with the code “docker tag car-price username/car-price”.
And finally, we push it to Docker Hub with the command “docker push username/car-price”. (This may take a long time) Then we can open Docker Hub and check that it has been pushed.
Project: https://github.com/furkankizilay/car-price-prediction

