# Disaster-Response-Pipeline

## Table of Contents
* [Introduction](#Introduction)
* [Installation](#Installation)
* [Project Results](#Results)
* [Licensing, Authors, Acknowledgements](#License)

## Introduction<a name="Introduction"></a>
### Project Motivation 
There are many disaster related messages throughout various platforms such as social media. These messages would be very useful if they are well-categorized and sent to a proper disaster relief agency. In this project, I built a web application where an emergency worker can input a new message and get classification results in several categories. 

### Project Description 
The project aims to classify the type of the disaster messages collected from [Figure Eight](https://appen.com/) and service the inference based on a new input through a web application. To eleborate, I analyze the disaster response data (ETL data pipeline), and then build a classification model (ML pipeline) for an API that categories disaster events. Finally, I deploy a web app to provide the interface to classify the type of new disaster reponse from an user, as well as display visualizations of the data. Check below to see more details of each step. 

`ETL Pipeline` in [data/process_data.py](https://github.com/suhongkim/Disaster-Response-Pipeline/blob/main/data/process_data.py) 

- [x] Load and merge the two dataset: [disaster_messages.csv](https://github.com/suhongkim/Disaster-Response-Pipeline/blob/main/data/disaster_messages.csv) and [disaster_categories.csv](https://github.com/suhongkim/Disaster-Response-Pipeline/blob/main/data/disaster_categories.csv)
- [x] Clean the data
- [x] Save the data into an [SQLite database](https://github.com/suhongkim/Disaster-Response-Pipeline/blob/main/data/DisasterResponse.db)

`ML Pipeline` in [models/train_classifier.py](https://github.com/suhongkim/Disaster-Response-Pipeline/blob/main/model/train_classifier.py)

- [x] Load data from the SQLite database
- [x] Build a machine learning pipeline
- [x] Train and evaluate a model to find the best model using Grid-Search  
- [x] Export the final model as a pickle file

`Flask Web App` in [app/run.py](https://github.com/suhongkim/Disaster-Response-Pipeline/blob/main/app/run.py)

- [x] Visualize the data using Plotly
- [x] Deploy the web app 

## Installation<a name="Installation"></a>
### Environment Setup
All libraries are available in Anaconda distribution of Python 3.*. The used libraries are:
```
pandas
re
sys
json
sklearn
nltk
sqlalchemy
pickle
Flask
plotly
sqlite3
```

### How to Run 
- Run the following commands in the project's root directory to set up your database and model.
	- To run ETL pipeline that cleans data and stores in database  
	```
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```
	- To run ML pipeline that trains classifier and saves 
	```
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    ```

- Run the following command in the app's directory to run your web app. 
	```
    cd app
    python run.py ../data/DisasterResponse.db ../models/classifier.pkl
    ```

- Go to http://0.0.0.0:3001/
   
## Project Results<a name="Results"></a>
### Model Evaluation Report 
---|accuracy|precision|recall|f1-score|distribution-0|distribution-1
---|---|---|---|---|---|---
related|0.247517188694|0.61|0.25|0.1|0.7534377387318564|0.24656226126814362
request|0.837471352177|0.86|0.84|0.76|0.8372803666921314|0.1627196333078686
offer|0.994652406417|0.99|0.99|0.99|0.9946524064171123|0.0053475935828877
aid_related|0.590336134454|0.76|0.59|0.44|0.5901451489686784|0.40985485103132163
medical_help|0.924751718869|0.86|0.92|0.89|0.9247517188693659|0.07524828113063407
medical_products|0.950534759358|0.9|0.95|0.93|0.9505347593582888|0.04946524064171123
search_and_rescue|0.970206264324|0.94|0.97|0.96|0.9702062643239114|0.029793735676088617
security|0.982238349885|0.96|0.98|0.97|0.9822383498854087|0.01776165011459129
military|0.968487394958|0.94|0.97|0.95|0.9684873949579832|0.031512605042016806
water|0.937356760886|0.94|0.94|0.91|0.9371657754010695|0.06283422459893048
food|0.891520244461|0.9|0.89|0.84|0.8913292589763178|0.1086707410236822
shelter|0.908708938121|0.83|0.91|0.87|0.9087089381207029|0.09129106187929717
clothing|0.983766233766|0.97|0.98|0.98|0.9837662337662337|0.016233766233766232
money|0.974980901451|0.95|0.97|0.96|0.9749809014514896|0.025019098548510314
missing_people|0.989113827349|0.98|0.99|0.98|0.9891138273491215|0.010886172650878533
refugees|0.967150496562|0.94|0.97|0.95|0.9671504965622613|0.03284950343773873
death|0.96256684492|0.93|0.96|0.94|0.9625668449197861|0.0374331550802139
other_aid|0.866692131398|0.75|0.87|0.8|0.8666921313980137|0.13330786860198626
infrastructure_related|0.933728036669|0.87|0.93|0.9|0.9337280366692131|0.06627196333078686
transport|0.955118411001|0.91|0.96|0.93|0.9551184110007639|0.044881588999236055
buildings|0.95282658518|0.91|0.95|0.93|0.9528265851795263|0.047173414820473644
electricity|0.976699770817|0.95|0.98|0.97|0.9766997708174179|0.023300229182582125
tools|0.992933537051|0.99|0.99|0.99|0.9929335370511841|0.00706646294881589
hospitals|0.99025974026|0.98|0.99|0.99|0.9902597402597403|0.00974025974025974
shops|0.993888464477|0.99|0.99|0.99|0.9938884644766998|0.006111535523300229
aid_centers|0.987967914439|0.98|0.99|0.98|0.9879679144385026|0.012032085561497326
other_infrastructure|0.955118411001|0.91|0.96|0.93|0.9551184110007639|0.044881588999236055
weather_related|0.711611917494|0.79|0.71|0.59|0.7108479755538579|0.2891520244461421
floods|0.913101604278|0.92|0.91|0.87|0.9129106187929718|0.08708938120702826
storm|0.906990068755|0.82|0.91|0.86|0.9069900687547746|0.09300993124522536
fire|0.988731856379|0.98|0.99|0.98|0.9887318563789153|0.011268143621084798
earthquake|0.904125286478|0.91|0.9|0.86|0.9035523300229182|0.09644766997708175
cold|0.978036669213|0.96|0.98|0.97|0.9780366692131398|0.021963330786860198
other_weather|0.946333078686|0.9|0.95|0.92|0.9463330786860199|0.05366692131398014
direct_report|0.81550802139|0.85|0.82|0.73|0.8153170359052712|0.1846829640947288
avg|0.9100294663319874|0.9037142857142857|0.9099999999999999|0.8745714285714284|0.9244134017243263|0.07558659827567389


### Web App 
![screenshot of web app](web_screenshot.png)


## Licensing, Authors, Acknowledgements<a name="License"></a>
You can find the Licensing for the data and other descriptive information at the Figure Eight available [here](https://appen.com/). Also, some parts of the codes in this project are provided by Udacity Data Scientist Program. If you think that it is useful, please connect with me via [linkedIn-Suhong](https://www.linkedin.com/in/suhongkim/)