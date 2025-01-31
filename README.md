# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - To install XGBOOST
        `pip install xgboost`
        
    - Tp update plotly, old version does not have Heatmap plot.
        `pip install plotly --upgrade`
        
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python ./app/run.py`

3. Go to http://0.0.0.0:3001/
