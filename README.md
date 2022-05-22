# Deploying an ML-Model on Heroku with FastAPI 
In this project, the goal was to develop an ML model for [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/census+income), implement data versioning using [DVC](https://dvc.org/), enable Continuous Integration (CI) with Github Actions and deploy the model via FastAPI on [Heroku](https://dashboard.heroku.com/apps).

## Dependencies
- DVC for data versioning
- AWS account for S3 bucket as remote data storage
- Github account to use Github Actions for CI
- Heroku account for CD
- Package dependencies are listed in [requirements](requirements.txt)

## Model Building
- Exploratory data analysis was performed in a [jupyter notebook](starter/starter/EDA_notebook.ipynb) to remove spaces from column names, handle missing values and save the cleaned dataset in the directory '/starter/data/'.
- Functions to aid in model training and validaion on slices of data were implemented in a [library](/starter/starter/ml/model.py).
- [Main script for training](starter/starter/training_main.py) was written for data splitting, preprocessing, model training and saving. The model and associated encoder are saved in the directory '/starter/model/'. The results of model validation on different categorical features in the data are saved in [slice_output](/starter/model/slice_output.txt).
- Details of the model training and performance are mentioned in [model card](/starter/model_card.md).
- [Tests](/starter/test_funcs/test_model_training.py) were developed for the model training library in order to make the CI robust. 
- The cleaned dataset, trained model and encoder are versioned using DVC and stored in remote Amazon S3 storage.
![DVC dag](/screenshots/dvcdag.png)

## Continuous Integration
- Github Actions was utilized for CI by using an appropriate [workflow](.github/workflows/python-app.yml) which installs dependencies, configures AWS credentials, sets up DVC, lints with flake8 and tests using pytest.
![Workflow](/screenshots/continuous_integration.png)

## API Creation
- [Inference API](/starter/starter/app_src/main.py) was created to serve inference of the developed model.
- [Tests](/starter/test_funcs/test_api.py) were developed for the API in order to ensure expected response.
- Example schema, for data types needed for model inference, was embedded in the Pydantic model so that its available in the API docs.
![Example](/screenshots/example.png)

## API Deployment
- Due to the recent [security issue](https://status.heroku.com/incidents/2413) of Heroku, deployment was done through Heroku git using Heroku CLI instead of Github. 
![Deployment](/screenshots/continuous_deloyment.png)
- Root of live API.
![Root](/screenshots/live_get.png)
- [Script](/starter/API_example.py) that uses requests module to test model inference via live API was developed.
![Live API Inference](/screenshots/live_post.png)