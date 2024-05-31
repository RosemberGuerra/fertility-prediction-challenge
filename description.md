# Description of Submission

## Overview

### Project Goal
The primary objective of this data challenge is to enhance the predictability of individual-level fertility behaviors using advanced data-driven methodologies. By improving our predictive capabilities, we aim to deepen our understanding of the factors influencing fertility decisions and trends.

### Challenges and Methodological Approach
One significant challenge encountered was the extensive amount of missing data within the provided datasets. Instead of employing imputation techniques, which could potentially introduce biases and inflate performance metrics, we opted for a more conservative approach by selecting variables with less than 20% missing data. This strategy prevents the model from learning from artificially created data, thereby maintaining the integrity and reliability of our predictive analysis.

### Model Selection and Evolution
#### Phase 1
In the initial phase, we employed the LightGBM algorithm for both variable selection and prediction. LightGBM is renowned for its ability to handle datasets with minor missing values directly within the algorithm, eliminating the need for preliminary data imputation. This feature, along with its robustness and accuracy across various conditions and time frames, makes LightGBM an ideal choice for our objectives.

During our first submission, we observed that the model performed well, indicating promising predictive accuracy. However, we recognized the potential for selection bias, where variables might have been chosen more by chance than for their actual predictive power. In subsequent iterations, we refined our approach to mitigate this bias, resulting in a more robust and reliable model.

#### Phase 2
For the next phase of the challenge, we plan to expand our methodology by incorporating additional algorithms like XGBoost and HGBoost. These models are chosen for their similar capabilities in handling missing data and for their proven effectiveness in generalizing predictive performance across different datasets and conditions.

## Data Cleaning 

### Process
- **Data Integration**: Combined training data (`PreFer_train_data.csv`) and background data (`PreFer_train_background_data.csv`) to create a comprehensive dataset. This merge ensures a richer set of features which could potentially improve model accuracy.
- **Filtering Data**: Selected only those records that match the `nomem_encr` identifiers in the non-null entries of the target dataset (`PreFer_train_outcome.csv`).
- **Handling Missing Data**: Columns with more than 20% missing values were excluded to maintain data quality and reliability.
- **Type Correction**: Utilized a codebook to assign the correct data types to variables, enhancing the consistency of the dataset for analysis.

### Rationale
The data cleaning steps were designed to ensure that the dataset was robust, with comprehensive coverage of features and minimal missing data. This process is crucial for maintaining the integrity of the modeling phase, particularly in handling large datasets which might contain inconsistencies.

## Variable Selection

### Process
- **Data Preparation**: The cleaned training dataset was merged with the outcome dataset to align the features with the target variable `new_child`.
- **Adding Noise Variables**: To benchmark the importance of each feature, two noise variables were introduced. `noise1` follows a normal distribution, and `noise2` follows a uniform distribution.
- **Model-Based Feature Importance**: A LightGBM classifier was utilized across multiple parameter settings to evaluate the importance of each feature relative to the noise variables. This was achieved through a simulation with a predefined number of iterations, ensuring robustness in the variable importance estimation.
- **Importance Calculation**: For each simulation, the model was trained, and the feature importances were recorded. Features were deemed significant if their importance exceeded that of both noise variables. This importance was then scaled relative to the sum of all importances to normalize across different model runs.
- **Frequency and Relevance of Selection**: Each feature's selection frequency and scaled importance were calculated across all simulations. Features frequently identified as important were retained for further analysis.

### Rationale
The selection of variables through model-based importance ensures that only the most predictive features are retained, reducing model complexity and potential overfitting. By comparing feature importance against noise variables, we can objectively assess whether a feature provides meaningful information or merely adds noise. This method is particularly effective in large datasets where distinguishing signal from noise is crucial. Additionally, using a grid of hyperparameters for LightGBM allows for a comprehensive evaluation under various modeling conditions, further validating the robustness of the selected features.

## Model Estimation

### Process
- **Data Preparation**: Merged the outcome dataset with the cleaned dataset containing the selected variables to form the final dataset for modeling.
- **Data Splitting**: The data was split into training and testing sets, with 30% of the data reserved for testing. This split helps validate the model on unseen data, ensuring that our model generalizes well beyond the training data.
- **Model Definition**: LightGBM, a gradient boosting framework that uses tree-based learning algorithms, was chosen for its effectiveness and efficiency with large data sets.
- **Hyperparameter Tuning**:
  - A grid search was conducted over a specified parameter grid including `num_leaves`, `max_depth`, `learning_rate`, and `n_estimators` to find the best model configuration.
  - Cross-validation with five folds was used during the grid search to ensure that the model's performance was robust across different subsets of the data.
  - The grid search was configured to refit the best model based on the F1 score, balancing the precision and recall of the model.
- **Model Evaluation**: The best model from the grid search was evaluated on the test set. Metrics such as accuracy, precision, recall, and F1 score were computed to assess the model's performance comprehensively.

### Variable Importance
We can observe in the following table the variable importance associated with the model we are using to predict fertility. One interesting result from our study is that the importance of variables presents an aggregation effect through surveys. For example, variables that present a higher importance come from the Family & Household, Economic Situation Assets, and Health surveys in that order.

| Unnamed: 0 | Importance | var_label | survey |
|------------|------------|-----------|--------|
| cs20m415 | 179 | Preload variable: Age respondent | Family & Household |
| brutohh_f_2020 | 145 | What is the year of birth of your mother? | Family & Household |
| cf20m128 | 143 | Do you currently have a partner? | Family & Household |
| cp20l193 | 138 | Do you think you will have [more] children in the future? | Family & Household |
| lftdhhh | 132 | Duration in seconds | Family & Household |
| cr20m120 | 132 | Duration in seconds | Economic Situation Assets |
| ca20g075 | 129 | Age respondent | Economic Situation Assets |
| cf20m009 | 112 | Age respondent | Economic Situation Income |
| netinc | 105 | preloaded variable: age | Health |
| cf20m397 | 102 | How much do you weigh, without clothes and shoes? | Health |
| ch20m259 | 101 | gynaecologist | Health |
| birthyear_bg | 99 | Duration in seconds | Health |
| ch20m017 | 98 | Often forget to put things back in their proper place. | Personality |
| cv20l303 | 91 | forgiving | Personality |
| cw20m576 | 82 | Duration in seconds | Personality |
| cs20m243 | 82 | It does not help a neighborhood if many people of foreign origin or descent move in. | Politics and Values |
| cw20m002 | 79 | Duration in seconds - part 3 | Politics and Values |
| sted_2020 | 79 | Do you speak Dutch with…your partner? | Religion and Ethnicity |
| nettohh_f_2020 | 74 | Duration in seconds | Religion and Ethnicity |
| burgstat_2020 | 71 | How often did you take a holiday within the Netherlands over the past 12 months? | Social Integration and Leisure |
| oplcat_2019 | 69 | average number of days per week that time is spent on: handwork | Social Integration and Leisure |
| cs20m102 | 68 | computer or laptop use, average number of hours per week: at work | Social Integration and Leisure |
| ci20m002 | 64 | Duration in seconds | Social Integration and Leisure |
| cv20l123 | 62 | average number of hours per week spent on: watching online films or TV programs | Social Integration and Leisure |
| werving | 58 | Respondent's year of birth | Work & Schooling |
| cp20l102 | 52 | Current income per month, based on values from the Core Questionnaire Income | Work & Schooling |
| ca20g082 | 51 | Year of birth [imputed by PreFer organisers] | Summary Background Variables |
| ch20m219 | 45 | Gross household income in Euros | Summary Background Variables |
| cf20m004 | 45 | Civil status | Summary Background Variables |
| cr20m093 | 42 | Net household income in Euros | Summary Background Variables |
| cs20m437 | 41 | Level of education in CBS (Statistics Netherlands) categories | Summary Background Variables |
| ch20m002 | 37 | Urban character of place of residence | Summary Background Variables |
| cp20l047 | 27 | Age of the household head | Background Variables |
| cs20m167 | 23 | Personal net monthly income in Euros | Background Variables |
| cf20m024 | 16 | From which recruitment wave the household originates | Background Variables |


### Rationale
The use of grid search and cross-validation ensures that the model is not only tuned to perform optimally in terms of prediction accuracy but is also stable and reliable across different data splits. This rigorous approach to model tuning and evaluation is crucial for developing a high-performing model that can be confidently used for making predictions.

## Conclusion
By continuously refining our approach and exploring various advanced machine learning techniques, we aim to build a model that not only predicts fertility behavior accurately but also adapts to and performs well under different analytical scenarios. Our strategy ensures that our findings are both scientifically robust and practically relevant.
