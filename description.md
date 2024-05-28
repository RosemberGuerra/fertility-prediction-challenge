# Description of submission

## Data Processing

### Loading Data
The initial data loading involves three primary datasets:
1. **Target Data (`PreFer_train_outcome.csv`)**: Contains outcome variables and is used as the primary dataset for merging and analysis.
2. **Training Data (`PreFer_train_data.csv`)**: Includes features linked to the outcomes, loaded for potential feature engineering (commented out in the provided snippet).
3. **Background Data (`PreFer_train_background_data.csv`)**: Contains additional contextual information about the subjects.

### Cleaning and Preprocessing
#### Target Data
- Loaded from the `PreFer_train_outcome.csv` file.
- All rows containing any missing values are removed to ensure the quality and completeness of the data used in further analysis.

#### Background Data
- Loaded from the `PreFer_train_background_data.csv` file.
- Filtered to include only records that match the `nomem_encr` identifiers found in the cleaned target data. This ensures consistency and relevance in the combined dataset.
- Further filtered to retain records from the specific wave `202012`, aligning the data with a specific time frame for analysis.

### Merging Data
- The cleaned target data and the filtered background data are merged on the `nomem_encr` field using a left join. This enriches the target dataset with contextual background data relevant to each record.

### Data Type Standardization
To ensure consistency and facilitate analysis, various fields are converted to appropriate data types:
- Categorical conversions are applied to fields such as `positie`, `lftdcat`, `aantalhh`, `aantalki`, `partner`, and many others to reflect their non-numeric nature.
- Integer conversions are applied to fields such as `brutoink`, `nettoink`, `age_imp`, and others that represent numeric values, ensuring they are treated correctly in any computational context.

### Saving the Cleaned Data
- The fully processed and merged dataset is saved to `cleaned_background_data.csv`, making it available for further analysis or model training without additional preprocessing steps.

### Proportion of Missing Values
- The proportion of missing values is calculated for each column in the merged dataset, allowing for a quick assessment of data completeness post-processing.

### Summary
The processing steps taken ensure that the data is clean, consistent, and specifically tailored to the needs of the project, focusing on relevant time frames and observations. This preprocessing foundation supports reliable and robust analysis or modeling in subsequent stages.

## Variable Selection

### Overview
Variable selection was performed to identify the most relevant features for predicting the target variable `new_child` from the dataset. This process helps in reducing model complexity and improving interpretability.

### Methodology
1. **Data Preparation**:
   - Data was loaded from `train_na_20.csv`, which presumably has been preprocessed to handle missing values up to a certain threshold (20% as suggested by the filename).
   - Initial data exploration was conducted to understand the data structure and types.

2. **Exclusion of Non-Numeric Variables**:
   - All columns of type `object` were identified and listed for exclusion from the feature set, as they are non-numeric and not directly usable in LightGBM without encoding.
   - Specific variables `new_child` and `nomem_encr` were also manually added to the exclusion list due to their nature (identifier and target variable).

3. **Feature Engineering**:
   - Two noise variables, `noice1` (normal distribution) and `noice2` (uniform distribution), were added to the dataset to later assess the baseline importance of features. These noise features serve as a benchmark for determining the significance of real features.

4. **Model Training**:
   - A LightGBM classifier was trained using the numeric and cleaned feature set. LightGBM is chosen for its efficiency with large datasets and ability to handle various types of data.

5. **Feature Importance Evaluation**:
   - The importance of each feature was calculated using LightGBM's built-in feature importance metric, which measures the increase in the model's prediction error after permuting the feature.
   - Features were sorted based on their importance, and a threshold was established based on the importance values of the noise features to filter out less relevant features.

6. **Feature Selection**:
   - Only features with an importance greater than that of either noise feature were retained. This method ensures that selected features have a significant impact on the model's predictive power.
   - The selected features, along with the identifier `nomem_encr`, were saved to `variable_lightGBM_importance.csv`.

### Results
- The final list of important variables was compared against another model's feature importance list (from a Random Forest model) to validate consistency and robustness across different model types.
- The datasets `outcome_l_GBM.csv` and `cleaned_l_GBM.csv` containing the outcome and cleaned feature sets, respectively, were saved for further use in modeling or analysis.

### Conclusion
The variable selection process effectively narrowed down the feature set to those most impactful, improving the efficiency and potentially the performance of subsequent models. This approach not only enhances model simplicity but also aids in better understanding the driving factors behind the prediction of `new_child`.
