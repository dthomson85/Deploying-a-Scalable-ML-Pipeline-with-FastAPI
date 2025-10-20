# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a Logistic Regression binary classification model implemented using scikit-learn (sklearn.linear_model.LogisticRegression). It is a linear model that calculates a weighted sum of the input features and uses a sigmoid function to output a probability between 0 and 1, which is then mapped to a class prediction.

## Intended Use
The primary intended use of this model is to predict whether an individual's income exceeds $50,000 per year based on demographic and employment data from the US Census. It is designed as a component within a larger MLOps pipeline project to demonstrate skills in data processing, model training, and API deployment. This model should not be used to make real-world decisions about individual credit, employment, or housing.

## Training Data
The model was trained on the "Adult Census Income" dataset from the UCI Machine Learning Repository, containing 32,561 entries. The training data consisted of a random 80% split of this dataset. Preprocessing steps included one-hot encoding for categorical features and standard scaling for numerical features.

## Evaluation Data
The model was evaluated on a held-out test set consisting of the remaining 20% of the "Adult Census Income" dataset. This data was not used during the training process. The same preprocessing steps (encoding and scaling) were applied to the test data using the statistics learned from the training data.

## Metrics
The model's performance was evaluated using three key metrics for binary classification: Precision, Recall, and the F1-score (which is an F-beta score where beta=1.0, creating a harmonic mean of Precision and Recall).

The model's performance on the evaluation data is as follows:

Precision: `0.7518`

Recall: `0.6149`

F1-score: `0.6765`

## Ethical Considerations
The analysis performed on data slices (saved in slice_output.txt) shows that the model's performance is not uniform across all demographic groups. Performance metrics vary significantly for different values of categorical features such as race, sex, and native-country. This indicates the presence of bias in the model's predictions, which likely reflects societal biases present in the training data. Using this model for any real-world decision-making could amplify and perpetuate existing inequalities, leading to unfair outcomes for under-represented groups.

## Caveats and Recommendations
This model serves as a baseline for an educational project and is not intended for production use. The performance metrics indicate reasonable but not exceptional predictive power. The identified biases are a significant concern; before any real-world deployment, further work would be required, including bias mitigation techniques and a thorough fairness audit. It is recommended that the performance on specific demographic slices be the primary focus for future improvement.