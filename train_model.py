import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    save_model,
    performance_on_categorical_slice
)

print("Loading data...")
data = pd.read_csv("data/census.csv")
data.columns = [col.strip() for col in data.columns]

print("Splitting data...")
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

print("Processing training data...")
X_train, y_train, encoder, lb, scaler = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

print("Processing test data...")
X_test, y_test, _, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
    scaler=scaler
)

print("Training model...")
model = train_model(X_train, y_train)

print("Saving model and processors...")
save_model(model, "model/model.pkl")
save_model(encoder, "model/encoder.pkl")
save_model(lb, "model/lb.pkl")
save_model(scaler, "model/scaler.pkl")
print("Model and processors saved successfully.")

print("Running inference on test data...")
preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)
print("\nOverall Model Performance:")
print(
    f"Precision: {precision:.4f} | "
    f"Recall: {recall:.4f} | "
    f"F1 (F-beta): {fbeta:.4f}"
)

print("\nComputing performance on categorical slices...")
with open("slice_output.txt", "w") as f:
    for feature in cat_features:
        f.write(f"\n--- Performance for feature: {feature} ---\n")
        for cls in sorted(test[feature].unique()):
            p, r, fb = performance_on_categorical_slice(
                test,
                feature,
                cls,
                cat_features,
                "salary",
                encoder,
                lb,
                model,
                scaler
            )
            f.write(f"  Value: {cls}\n")
            f.write(
                f"    Precision: {p:.4f} | "
                f"Recall: {r:.4f} | "
                f"F1: {fb:.4f}\n"
            )

print("Slice performance metrics saved to slice_output.txt")
