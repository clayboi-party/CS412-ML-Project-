# CS412-ML-Project-
CS412 Machine Learning Project Influencers / Sarp Kaan Özdemir


## 1. Overview of the Repository
This repository contains a Python-based project that performs **two core tasks**:

1. **Classification** of Instagram profiles into 10 categories (food, fashion, health & lifestyle, tech, travel, entertainment, mom & children, art, gaming, sports).  
2. **Regression** to predict the **like counts** of social media posts.

### Key Files & Scripts
- **`main.py`**  
  Orchestrates data loading, model training, and prediction for both classification and regression.

- **`load_and_preprocess_data()`**  
  Loads dataset files (training classification CSV, training JSONL data, test usernames, and test regression data).

- **`process_user_data_for_classification()`**  
  Extracts numeric and textual features (biography, category name, and post captions) for the classification task.

- **`train_classification_model_with_text()`**  
  Builds a pipeline using TF-IDF for text data and trains a Random Forest classifier.

- **`extract_posts_data()`** and **`prepare_regression_features()`**  
  Collect and transform post-level data for the regression model, including TF-IDF features for captions and numeric features (comments count, media type).

### Data Files
- **`train-classification.csv`**: Contains usernames and their respective class labels.  
- **`training-dataset.jsonl`**: Main JSONL dataset with user profiles and posts.  
- **`test-classification-round3.dat`**: List of usernames for classification testing.  
- **`test-regression-round3.jsonl`**: Contains new posts for which we predict like counts.

All outputs are saved as **`classification_output.json`** and **`regression_output.json`**.

---

## 2. Methodology
1. **Data Preprocessing**  
   - Profile-level numeric and boolean features are extracted (follower count, following count, etc.).  
   - Biography, category name, and up to five post captions are concatenated for each user (classification).  
   - Post-level features (caption TF-IDF, comments count, media type) are extracted for regression.

2. **Feature Engineering**  
   - TF-IDF transforms textual data (bio, captions).  
   - Numeric features are standardized.  
   - One-hot encoding is used for categorical fields like `media_type`.

3. **Modeling**  
   - **Classification**: A `RandomForestClassifier` is trained on both numeric features and TF-IDF vectors.  
   - **Regression**: A `RandomForestRegressor` predicts `like_count` using TF-IDF on post captions plus numeric fields.

4. **Evaluation & Predictions**  
   - Classification predictions are generated for unseen usernames.  
   - Regression predictions output estimated like counts for new posts.

---

## 3. Results
- **Classification**  
  - Before adding text features, classification relied on numeric/boolean fields alone.  
  - After integrating biography and captions, performance (in a hypothetical x/10 scale) improved from **4/10** to **7/10**, showing stronger ability to distinguish among the 10 content categories.

- **Regression**  
  - The regression approach remained mostly consistent, yielding modest improvement when tuned or tested on larger data (from a hypothetical **6/10** to **6.5–7/10**).  
  - Predicted like counts are saved in `regression_output.json`.

> *Note: These scores are illustrative placeholders, as the actual performance depends on real evaluation metrics.*

---

## 4. Team Contributions
This project was completed **solo** by **Sarp Kaan Özdemir**. All code, data processing, and model development were done individually.
