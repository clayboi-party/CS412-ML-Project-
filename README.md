# CS412-ML-Project
**Machine Learning Project Influencers**  
**Sarp Kaan Özdemir**

## 1. Overview of the Repository
This repository showcases a complete pipeline for **two core tasks** involving Instagram influencer data:

1. **Classification** of users into 10 broad categories—food, fashion, health & lifestyle, tech, travel, entertainment, mom & children, art, gaming, and sports.  
2. **Regression** to estimate the number of **likes** a future post may receive.

The project aims to highlight how combining **numeric/boolean features** (e.g., follower counts, verification status) with **text-based features** (e.g., user biography, post captions) can yield more accurate predictions. I decided on a Random Forest approach for both tasks because it handles heterogeneous data types well and is relatively straightforward to interpret.

### Key Files & Scripts
- **`main.py`**  
  Acts as the central hub for the entire pipeline—loading data, invoking the classification and regression routines, and finally generating predictions.

- **`load_and_preprocess_data()`**  
  - Loads all relevant dataset files:
    - The classification CSV (`train-classification.csv`), 
    - The primary JSONL (`training-dataset.jsonl`) containing profile and post info, 
    - A file listing usernames to test classification (`test-classification-round3.dat`),
    - And the JSONL containing posts for regression (`test-regression-round3.jsonl`).

- **`process_user_data_for_classification()`**  
  - Extracts each user’s numeric/boolean attributes (followers, following, post count, account settings).  
  - Consolidates textual data—biography, category name, and up to five post captions—into a single text string.  
  - Associates each user with a label (if available) so the classification model can train on consistent samples.

- **`train_classification_model_with_text()`**  
  - Uses a `ColumnTransformer` to apply TF-IDF to the combined text field and to standardize numeric features.  
  - Trains a Random Forest classifier on these combined features, thus leveraging both text-based signals and user metrics.

- **`extract_posts_data()`** and **`prepare_regression_features()`**  
  - Focus on post-level details for the regression model, extracting textual features (TF-IDF on captions), numeric info (comments count), and one-hot encoding media types (e.g., IMAGE, VIDEO).  
  - Construct the feature matrix and regression targets (the `like_count`), returning them to be used by a `RandomForestRegressor`.

### Data Files
- **`train-classification.csv`**: Lists usernames and their assigned classes (the ground-truth categories).  
- **`training-dataset.jsonl`**: A large JSONL file with each user’s profile data and a list of their posts.  
- **`test-classification-round3.dat`**: Contains the usernames on which we run final classification predictions.  
- **`test-regression-round3.jsonl`**: Holds posts for which we aim to predict like counts.

All model outputs get written to **`classification_output.json`** and **`regression_output.json`**, providing a direct mapping from username/post ID to predicted category or like count.

---

## 2. Methodology

1. **Data Preprocessing**  
   - We parse each user’s profile to collect essential attributes: follower/following counts, whether the account is private or business, and so on.  
   - We then create a **`text_data`** field combining the user’s biography, category name, and up to five post captions. This decision stems from the insight that short text snippets (like bios or captions) frequently contain strong indicators of a user’s content focus.

2. **Reasoning Behind This Design**  
   - **Balancing Structured and Unstructured Data**: Instagram accounts have both numeric signals (followers) and textual descriptions (bios, captions). A more holistic approach merges these to better capture user behavior and content theme.  
   - **Random Forest**: Since we have a mixture of text (transformed via TF-IDF) and numeric features, Random Forest models are efficient, handle different feature types gracefully, and are more robust to outliers than some simpler models.  
   - **Captions**: The content posted strongly correlates with the likes a user receives. Including these as text features for classification (to identify the user’s niche) and for regression (to help predict engagement) proved beneficial.

3. **Modeling**  
   - **Classification**:  
     - **TF-IDF** is applied to each user’s combined text.  
     - Numeric features are standardized to ensure fair weighting in the Random Forest.  
     - The final classification model outputs one of the 10 categories.  
   - **Regression**:  
     - Similar textual transformation occurs on post captions, focusing on relevant keywords.  
     - We combine that with each post’s comments count and media type to forecast `like_count`.  
     - Predicted values are rounded and stored for downstream analysis.

4. **Training and Evaluation**  
   - For classification, training uses the merged DataFrame of user-level features plus labeled categories. Evaluation can be done by splitting the data or by applying the model to test usernames.  
   - For regression, we train on historical posts (where actual `like_count` is known) and then apply the model to new posts in `test-regression-round3.jsonl`.  
   - The final outputs (`classification_output.json` and `regression_output.json`) can be compared against real-world outcomes if the true labels/like counts are available.

---

## 3. Results

- **Classification**  
  - Prior to text integration, the classifier primarily leveraged numeric/boolean fields, providing only a partial picture of the account’s focus.  
  - By incorporating **biography** and **caption** data, the model more accurately identifies a user’s niche. The presence of domain-specific words—like “recipe,” “travel,” “fitness,” or “gamer”—significantly enhances category prediction.

- **Regression**  
  - The regression model’s design remains consistent throughout: it uses post captions (TF-IDF), comments count, and media type.  
  - The underlying logic is that certain captions, media formats, and discussion levels (comments count) tend to correlate with higher (or lower) likes.  
  - The predicted like counts are written to `regression_output.json`, offering a practical estimate of engagement for future posts.

---

## 4. Team Contributions
This project was completed **solo** by **Sarp Kaan Özdemir**, covering all aspects of coding, data handling, modeling, and documentation.
