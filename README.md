# CS412-ML-Project
**Machine Learning Project Influencers**  
**Sarp Kaan Özdemir**

## 1. Overview of the Repository
This repository demonstrates how we classify Instagram profiles into 10 distinct categories (e.g., food, fashion, tech, etc.) and also predict the like counts of individual posts. The primary goal is to showcase how incorporating textual data—like user biographies and captions—improves performance beyond simply using numeric/boolean features.

### Key Files & Scripts
- **`main.py`**  
  Central script that coordinates data loading, training, and prediction for both classification and regression.

- **`load_and_preprocess_data()`**  
  Retrieves dataset files (classification CSV, JSONL data, test usernames, and regression test data).

- **`process_user_data_for_classification()`**  
  Gathers each user’s numeric/boolean attributes and combines their textual data (biography, category, and captions).

- **`train_classification_model_with_text()`**  
  Employs TF-IDF for text features, then trains a Random Forest for the 10-category classification.

- **`extract_posts_data()`** and **`prepare_regression_features()`**  
  Assemble post-level information (captions, comments, media type) for the regression model, including text vectorization.

### Data Files
- **`train-classification.csv`**: Usernames and their assigned classes.  
- **`training-dataset.jsonl`**: Comprehensive JSONL file with profile and post details.  
- **`test-classification-round3.dat`**: Usernames for classification testing.  
- **`test-regression-round3.jsonl`**: Posts for which like counts need to be predicted.

All final predictions are saved to **`classification_output.json`** and **`regression_output.json`**.

---

## 2. Methodology
1. **Data Preprocessing**  
   - Extract numeric attributes (follower/following/post counts, verification status, etc.).  
   - Merge biography, category, and up to five post captions for classification inputs.  
   - Isolate key post features (caption, comments count, media type) for the regression task.

2. **Reasoning Behind Design Choices**  
   - **Textual Focus**: Bios and captions often contain keywords indicative of a user’s domain (food vs. art vs. gaming).  
   - **Variety of Features**: Combining numeric, boolean, and text data gives the model a more rounded perspective.

3. **Modeling**  
   - **Classification**: Uses TF-IDF for text transformation and a Random Forest to handle the varied nature of numeric/boolean data alongside text.  
   - **Regression**: Leverages similar text processing on captions, plus relevant numerical fields (e.g., comments count).

4. **Evaluation & Predictions**  
   - Classification outputs each username’s predicted category.  
   - Regression outputs estimated like counts for new/unseen posts.

---

## 3. Results
- **Classification**  
  - Numeric-only features gave limited insight into content types.  
  - After including textual data (e.g., user biography, captions), classification became noticeably better at separating distinct categories.

- **Regression**  
  - The approach to predicting like counts remained consistent, focusing on captions and a few numerical indicators.  
  - Results are stored in `regression_output.json`.

---

## 4. Team Contributions
This project was completed **solo** by **Sarp Kaan Özdemir**, covering all aspects of coding, data handling, modeling, and documentation.
