# Grammar Scoring Engine from Spoken Audio

This project presents a machine learning pipeline for predicting grammar proficiency scores from spoken English audio samples. The goal is to build a regression model that outputs a continuous grammar score (0–5) based on linguistic features extracted from transcribed speech.

## 📌 Overview

- **Input**: `.wav` audio files (45–60 seconds each)  
- **Output**: Continuous grammar score between 0 and 5  
- **Training samples**: 409  
- **Test samples**: 197

## 🧠 Approach

1. **Transcription**: Used Whisper (tiny model) to convert audio to text  
2. **Feature Extraction**: Used SpaCy to extract linguistic features (noun count, verb count, etc.)  
3. **Modeling**: Trained a RandomForestRegressor on extracted features  
4. **Evaluation**: Reported RMSE and Pearson correlation on validation set  
5. **Prediction Output**: Generated `submission.csv` with predicted scores

## 📊 Metrics

- **Validation RMSE**: *0.852397801613262*  
- **Pearson Correlation**: *0.0711472029458716*

## 📈 Evaluation Visuals

### 📊 Validation Predictions  
This scatter plot shows the relationship between the true grammar scores and the model's predicted scores on the validation set. A tighter diagonal pattern would indicate stronger predictive performance.

<img width="800" height="600" alt="validation_plot" src="https://github.com/user-attachments/assets/fe040b48-464e-4f32-9322-83a180e6c359" />

### 📌 Feature Importance  
This bar chart highlights which linguistic features contributed most to the model's predictions. It helps interpret which aspects of grammar (e.g., sentence count, verb usage) were most influential.

<img width="640" height="480" alt="feature_importance" src="https://github.com/user-attachments/assets/bc013d84-be9a-42a7-b688-eabd4eed18f9" />

## 📁 Files

- `grammar_scoring.ipynb`: Main notebook with code and documentation  
- `submission.csv`: Final predictions for test set

## ⚙️ Requirements

To run this notebook, install the following dependencies:

```bash
pip install openai-whisper spacy language_tool_python librosa scikit-learn matplotlib seaborn
python -m spacy download en_core_web_sm
