# Breast Cancer Detection 2.0

This is a Machine Learning web application built with Flask that predicts whether a breast tumor is cancerous (Malignant) or not (Benign) based on 30 different cell nucleus features.

## 🚀 Features
- **Machine Learning Model**: Uses a Random Forest Classifier trained on the Wisconsin Breast Cancer Dataset.
- **Web Interface**: Clean, modern UI built with Bootstrap 5.
- **Real-time Validation**: Validates user inputs on the frontend before submitting.
- **Auto-fill**: Quickly test the model by pasting a CSV row from the dataset.
- **Confidence Score**: Provides a probability score for the prediction.

## 📁 Project Structure
- `app.py`: The main Flask application server.
- `train_model.py`: Script to train the Machine Learning model and generate the pipeline.
- `model.joblib`: The trained model (StandardScaler + RandomForestClassifier).
- `data/`: Directory containing the dataset (`breast-cancer.csv`).
- `templates/`: HTML templates for the web interface (`index.html`).
- `static/`: Static assets such as images used in the UI.

## 🛠️ Setup Instructions

### 1. Prerequisites
Ensure you have Python installed on your system.

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies:
```bash
python -m venv venv
```

### 3. Activate the Virtual Environment
- **Windows**:
  ```bash
  .\venv\Scripts\activate
  ```
- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## 💻 Running the Application
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## 🧠 Retraining the Model (Optional)
If you wish to retrain the model with new data:
1. Ensure your dataset is located at `data/breast-cancer.csv`.
2. Run the training script:
   ```bash
   python train_model.py
   ```
3. A new `model.joblib` file will be generated. Restart the Flask app to use the new model.
