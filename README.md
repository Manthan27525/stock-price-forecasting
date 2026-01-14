# 🚀 Stocks Price Forecasting Project using LSTM

This repository showcases a complete **Machine Learning pipeline** for downloading, preprocessing, modeling, forecasting ,visualizing stocks prizes, evaluation, artifact management, and deployment-ready application code.  

---

## 📌 Project Highlights

- Modular and clean project structure  
- Reproducible ML workflow  
- Trained model and preprocessing artifacts stored separately  
- Clear separation between experimentation and production code  
- Ready for extension into real-world deployment  

---

## 📂 Repository Structure

```
├── artifacts/          # Trained models, scalers, and saved ML artifacts
├── notebooks/          # Jupyter notebooks for EDA and experiments
├── src/                # Source utilities and helper functions
│   └── plot_utils.py
├── app.py              # Application entry point (inference / UI)
├── model_trainer.py    # Model training and evaluation pipeline
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── .gitignore          # Ignored files and directories
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Manthan27525/stock-price-forecasting.git
cd stock-price-forecasting
```

### 2️⃣ Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training

Run the training pipeline using:

```bash
python model_trainer.py
```

This script will:
- Load and preprocess the dataset  
- Train the machine learning model  
- Evaluate model performance  
- Save trained models and scalers to the `artifacts/` directory  

---

## 🖥️ Running the Application

After training, start the application:

```bash
streamlit run  app.py
```

The application uses the saved artifacts for inference and visualization.

---

## 📊 Notebooks

The `notebooks/` directory includes a file with:
- Exploratory Data Analysis (EDA)
- Feature engineering experiments
- Model comparison and validation

> These notebooks are for research and experimentation and are not required for running the final application.

---

## 🛠️ Tech Stack

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  
- Jupyter Notebook
- Tensorflow
- Streamlit  

---

## 📁 Artifacts

All trained assets such as:
- Machine learning models  
- Scalers and encoders  

are saved inside the `artifacts/` directory for easy reuse during deployment.

---

## 📈 Future Enhancements

- Hyperparameter optimization  
- Model versioning  
- Logging & monitoring  
- Docker & cloud deployment  
- CI/CD integration  

---

## 👤 Author

**Manthan Singh**  
Machine Learning & Data Science Enthusiast  

---


⭐ If you found this project helpful, consider giving it a star!
