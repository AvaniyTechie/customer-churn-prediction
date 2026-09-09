\# 🤖 AI Customer Churn Intelligence



An end-to-end \*\*Customer Churn Prediction and Retention Intelligence\*\* application that combines \*\*Machine Learning, model explainability, and Generative AI\*\* to identify customers at risk of churn and generate customer-specific retention recommendations.



🔗 \*\*Live Application:\*\* \[Streamlit App](https://customer-churn-prediction-i8hmxylwrgpva6zgbiezdt.streamlit.app/)



\---



\## 📌 Project Overview



Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave allows businesses to take proactive retention actions.



This project builds an interactive machine learning application that:



\* Predicts customer churn probability

\* Classifies customers into Low, Medium, and High risk

\* Explains why the model made its prediction

\* Identifies factors pushing the prediction toward churn or staying

\* Uses Generative AI to convert model insights into business-focused retention recommendations

\* Provides an interactive Streamlit interface for real-time analysis



The project follows an end-to-end pipeline:



\*\*Customer Input → Preprocessing → Machine Learning Prediction → Feature Contributions → Risk Assessment → Generative AI Recommendations\*\*



\---



\## 🎯 Objectives



The main objectives of this project are:



1\. Predict whether a customer is likely to churn.

2\. Estimate the probability of churn.

3\. Identify the strongest factors influencing the prediction.

4\. Provide interpretable model-based insights.

5\. Generate personalized retention strategies using Generative AI.

6\. Deploy the solution as an interactive web application.



\---



\## 📊 Dataset



The project uses a telecom customer churn dataset containing customer demographics, account information, services, billing information, and churn status.



\### Dataset characteristics



\* \*\*Rows:\*\* 7,043 customers

\* \*\*Features:\*\* 21 columns

\* \*\*Target:\*\* `Churn`



Important features include:



\* Gender

\* Senior Citizen

\* Partner

\* Dependents

\* Tenure

\* Phone Service

\* Multiple Lines

\* Internet Service

\* Online Security

\* Online Backup

\* Device Protection

\* Tech Support

\* Streaming TV

\* Streaming Movies

\* Contract

\* Paperless Billing

\* Payment Method

\* Monthly Charges

\* Total Charges



\---



\## 🧹 Data Preprocessing



The preprocessing pipeline includes:



\* Handling missing values

\* Separating numerical and categorical features

\* Encoding categorical variables using One-Hot Encoding

\* Scaling numerical features

\* Splitting the dataset into training and testing data

\* Using a `ColumnTransformer` to apply preprocessing consistently



The preprocessing and model are combined into a single \*\*Scikit-learn Pipeline\*\*, ensuring that the same transformations are applied during prediction.



\---



\## 🤖 Machine Learning Models



Two classification models were evaluated:



\### Logistic Regression



Logistic Regression was selected as the primary model because it provides:



\* Churn probability estimates

\* Interpretable coefficients

\* Clear feature-level contributions

\* Efficient prediction



\### Random Forest



Random Forest was evaluated as a second model to compare predictive performance with Logistic Regression.



\### Model Results



| Model               | Accuracy |

| ------------------- | -------: |

| Logistic Regression |   79.10% |

| Random Forest       |   80.38% |



The Logistic Regression model was selected for the final application because its coefficients allow customer-level feature contribution analysis.



\---



\## 📈 Model Evaluation



The models were evaluated using:



\* Accuracy

\* Confusion Matrix

\* Precision

\* Recall

\* F1-score

\* ROC-AUC



The Logistic Regression model achieved an ROC-AUC of approximately \*\*0.836\*\*.



Because customer churn datasets are often imbalanced, performance was evaluated using more than accuracy alone.



\---



\## 🔍 Model Explainability



One of the key features of this project is \*\*customer-level model explainability\*\*.



After preprocessing the customer's information, the application calculates:



```text

Feature Contribution = Transformed Feature Value × Logistic Regression Coefficient

```



These contributions indicate how individual features influence the model's churn score.



\### Positive Contribution



A positive contribution pushes the model toward the churn class.



\### Negative Contribution



A negative contribution pushes the model toward the stay class.



The application displays:



\* Top factors increasing churn risk

\* Top factors reducing churn risk

\* A contribution visualization

\* Individual model contribution values



This makes the prediction more transparent than simply displaying:



> "Customer will churn."



\---



\## 🎯 Churn Risk Classification



The predicted probability is converted into three business-friendly risk levels:



| Churn Probability | Risk Level     |

| ----------------: | -------------- |

|           `< 30%` | 🟢 Low Risk    |

|      `30% – <60%` | 🟡 Medium Risk |

|           `≥ 60%` | 🔴 High Risk   |



The system also displays the underlying machine learning prediction.



\---



\## 🧠 Generative AI Integration



The project integrates \*\*Google Gemini\*\* to transform machine learning outputs into business-oriented retention recommendations.



The application sends Gemini:



\* Customer profile

\* Churn probability

\* Risk level

\* ML prediction

\* Top churn-driving factors

\* Top protective factors



Gemini then generates four sections:



\### 1. Churn Risk Explanation



Explains why the customer received the current risk assessment.



\### 2. Key Risk Drivers



Lists the strongest model-derived factors contributing toward churn.



\### 3. Recommended Retention Actions



Provides practical business actions based only on the available customer information.



\### 4. Personalized Retention Strategy



Creates a concise customer-specific retention strategy.



The Generative AI layer is instructed \*\*not to invent customer information or claim that model factors directly cause churn\*\*.



\---



\## 🏗️ System Architecture



```text

&#x20;                 Customer Information

&#x20;                         │

&#x20;                         ▼

&#x20;                   Streamlit UI

&#x20;                         │

&#x20;                         ▼

&#x20;               Preprocessing Pipeline

&#x20;                         │

&#x20;                         ▼

&#x20;                Logistic Regression

&#x20;                         │

&#x20;            ┌────────────┴────────────┐

&#x20;            ▼                         ▼

&#x20;     Churn Probability          ML Prediction

&#x20;            │                         │

&#x20;            └────────────┬────────────┘

&#x20;                         ▼

&#x20;                 Risk Classification

&#x20;                         │

&#x20;                         ▼

&#x20;               Feature Contributions

&#x20;                         │

&#x20;             ┌───────────┴───────────┐

&#x20;             ▼                       ▼

&#x20;       Churn Drivers            Stay Drivers

&#x20;             │                       │

&#x20;             └───────────┬───────────┘

&#x20;                         ▼

&#x20;                  Customer Context

&#x20;                         │

&#x20;                         ▼

&#x20;                    Google Gemini

&#x20;                         │

&#x20;                         ▼

&#x20;             AI Retention Recommendations

```



\---



\## 🖥️ Application Features



\### Customer Input



The application allows users to enter:



\* Demographic information

\* Service subscriptions

\* Contract information

\* Billing information

\* Payment method

\* Tenure

\* Monthly charges

\* Total charges



\### Churn Risk Dashboard



The application displays:



\* Churn probability

\* Risk level

\* ML prediction

\* Probability progress bar



\### Customer Profile



Important customer information is summarized for quick interpretation.



\### Model Contribution Dashboard



The application visualizes the strongest factors affecting the prediction.



\### AI Retention Recommendations



Gemini generates contextual recommendations based on the actual customer profile and model insights.



\---



\## 🛠️ Tech Stack



\### Programming



\* Python



\### Data Analysis



\* Pandas

\* NumPy



\### Machine Learning



\* Scikit-learn

\* Logistic Regression

\* Random Forest



\### Visualization



\* Matplotlib

\* Streamlit



\### Generative AI



\* Google Gemini API



\### Model Persistence



\* Joblib



\### Deployment



\* Streamlit Community Cloud



\### Version Control



\* Git

\* GitHub



\---



\## 📁 Project Structure



```text

customer-churn-prediction/

│

├── app.py

├── churn\_model.pkl

├── requirements.txt

├── .gitignore

├── README.md

└── .streamlit/

&#x20;   └── secrets.toml   # Local only - not committed

```



\---



\## 🔐 API Key Security



The Gemini API key is stored using Streamlit Secrets.



The key is accessed through:



```python

st.secrets\["GEMINI\_API\_KEY"]

```



The secrets file is excluded from Git using `.gitignore`.



\*\*API keys and other credentials should never be committed to the repository.\*\*



\---



\## 🚀 Run the Project Locally



\### 1. Clone the repository



```bash

git clone https://github.com/AvaniyTechie/customer-churn-prediction.git

```



\### 2. Navigate into the project



```bash

cd customer-churn-prediction

```



\### 3. Install dependencies



```bash

pip install -r requirements.txt

```



\### 4. Configure Gemini API



Create:



```text

.streamlit/secrets.toml

```



and add:



```toml

GEMINI\_API\_KEY = "your\_api\_key\_here"

```



\### 5. Run Streamlit



```bash

streamlit run app.py

```



\---



\## 🌐 Live Demo



The application is deployed using Streamlit Community Cloud.



🔗 \*\*Live Demo:\*\*

https://customer-churn-prediction-i8hmxylwrgpva6zgbiezdt.streamlit.app/



\---



\## 💡 Business Value



This application demonstrates how machine learning can move beyond prediction into \*\*actionable customer intelligence\*\*.



Instead of only identifying:



> "This customer has a high probability of churn."



the system provides:



> \*\*Why the model predicts churn → Which factors matter most → What practical retention actions could be considered.\*\*



This can help businesses prioritize retention efforts and focus attention on customers with higher predicted churn risk.



\---



\## ⚠️ Limitations



\* Model predictions are probabilistic and not guaranteed outcomes.

\* Feature contributions explain the Logistic Regression model rather than proving causal relationships.

\* Retention recommendations are AI-generated decision support and should be validated against actual business policies.

\* The model is based on historical customer data and may not generalize perfectly to other telecom businesses.

\* Business-specific pricing, offers, and policies are not included in the model.



\---



\## 🔮 Future Improvements



Potential improvements include:



\* SHAP-based explainability

\* Model calibration

\* Advanced ensemble models

\* Cost-sensitive learning for churn detection

\* Customer lifetime value prediction

\* Churn probability monitoring over time

\* Automated retention campaign integration

\* Customer segmentation

\* A/B testing of retention strategies

\* Business ROI estimation for retention actions



\---



\## 👩‍💻 Author



\*\*Avaniy V\*\*



CSE – Artificial Intelligence \& Machine Learning



Interested in:



\* Data Analytics

\* Machine Learning

\* Artificial Intelligence

\* Business Intelligence

\* Customer Analytics



\---



⭐ If you find this project useful, consider giving the repository a star.



