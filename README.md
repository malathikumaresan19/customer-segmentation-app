# 🛍️ Customer Segmentation using Machine Learning

A K-Means clustering project that segments customers based on behavioral and
purchasing attributes (age, income, spending score) to support targeted
marketing and improve retention.

**Tools:** Python · Scikit-learn · Pandas · Seaborn · Streamlit

## 🔍 What this project does
- Performs exploratory data analysis (EDA) on customer data
- Scales features and determines the optimal number of clusters using the
  **Elbow Method** and **Silhouette Score**
- Fits a **K-Means** model to segment customers into distinct groups
- Profiles each segment (avg. age, income, spending score, size) to generate
  **actionable marketing insights**
- Ships as an interactive **Streamlit dashboard** so anyone (recruiters,
  stakeholders) can explore the clusters live, upload their own data, and
  download the segmented results

## 📁 Project structure
```
customer_segmentation/
├── app.py                     # Streamlit dashboard
├── segmentation_analysis.py   # End-to-end ML pipeline (EDA -> clustering -> insights)
├── requirements.txt
├── data/
│   ├── generate_data.py       # Creates a sample dataset (swap for a real one anytime)
│   └── Mall_Customers.csv     # Sample dataset
├── images/                    # Saved charts from segmentation_analysis.py
└── models/                    # Saved KMeans model + scaler (joblib)
```

## ▶️ Run locally
```bash
git clone https://github.com/<your-username>/customer-segmentation.git
cd customer-segmentation
pip install -r requirements.txt

# (optional) regenerate the sample dataset
python data/generate_data.py

# run the full analysis script (saves charts + model)
python segmentation_analysis.py

# launch the interactive dashboard
streamlit run app.py
```

## 📊 Sample results
With K=5, the model achieves a silhouette score of ~0.52, identifying five
clear customer segments, e.g.:
- **High income, low spenders** → target with premium/aspirational campaigns
- **High income, high spenders** → ideal for loyalty & retention programs
- **Low income, high spenders** → responsive to discounts and value bundles
- **Low income, low spenders** → price-sensitive, needs re-engagement offers
- **Average across the board** → general marketing, upsell opportunities

## 🚀 Deploy your own copy
See the deployment guide below for pushing to GitHub and hosting free on
Streamlit Community Cloud.
