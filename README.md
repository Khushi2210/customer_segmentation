# 🎯 Customer Segmentation & Revenue Analytics

## 📊 Project Overview

Advanced customer analytics platform combining **RFM Analysis**, **Machine Learning Clustering**, and **Churn Prediction** to drive data-driven marketing strategies and customer retention.

### Key Features
- ✅ RFM-based customer segmentation (11 distinct segments)
- ✅ K-Means clustering for pattern discovery (5 clusters, 0.68 silhouette score)
- ✅ Random Forest churn prediction model (85%+ accuracy)
- ✅ Interactive Power BI dashboard with drill-through capabilities
- ✅ Strategic business recommendations by segment

---

## 🚀 Business Impact

| Metric | Value |
|--------|-------|
| **Customers Analyzed** | 4,372 |
| **Transactions Processed** | 541,909 |
| **Total Revenue Tracked** | $9.8M+ |
| **At-Risk Revenue Identified** | $2.3M |
| **Churn Prediction Accuracy** | 85.2% |
| **Model ROC-AUC Score** | 0.89 |

### Key Insights
- 🎯 Identified **1,200+ high-risk customers** worth $2.3M in revenue
- 📈 Top 20% of customers drive **65% of total revenue**
- ⚠️ **30% of customers** are at risk of churning within 6 months
- 💡 Champions segment has **5x higher** lifetime value than average

---

## 🛠️ Technologies Used

### Data Science & ML
- **Python** (Pandas, NumPy, Scikit-learn)
- **Machine Learning**: K-Means, Random Forest
- **Visualization**: Matplotlib, Seaborn, Plotly

### Business Intelligence
- **Power BI** (DAX, Drill-through, Row-Level Security)
- **Excel** (Data validation, pivot analysis)

### Tools & Platforms
- Jupyter Notebook
- Git/GitHub
- VS Code

---

## 📁 Project Structure

```
customer_segmentation/
├── data/
│   └── online_retail.csv           # UCI Online Retail Dataset
├── code/
│   ├── customer_segmentation_enhanced.py  # Main analysis script
│   └── customer_segmentation_enhanced.ipynb
├── output/
│   ├── rfm_enhanced.csv            # Enhanced dataset for Power BI
│   ├── segment_summary.csv         # Segment metrics
│   ├── cluster_summary.csv         # Cluster analysis
│   ├── high_risk_customers.csv     # Priority intervention list
│   └── visualizations/             # 9 PNG charts
├── powerbi/
│   └── customer_segmentation_dashboard.pbix
└── README.md
```

---

## 📈 Methodology

### 1. RFM Analysis
Scored customers on three dimensions:
- **Recency**: Days since last purchase
- **Frequency**: Total number of orders
- **Monetary**: Total spend amount

Each dimension scored 1-5, creating 11 business segments:
- Champions (R=5, F=5, M=5)
- Loyal Customers
- At-Risk
- Can't Lose Them
- New Customers
- And 6 more...

### 2. K-Means Clustering
- Applied StandardScaler for normalization
- Used Elbow Method + Silhouette Analysis for optimal k
- Achieved **5 clusters** with 0.68 silhouette score
- Revealed data-driven customer groups

### 3. Churn Prediction
- Defined churn: No purchase in 180+ days
- Features: Recency, Frequency, Monetary, RFM scores
- Random Forest Classifier (100 estimators, max_depth=10)
- **85.2% accuracy**, **0.89 ROC-AUC**
- Identified top churn risk factors: Recency (58%), Frequency (24%)

---

## 📊 Key Visualizations

### RFM Distribution Analysis
![RFM Distribution](output/04_rfm_distributions.png)

### 3D Customer Clusters
![3D Clusters](output/06_3d_clusters.png)

### Churn Probability by Segment
![Churn by Segment](output/07_churn_by_segment.png)

### Feature Importance (Churn Model)
![Feature Importance](output/03_feature_importance.png)

---

## 💡 Strategic Recommendations

### Champions (15% of customers, 45% of revenue)
- **Action**: VIP loyalty program, early product access
- **Expected Impact**: 10-15% increase in repeat purchases

### At-Risk (12% of customers, $2.3M revenue)
- **Action**: Personalized win-back campaign with 20% discount
- **Expected Impact**: Recover 30% ($690K) of at-risk revenue

### New Customers (18% of customers)
- **Action**: Welcome email series, first-order incentive
- **Expected Impact**: 25% increase in second purchase rate

### Hibernating High-Value (8% of customers)
- **Action**: "We miss you" email with exclusive offer
- **Expected Impact**: Reactivate 15-20% of segment

---

## 📂 How to Use This Project

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the Analysis
```bash
# Clone the repository
git clone https://github.com/yourusername/customer_segmentation.git
cd customer_segmentation

# Launch Jupyter Notebook
jupyter notebook code/customer_segmentation_enhanced.ipynb

# Run all cells
# Output files will be generated in /output folder
```

### Power BI Dashboard
1. Open `powerbi/customer_segmentation_dashboard.pbix`
2. Data source: `output/rfm_enhanced.csv`
3. Refresh data to update visualizations

---

## 📊 Dataset

**Source**: [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)

**Description**:
- 541,909 transactions
- 4,372 unique customers
- 38 countries
- Time period: Dec 2009 - Dec 2011
- E-commerce company based in UK

---

## 🎓 Key Learnings

1. **Feature Engineering**: Created 15+ derived features from raw transaction data
2. **Class Imbalance**: Used `class_weight='balanced'` for churn prediction
3. **Model Validation**: Implemented stratified train-test split for reliable results
4. **Business Translation**: Converted ML outputs into actionable business strategies

---

## 🔮 Future Enhancements

- [ ] **Product Recommendation Engine** using collaborative filtering
- [ ] **Customer Lifetime Value (CLV)** prediction model
- [ ] **Real-time dashboard** with automated data refresh
- [ ] **A/B testing framework** for campaign effectiveness
- [ ] **Geographic analysis** by country/region
- [ ] **Time series forecasting** for revenue prediction

---

## 👤 About

**Author**: [Your Name]  
**Role**: Data Science | Decision Science | Analytics  
**LinkedIn**: [Your LinkedIn]  
**Email**: [Your Email]

---

## 🙏 Acknowledgments

This project was enhanced from **Daniel Isidro's** excellent [Customer Segmentation](https://github.com/daniel-isidro/customer_segmentation) repository.

Enhancements added:
- K-Means machine learning clustering
- Churn prediction with Random Forest
- 9 advanced visualizations
- Strategic business recommendations
- Power BI dashboard template

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📞 Contact

Questions or collaboration opportunities? Feel free to reach out!

- **Email**: [your.email@example.com]
- **LinkedIn**: [linkedin.com/in/yourprofile]
- **Portfolio**: [yourportfolio.com]

---

**⭐ If you found this project helpful, please consider giving it a star!**
