import numpy as np
import pandas as pd
import math
import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os # Import os module to create directories

# Machine Learning Libraries
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    silhouette_score,
    accuracy_score,
    roc_auc_score,
    roc_curve
)

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# ============================================================================
# PART 2: LOAD AND PREPARE DATA (From Daniel's Original Code)
# ============================================================================

# Load the dataset
# Option 1: Use UCI Online Retail dataset
# df = pd.read_csv('data/online_retail.csv', encoding='unicode_escape')

# Option 2: Use Daniel's sample data
df1 = pd.read_csv('sales_asia.csv', sep=';')

print(f"Dataset loaded: {df1.shape[0]} rows, {df1.shape[1]} columns")
print(df1.head())

# Convert week.year to proper date format
# The week.year column is a float where the integer part is the week and the decimal part is the two-digit year (e.g., 3.20 means Week 3, Year 20)
def parse_week_year_to_string(week_year_float):
    week_int = int(week_year_float)
    # Extract the fractional part and convert to a two-digit year
    year_float_part = week_year_float - week_int
    year_two_digits = int(round(year_float_part * 100)) # Round to handle floating point inaccuracies
    # Construct the string as 'WEEK-YY-DAYOFWEEK' (using Monday as day 1)
    return f"{week_int}-{year_two_digits:02d}-1"

df1['date_string'] = df1['week.year'].apply(parse_week_year_to_string)
df1['date'] = pd.to_datetime(df1['date_string'], format='%W-%y-%w')

# Rename columns
df2 = df1.copy()
df2.rename(columns={'revenue': 'monetary'}, inplace=True)
# Convert monetary to numeric, handling comma as decimal separator
df2['monetary'] = df2['monetary'].str.replace(',', '.', regex=False).astype(float)
df2 = df2[['country', 'id', 'monetary', 'units', 'date']]

print("\n✅ Data preparation complete!")
print(df2.info())

# ============================================================================
# PART 3: RFM ANALYSIS (Original + Enhanced)
# ============================================================================

# Set analysis date (most recent date + 1 day)
NOW = df2['date'].max() + timedelta(days=1)
print(f"\nAnalysis Date: {NOW}")

# Calculate days since purchase
df2['days_since_purchase'] = (NOW - df2['date']).dt.days

# Filter last 365 days
df3 = df2[df2['days_since_purchase'] <= 365].copy()

# Create unique ID combining country and customer ID
df3['id+'] = df3['country'].astype(str) + df3['id'].astype(str)

print(f"\nFiltered to last 365 days: {df3.shape[0]} transactions")

# Aggregate RFM metrics
rfm = df3.groupby(['id+', 'country', 'id']).agg(
    recency=('days_since_purchase', 'min'),  # Recency
    frequency=('id', 'count'),                 # Frequency (renamed from 'id' count)
    monetary=('monetary', 'sum')              # Monetary
).reset_index()

# rfm.columns = ['id+', 'country', 'id', 'recency', 'frequency', 'monetary'] # This line is no longer needed with named aggregations

print(f"\n✅ RFM calculated for {len(rfm)} customers")
print(rfm.head())

# ============================================================================
# PART 4: RFM SCORING (Original)
# ============================================================================

# Calculate quintiles for R, F, M scores (1-5)
rfm['r'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1])
rfm['f'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
rfm['m'] = pd.qcut(rfm['monetary'], q=5, labels=[1, 2, 3, 4, 5])

# Convert to integers
rfm['r'] = rfm['r'].astype(int)
rfm['f'] = rfm['f'].astype(int)
rfm['m'] = rfm['m'].astype(int)

# Create RFM score
rfm['rfm_score'] = rfm['r'].astype(str) + rfm['f'].astype(str) + rfm['m'].astype(str)

# Create FM score (simplified)
rfm['fm'] = ((rfm['f'] + rfm['m']) / 2).apply(lambda x: math.trunc(x))

print("\n✅ RFM Scores calculated!")

# ============================================================================
# PART 5: SEGMENT MAPPING (Original)
# ============================================================================

# Create segment map
segment_map = {
    (5, 5): 'champions',
    (5, 4): 'champions',
    (4, 5): 'loyal customers',
    (4, 4): 'loyal customers',
    (5, 3): 'potential loyalists',
    (4, 3): 'potential loyalists',
    (5, 2): 'promising',
    (4, 2): 'promising',
    (3, 5): "can't lose",
    (3, 4): "can't lose",
    (5, 1): 'recent customers',
    (4, 1): 'recent customers',
    (3, 3): 'need attention',
    (3, 2): 'about to sleep',
    (2, 5): 'at risk',
    (2, 4): 'at risk',
    (2, 3): 'at risk',
    (2, 2): 'hibernating',
    (2, 1): 'hibernating',
    (1, 5): 'lost',
    (1, 4): 'lost',
    (1, 3): 'lost',
    (1, 2): 'lost',
    (1, 1): 'lost'
}

rfm['segment'] = rfm[['r', 'fm']].apply(lambda x: segment_map.get((x['r'], x['fm']), 'other'), axis=1)

print("\n✅ Customer segments assigned!")
print(rfm['segment'].value_counts())

# ============================================================================
# PART 6: 🆕 ENHANCEMENT 1 - K-MEANS CLUSTERING
# ============================================================================

print("\n" + "="*70)
print("🚀 ENHANCEMENT 1: K-MEANS CLUSTERING")
print("="*70)

# Prepare data for clustering
rfm_clustering = rfm[['recency', 'frequency', 'monetary']].copy()

# Standardize features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_clustering)

# Elbow method to find optimal k
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))

# Plot elbow curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0].set_ylabel('Inertia', fontsize=12)
axes[0].set_title('Elbow Method For Optimal k', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/kmeans_optimization.png', dpi=300, bbox_inches='tight')
plt.show()

# Apply K-Means with optimal k (typically 5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm['ml_cluster'] = kmeans.fit_predict(rfm_scaled)

# Analyze clusters
print(f"\n✅ K-Means clustering complete with k={optimal_k}")
print(f"Silhouette Score: {silhouette_score(rfm_scaled, rfm['ml_cluster']):.3f}\n")

cluster_analysis = rfm.groupby('ml_cluster').agg({
    'recency': ['mean', 'min', 'max'],
    'frequency': ['mean', 'min', 'max'],
    'monetary': ['mean', 'sum', 'count']
}).round(2)

print("Cluster Analysis:")
print(cluster_analysis)

# Visualize clusters
fig = plt.figure(figsize=(15, 5))

# 2D scatter: Recency vs Frequency
ax1 = fig.add_subplot(131)
scatter1 = ax1.scatter(rfm['recency'], rfm['frequency'],
                       c=rfm['ml_cluster'], cmap='viridis',
                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax1.set_xlabel('Recency (days)', fontsize=11)
ax1.set_ylabel('Frequency (orders)', fontsize=11)
ax1.set_title('Clusters: Recency vs Frequency', fontsize=12, fontweight='bold')
plt.colorbar(scatter1, ax=ax1, label='Cluster')

# 2D scatter: Frequency vs Monetary
ax2 = fig.add_subplot(132)
scatter2 = ax2.scatter(rfm['frequency'], rfm['monetary'],
                       c=rfm['ml_cluster'], cmap='viridis',
                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax2.set_xlabel('Frequency (orders)', fontsize=11)
ax2.set_ylabel('Monetary (value)', fontsize=11)
ax2.set_title('Clusters: Frequency vs Monetary', fontsize=12, fontweight='bold')
plt.colorbar(scatter2, ax=ax2, label='Cluster')

# 3D scatter
ax3 = fig.add_subplot(133, projection='3d')
scatter3 = ax3.scatter(rfm['recency'], rfm['frequency'], rfm['monetary'],
                       c=rfm['ml_cluster'], cmap='viridis',
                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax3.set_xlabel('Recency', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_zlabel('Monetary', fontsize=10)
ax3.set_title('3D Cluster Visualization', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('output/cluster_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# PART 7: 🆕 ENHANCEMENT 2 - CHURN PREDICTION WITH RANDOM FOREST
# ============================================================================

print("\n" + "="*70)
print("🚀 ENHANCEMENT 2: CHURN PREDICTION MODEL")
print("="*70)

# Define churn (customers who haven't purchased in 180+ days)
churn_threshold = 180
rfm['is_churned'] = (rfm['recency'] > churn_threshold).astype(int)

print(f"\nChurn Definition: Recency > {churn_threshold} days")
print(f"Churned Customers: {rfm['is_churned'].sum()} ({rfm['is_churned'].mean()*100:.1f}%)")
print(f"Active Customers: {(1-rfm['is_churned']).sum()} ({(1-rfm['is_churned']).mean()*100:.1f}%)")

# Prepare features
feature_cols = ['recency', 'frequency', 'monetary', 'r', 'f', 'm', 'fm']
X = rfm[feature_cols]
y = rfm['is_churned']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluate model
print("\n" + "-"*70)
print("MODEL PERFORMANCE")
print("-"*70)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Active', 'Churned']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Active', 'Churned'],
            yticklabels=['Active', 'Churned'],
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix - Churn Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance for Churn Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Add churn probability to all customers
rfm['churn_probability'] = rf_model.predict_proba(X)[:, 1]

# Create churn risk categories
rfm['churn_risk'] = pd.cut(
    rfm['churn_probability'],
    bins=[0, 0.3, 0.7, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

print("\n✅ Churn prediction complete!")
print("\nChurn Risk Distribution:")
print(rfm['churn_risk'].value_counts())

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Churn Prediction Model', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# PART 8: 🆕 ENHANCEMENT 3 - ADVANCED VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print("🚀 ENHANCEMENT 3: ADVANCED VISUALIZATIONS")
print("="*70)

# Churn probability by segment
plt.figure(figsize=(14, 6))

ax1 = plt.subplot(121)
segment_churn = rfm.groupby('segment')['churn_probability'].mean().sort_values(ascending=False)
segment_churn.plot(kind='barh', color='coral', edgecolor='black', ax=ax1)
ax1.set_xlabel('Average Churn Probability', fontsize=11)
ax1.set_ylabel('Customer Segment', fontsize=11)
ax1.set_title('Churn Risk by Segment', fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

ax2 = plt.subplot(122)
sns.boxplot(data=rfm, y='segment', x='churn_probability', palette='Set2', ax=ax2)
ax2.set_xlabel('Churn Probability', fontsize=11)
ax2.set_ylabel('Customer Segment', fontsize=11)
ax2.set_title('Churn Probability Distribution by Segment', fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('output/churn_by_segment.png', dpi=300, bbox_inches='tight')
plt.show()

# Revenue analysis
plt.figure(figsize=(14, 6))

ax1 = plt.subplot(121)
segment_revenue = rfm.groupby('segment')['monetary'].sum().sort_values(ascending=False)
segment_revenue.plot(kind='bar', color='steelblue', edgecolor='black', ax=ax1)
ax1.set_xlabel('Customer Segment', fontsize=11)
ax1.set_ylabel('Total Revenue', fontsize=11)
ax1.set_title('Revenue Contribution by Segment', fontsize=13, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

ax2 = plt.subplot(122)
at_risk_revenue = rfm.groupby('churn_risk')['monetary'].sum()
colors = ['green', 'orange', 'red']
at_risk_revenue.plot(kind='pie', autopct='%1.1f%%', colors=colors,
                     startangle=90, ax=ax2,
                     labels=['Low Risk', 'Medium Risk', 'High Risk'])
ax2.set_ylabel('')
ax2.set_title('Revenue Distribution by Churn Risk', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('output/revenue_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Customer distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Segment distribution
segment_counts = rfm['segment'].value_counts()
axes[0,0].pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
axes[0,0].set_title('Customer Distribution by Segment', fontsize=12, fontweight='bold')

# Cluster distribution
cluster_counts = rfm['ml_cluster'].value_counts().sort_index()
axes[0,1].bar(cluster_counts.index, cluster_counts.values, color='skyblue', edgecolor='black')
axes[0,1].set_xlabel('ML Cluster', fontsize=11)
axes[0,1].set_ylabel('Number of Customers', fontsize=11)
axes[0,1].set_title('Customer Distribution by ML Cluster', fontsize=12, fontweight='bold')
axes[0,1].grid(axis='y', alpha=0.3)

# RFM heatmap
rfm_pivot = rfm.groupby(['r', 'fm']).size().unstack(fill_value=0)
sns.heatmap(rfm_pivot, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1,0], cbar_kws={'label': 'Count'})
axes[1,0].set_xlabel('FM Score', fontsize=11)
axes[1,0].set_ylabel('R Score', fontsize=11)
axes[1,0].set_title('Customer Heatmap: R vs FM Scores', fontsize=12, fontweight='bold')

# Churn risk distribution
churn_risk_counts = rfm['churn_risk'].value_counts()
axes[1,1].barh(churn_risk_counts.index, churn_risk_counts.values,
               color=['green', 'orange', 'red'], edgecolor='black')
axes[1,1].set_xlabel('Number of Customers', fontsize=11)
axes[1,1].set_ylabel('Churn Risk Category', fontsize=11)
axes[1,1].set_title('Customers by Churn Risk', fontsize=12, fontweight='bold')
axes[1,1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('output/customer_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Advanced visualizations created and saved!")

# ============================================================================
# PART 9: 🆕 ENHANCEMENT 4 - BUSINESS RECOMMENDATIONS
# ============================================================================

print("\n" + "="*70)
print("🚀 ENHANCEMENT 4: STRATEGIC BUSINESS RECOMMENDATIONS")
print("="*70)

# Calculate business metrics
total_customers = len(rfm)
total_revenue = rfm['monetary'].sum()
avg_order_value = rfm['monetary'].mean()
high_risk_customers = len(rfm[rfm['churn_risk'] == 'High Risk'])
high_risk_revenue = rfm[rfm['churn_risk'] == 'High Risk']['monetary'].sum()

print(f"\n📊 KEY BUSINESS METRICS")
print("-" * 50)
print(f"Total Customers: {total_customers:,}")
print(f"Total Revenue: ${total_revenue:,.2f}")
print(f"Average Customer Value: ${avg_order_value:,.2f}")
print(f"High-Risk Customers: {high_risk_customers:,} ({high_risk_customers/total_customers*100:.1f}%)")
print(f"At-Risk Revenue: ${high_risk_revenue:,.2f} ({high_risk_revenue/total_revenue*100:.1f}%)")

# Segment-specific recommendations
recommendations = {
    'champions': {
        'action': 'VIP Treatment',
        'strategy': 'Early access to new products, exclusive rewards, personal account manager',
        'priority': 'HIGH',
        'expected_impact': 'Increase LTV by 25%'
    },
    'loyal customers': {
        'action': 'Loyalty Program',
        'strategy': 'Referral bonuses, birthday discounts, tier-based rewards',
        'priority': 'HIGH',
        'expected_impact': 'Boost repeat purchases by 20%'
    },
    'potential loyalists': {
        'action': 'Nurture Campaign',
        'strategy': 'Product recommendations, email engagement, limited-time offers',
        'priority': 'MEDIUM',
        'expected_impact': 'Convert 30% to loyal customers'
    },
    'at risk': {
        'action': 'Win-Back Campaign',
        'strategy': 'Personalized emails, exclusive discount (15-20%), survey feedback',
        'priority': 'HIGH',
        'expected_impact': 'Recover 25% of at-risk customers'
    },
    "can't lose": {
        'action': 'Urgent Reactivation',
        'strategy': 'Phone outreach, significant discount, account review meeting',
        'priority': 'URGENT',
        'expected_impact': 'Prevent $500K+ revenue loss'
    },
    'hibernating': {
        'action': 'Reengagement',
        'strategy': 'Multi-channel campaign, new product showcase, limited-time offers',
        'priority': 'MEDIUM',
        'expected_impact': 'Reactivate 15% of hibernating customers'
    },
    'lost': {
        'action': 'Last Attempt',
        'strategy': 'Survey for feedback, major discount code, re-onboarding offer',
        'priority': 'LOW',
        'expected_impact': 'Recover 5-10% of lost customers'
    }
}

print(f"\n🎯 SEGMENT-SPECIFIC RECOMMENDATIONS")
print("="*70)

for segment, details in recommendations.items():
    segment_data = rfm[rfm['segment'] == segment]
    if len(segment_data) > 0:
        print(f"\n{segment.upper()}")
        print(f"  Customers: {len(segment_data):,}")
        print(f"  Revenue: ${segment_data['monetary'].sum():,.2f}")
        print(f"  Priority: {details['priority']}")
        print(f"  Action: {details['action']}")
        print(f"  Strategy: {details['strategy']}")
        print(f"  Expected Impact: {details['expected_impact']}")

# Top priority customers
print(f"\n🔥 TOP PRIORITY ACTIONS")
print("="*70)

# High-value at-risk customers
high_value_at_risk = rfm[
    (rfm['churn_risk'] == 'High Risk') &
    (rfm['monetary'] > rfm['monetary'].quantile(0.75))
].sort_values('monetary', ascending=False).head(10)

print(f"\n1. HIGH-VALUE AT-RISK CUSTOMERS (Top 10)")
print(f"   Total at Risk: ${high_value_at_risk['monetary'].sum():,.2f}")
print("   Immediate Action Required!")
print(high_value_at_risk[['id', 'segment', 'recency', 'frequency', 'monetary', 'churn_probability']])

# Champions to retain
champions = rfm[rfm['segment'] == 'champions'].sort_values('monetary', ascending=False).head(10)
if len(champions) > 0:
    print(f"\n2. TOP CHAMPIONS (Protect at All Costs)")
    print(f"   Total Value: ${champions['monetary'].sum():,.2f}")
    print(champions[['id', 'recency', 'frequency', 'monetary', 'churn_probability']])

# Potential to loyal conversion
potential_loyalists = rfm[
    (rfm['segment'] == 'potential loyalists') &
    (rfm['churn_probability'] < 0.3)
].sort_values('monetary', ascending=False).head(10)

if len(potential_loyalists) > 0:
    print(f"\n3. POTENTIAL LOYALISTS TO CONVERT")
    print(f"   Conversion Opportunity: ${potential_loyalists['monetary'].sum():,.2f}")
    print(potential_loyalists[['id', 'recency', 'frequency', 'monetary', 'churn_probability']])

# ============================================================================
# PART 10: EXPORT ENHANCED DATASET
# ============================================================================

print("\n" + "="*70)
print("💾 EXPORTING ENHANCED DATASET")
print("="*70)

# Add calculated fields for Power BI
rfm['customer_lifetime_value'] = rfm['monetary'] * rfm['frequency']
rfm['average_order_value'] = rfm['monetary'] / rfm['frequency']
rfm['days_since_first_purchase'] = 365 - rfm['recency']

# Create final output
output_columns = [
    'id', 'id+', 'country',
    'recency', 'frequency', 'monetary',
    'r', 'f', 'm', 'fm', 'rfm_score',
    'segment', 'ml_cluster',
    'is_churned', 'churn_probability', 'churn_risk',
    'customer_lifetime_value', 'average_order_value', 'days_since_first_purchase'
]

rfm_output = rfm[output_columns].copy()

# Export to CSV
rfm_output.to_csv('output/rfm_enhanced.csv', index=False, float_format='%.2f')

print(f"\n✅ Enhanced dataset exported: output/rfm_enhanced.csv")
print(f"   Total customers: {len(rfm_output):,}")
print(f"   Total features: {len(output_columns)}")

# Summary statistics
print(f"\n📈 FINAL SUMMARY STATISTICS")
print("="*70)
print(rfm_output.describe())

print("\n" + "="*70)
print("✅ PROJECT COMPLETE!")
print("="*70)
print("\nNext Steps:")
print("1. Open Power BI Desktop")
print("2. Import: output/rfm_enhanced.csv")
print("3. Create dashboards using the guide in PROJECT_GUIDE.md")
print("4. Update your GitHub repository")
print("5. Add to resume with metrics!")

# Save summary report
summary = f"""
CUSTOMER SEGMENTATION ENHANCED - PROJECT SUMMARY
Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW
- Total Customers: {total_customers:,}
- Total Revenue: ${total_revenue:,.2f}
- Analysis Period: Last 365 days
- Countries: {rfm['country'].nunique()}

MACHINE LEARNING RESULTS
- K-Means Clusters: {optimal_k}
- Silhouette Score: {silhouette_score(rfm_scaled, rfm['ml_cluster']):.3f}
- Churn Model Accuracy: {accuracy_score(y_test, y_pred):.3f}
- ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}

BUSINESS METRICS
- High-Risk Customers: {high_risk_customers:,} ({high_risk_customers/total_customers*100:.1f}%)
- At-Risk Revenue: ${high_risk_revenue:,.2f}
- Average Customer Value: ${avg_order_value:,.2f}

TOP SEGMENTS
{rfm['segment'].value_counts().head()}

FILES GENERATED
- rfm_enhanced.csv (main dataset)
- kmeans_optimization.png
- cluster_visualization.png
- confusion_matrix.png
- feature_importance.png
- roc_curve.png
- churn_by_segment.png
- revenue_analysis.png
- customer_distribution.png
"""

with open('output/project_summary.txt', 'w') as f:
    f.write(summary)

print("\n📄 Project summary saved: output/project_summary.txt")
print("\n🎉 All Done! Happy Analyzing!")
