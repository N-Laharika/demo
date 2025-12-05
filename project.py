import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
# Load the food_choices.csv dataset
food_df = pd.read_csv("C:/Users/lahar/Downloads/archive (4)/food_choices.csv")

# ==============================================================================
# 1. Feature Engineering and Target Aggregation
# ==============================================================================

# Create 'primary_food_type' feature
food_df['comfort_food'] = food_df['comfort_food'].fillna('none').str.lower()
food_df['comfort_food'] = food_df['comfort_food'].str.replace(' and | or ', ',', regex=True).str.strip()
food_df['primary_food'] = food_df['comfort_food'].apply(lambda x: x.split(',')[0].strip())

sweet_keywords = ['chocolate', 'ice cream', 'candy', 'cookies', 'cake', 'sweet', 'donuts', 'pie']
savory_keywords = ['pizza', 'chips', 'fries', 'salty', 'burger', 'sandwich', 'taco', 'chicken']
starchy_keywords = ['pasta', 'mac', 'rice', 'bread', 'waffle', 'pancakes', 'noodle', 'soup']

def categorize_food(food):
    if any(keyword in food for keyword in sweet_keywords):
        return 'Sweet'
    elif any(keyword in food for keyword in savory_keywords):
        return 'Savory'
    elif any(keyword in food for keyword in starchy_keywords):
        return 'Starchy'
    else:
        return 'Other'
food_df['primary_food_type'] = food_df['primary_food'].apply(categorize_food)


# Create the binary target: 1 (Negative Mood: Stress, Boredom, Sadness, Anxiety), 0 (Other)
negative_mood_codes = [1.0, 2.0, 3.0, 4.0]
target_col = 'comfort_food_reasons_coded'
food_df['negative_mood'] = food_df[target_col].apply(
    lambda x: 1 if x in negative_mood_codes else (0 if not pd.isna(x) else np.nan)
)

# ==============================================================================
# 2. Final Data Preparation (X and Y)
# ==============================================================================
new_target_col = 'negative_mood'
numerical_cols = ['income', 'calories_day', 'healthy_feeling']
categorical_col = 'primary_food_type'
gpa_col = 'GPA' 

cols_to_use = [new_target_col, gpa_col] + numerical_cols + [categorical_col]
df_model = food_df[cols_to_use].copy()

# Clean and Impute GPA/Numerical features
df_model[gpa_col] = pd.to_numeric(df_model[gpa_col].astype(str).str.split(' ').str[0].replace('nan', np.nan), errors='coerce')
df_model.dropna(subset=[new_target_col], inplace=True)
Y = df_model[new_target_col].astype(int)

X_cat = pd.get_dummies(df_model[categorical_col], drop_first=True, prefix='food')
X_num = df_model[numerical_cols + [gpa_col]]

imputer = SimpleImputer(strategy='mean')
X_num_imputed = pd.DataFrame(imputer.fit_transform(X_num), columns=X_num.columns, index=X_num.index)

X = pd.concat([X_num_imputed, X_cat], axis=1).dropna()
Y = Y[Y.index.isin(X.index)]

# ==============================================================================
# 3. Decision Tree Classifier Training and Evaluation
# ==============================================================================

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)

# Initialize and train the Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, Y_train)

# Make predictions and evaluate
Y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(Y_test, Y_pred_dt)

print("--- DECISION TREE CLASSIFIER ACCURACY---")
print(f"Model Accuracy: {accuracy_dt:.4f}")
plt.figure(figsize=(7, 5))
target_counts = Y.value_counts().sort_index()
target_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Binary Mood Target')
plt.xticks([0, 1], ['0: Other Mood', '1: Negative Mood'], rotation=0)
plt.ylabel('Count')
plt.xlabel('Mood Category')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('binary_mood_target_distribution.png')
plt.show()
importances = dt_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='lightgreen')
plt.title('Decision Tree Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('decision_tree_feature_importance.png')
plt.show()
