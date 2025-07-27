# 🧠 AI-Powered Image Categorization for E-Commerce

This project showcases a practical application of Artificial Intelligence (AI), Computer Vision techniques, and Data Analysis in the e-commerce domain. It focuses on automatically classifying product images into categories by extracting visual features (e.g., brightness, dimensions, color distribution) and combining them with structured product data. The solution uses traditional machine learning models for fast, interpretable, and efficient performance—without relying on deep learning.

## 📊 Project Summary

- **Title**: AI-Powered Image Categorization for E-Commerce
- **Type**: Freelance-Style Portfolio Project
- **Skills**: Data Science · Image Classification · Deep Learning · Convolutional Neural Networks · Python · Pandas · Matplotlib . scikit-learn classifiers  · EDA · Visualization


 ## 🧰 Tools & Technologies  
- **Languages**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, scikit-learn (metrics, model_selection)  
- **Image Handling**: PIL, manual image loading with Matplotlib  
- **Development**: Jupyter Notebook  
- **Assets**: 200 Real Product Images + CSV Dataset


---

## 📁 Dataset Overview

The dataset simulates a real e-commerce environment and contains:

- 🖼️ **200 high-quality product images** named `product-1.jpg`, `product-2.jpg`, etc.
- 📄 **Structured CSV file** (`ecommerce_data.csv`) with the following columns:

| Column           | Description                                 |
|------------------|---------------------------------------------|
| Product_ID       | Unique product identifier (e.g., P001)      |
| Product_Name     | Name of the product                         |
| Price            | Product price in USD                        |
| Rating           | Customer rating (1 to 5)                    |
| Category         | Product category (A or B)                   |
| Image_Filename   | Corresponding image filename                |

---

## 🚀 Project Phases

### ✅ Phase 1: Exploratory Data Analysis (EDA)
- Analyzed pricing distribution, customer ratings, category splits, and basic product insights.
- Visualized category-wise patterns and missing values.
- Key findings: Balanced category distribution, slight bias in average ratings across categories.

### ✅ Phase 2: Data Preprocessing & Cleaning
- Cleaned structured data, ensured all images exist and match their filenames.
- Checked for missing/null values (None found).
- Verified image dimensions, file integrity, and consistency.

### ✅ Phase 3: Image Loading & Visualization
- Loaded and visualized product images using Matplotlib.
- Displayed random samples per category for quality assurance.
- Ensured proper mapping between image files and structured data.

### ✅ Phase 4: Feature Extraction & Machine Learning Classification  
- Extracted basic image features: size, mode, pixel-level metrics (e.g., mean brightness, color distribution).  
- Engineered a feature set combining structured and visual attributes.  
- Used scikit-learn classifiers like RandomForestClassifier and LogisticRegression to classify Category A vs B.  
- Split data using train_test_split, evaluated models with accuracy_score, classification_report, and confusion_matrix.  
- Best-performing model achieved strong predictive accuracy without the need for deep learning.  
  
📝 **Note**: Instead of deep learning (CNN), we opted for a lightweight ML approach using image-level statistics + structured features for faster prototyping and higher interpretability.


### ✅ Phase 5: Integration of Structured & Visual Data
- Combined model predictions with structured fields (e.g., price, rating).
- Explored correlation between visual category predictions and numerical data.
- Detected patterns like high-rated products clustering in Category A.

### ✅ Phase 6: Final Dashboard & Summary
- Created summary visualizations for final insights.
- Exported model performance metrics (accuracy, loss).
- Prepared the results for delivery as both a **notebook** and **PDF**.

---

## 📈 Key Results

- The model can support recommendation systems and automated product tagging.
- Image-based insights enriched structured data understanding.

---

## 👨‍💻 Developed by

**Mohamed Alshahat**  
 Data Analyst & AI Developer  
 
🔗 [LinkedIn](https://linkedin.com/in/mohamed-alshahat-8754992a4)  
🔗 [GitHub](https://github.com/Alshahatmohamed)

---
👨‍💻 About the Developer 
— Mohamed Alshahat 
– Data Analyst & AI Developer 
• Specialized in combining machine learning, computer vision, and real-world data analysis 
• Focused on building elegant, freelance-ready portfolio projects with real impact 
• Available for freelance work via platforms like Upwork & Freelancer


## 💼 Use Cases

- Automated product categorization for e-commerce platforms.
- Enhancing product listing workflows with AI.
- Visual quality control and tagging systems.

---

## 🧠 Technologies Used

- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn classifiers
- Jupyter Notebook

---

## ⭐ Highlights — 
  🔍 Combined structured and image data for real-world product classification  
  📊 Created presentation-ready dashboards for client use


