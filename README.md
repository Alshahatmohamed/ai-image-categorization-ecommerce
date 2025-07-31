# 🧠🛒 AI-Powered E-Commerce Product Classification using Machine Learning & Deep Learning (PyTorch)

## 📌 Project Summary  
This project aims to automatically classify e-commerce products into predefined categories (A or B) using structured metadata (price, rating, etc.) and image data. It leverages a full machine learning pipeline with exploratory data analysis (EDA), feature engineering, classical ML models, and deep learning via PyTorch (CNN + Transfer Learning with ResNet18). This end-to-end solution simulates a real e-commerce scenario where both text and image inputs contribute to smart automation of product classification.
## 📁 Project Directory Structure
To ensure everything runs smoothly, please organize your project files as follows:
AI-Powered Image Categorization for E-Commerce/
│
├── your_data.csv           ← CSV file containing product data and image filenames
├── code.py                 ← Main Python script for image categorization
├── project-poster.jpg      ← Project poster or thumbnail image
├── images/                 ← 📁 Folder containing product images
│   ├── product-1.jpg
│   ├── product-2.jpg
│   ├── ...
🔹 Important Notes:
	•	The images/ folder must be placed in the same directory as the Python script and the CSV file.
	•	Inside the CSV file, the image filenames (in the Image_URL column) should be just the names, such as:
product-1.jpg
product-2.jpg
⚠️ **Note**: Images are not included in the current version of this repository to reduce file size. However, the code is fully functional and ready to work with images — just place your product images inside the /images folder, as specified in the README
## 📦 Dataset Overview  
The dataset mimics a realistic online store with both tabular and image data. It contains:  
• 🖼️ 200 high-quality product images (`product-1.jpg`, `product-2.jpg`, ..., `product-200.jpg`)  
• 📄 CSV file (`product_data.csv`) with the following columns:  
| Column | Description |  
|--------|-------------|  
| Product_ID | Unique product identifier (e.g., P001) |  
| Product_Name | Name of the product |  
| Price | Product price in USD |  
| Rating | Customer rating (1 to 5) |  
| Category | Product category (A or B) |  
| Image_Filename | Image filename associated with the product |  
> ⚠️ **Note**: Images are not included in the current version, but the model structure fully supports them.

## 🔧 Tools & Technologies Used  
- Python, Pandas, NumPy, Seaborn, Matplotlib  
- Scikit-learn (Random Forest, Label Encoding, Evaluation)  
- PyTorch (Deep Learning CNN + Transfer Learning using ResNet18)  
- PIL, TorchVision (Image preprocessing and augmentation)  
- Google Colab / Jupyter Notebook

---

## 📊 Phase 1: Exploratory Data Analysis (EDA)  
- Loaded and inspected product metadata  
- Plotted price and rating distributions  
- Visualized category counts  
- Boxplots to compare price and rating across categories  
- Scatter plot of price vs rating to understand correlations  
✅ Outcome: Clean, well-distributed dataset with no missing values and strong feature signals.

---

## 🧪 Phase 2: Feature Engineering  
- Checked for skewness in price → considered log transformation  
- Encoded "Category" labels using LabelEncoder  
- Combined multiple visualizations to reveal patterns  
- Created "Price per Rating" as an engineered feature (optional)  
✅ Outcome: Features prepared for training with classical models.

---

## 🤖 Phase 3: Machine Learning Model (Structured Data)  
- Used `RandomForestClassifier` with basic hyperparameters  
- Trained using Price and Rating to predict product Category  
- Achieved strong baseline accuracy on validation set  
✅ Output: Classical model serves as benchmark for comparison with CNN.

---

## 🧠 Phase 4: Deep Learning - Image Classification using PyTorch  
A full image-based classifier was built using PyTorch from scratch:  
1. **Data Preparation**  
   - Custom PyTorch `Dataset` created to load images and labels from CSV  
   - Applied image transforms: resize, normalization, horizontal flip  
   - DataLoaders for training and validation  

2. **Model Definition**  
   - Used pre-trained `ResNet18` as base CNN  
   - Modified final fully-connected layer to output 2 classes (A/B)  
   - Moved model to GPU if available  

3. **Training Loop**  
   - CrossEntropyLoss + Adam optimizer  
   - 5 epochs training with accuracy printed per epoch  

4. **Model Saving**  
   - Trained model saved as `product_classifier_pytorch.pth`  

✅ Output: A deep learning model ready to classify products using images alone. Can be combined with metadata later for multi-modal classification.

---

## 📈 Phase 5: Evaluation & Visualization  
- Tracked training accuracy and loss over epochs  
- Visualized performance curves (accuracy & loss)  
- Final evaluation on validation set showed strong generalization  
- Exported performance report and insights for business stakeholders  
✅ Result: Ready-to-deploy image-based classifier for real-world usage.

---

## 💡 Business Applications  
- Automatic tagging and classification of newly uploaded products  
- Fraud detection (image-label mismatch)  
- Hybrid recommender systems using structured + visual data  
- Enhanced user experience with consistent product grouping  

---

## 📁 Project Deliverables  
- Cleaned structured dataset (`ecommerce_data.csv`)  
- Complete Python codebase (ML + Deep Learning)  
- Trained PyTorch model (`product_classifier_pytorch.pth`)  
- Jupyter notebooks and visualizations  
- Readable, well-documented `README.md`  

---

## 👨‍💻 Developed by  
**Mohamed Alshahat**  
🔗 GitHub: [github.com/Alshahatmohamed](https://github.com/Alshahatmohamed)  
🔗 LinkedIn: [linkedin.com/in/mohamed-alshahat-8754992a4](https://linkedin.com/in/mohamed-alshahat-8754992a4)  
🧠 AI & Data Science Enthusiast | E-Commerce Automation | PyTorch & ML

👨‍💻 About the Developer — Mohamed Alshahat · Data Analyst & AI Developer 
• Specialized in combining machine learning, computer vision, and real-world data analysis 
• Focused on building elegant, freelance-ready portfolio projects with real impact 
• Available for freelance work via platforms 


---

💼 Use Cases  
- Product categorization for online marketplaces  
- AI-assisted product tagging pipelines  
- Lightweight visual quality control systems  

🧠 Technologies Used  
Python 3.x · Pandas · NumPy · Matplotlib · Seaborn  
scikit-learn · PyTorch · Jupyter Notebook  

⭐ Highlights  
🔍 Combined structured and visual data for intelligent classification  
📊 Built real-time dashboards and machine learning pipelines  
⚙️ Switched from classic ML to PyTorch-based deep learning for scalability
