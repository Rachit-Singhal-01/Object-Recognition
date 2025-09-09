# 🖼️ Object Recognition on CIFAR-10

This project implements and compares **traditional computer vision pipelines** (HOG/SIFT + SVM) with a **Convolutional Neural Network (CNN)** baseline for image classification on the **CIFAR-10** dataset.  
The goal is to evaluate how handcrafted features and classical machine learning compare against modern deep learning approaches.

---

## 📂 Repository Structure

- `Computer_Vision_Course_Work.ipynb` → End-to-end notebook (data loading, feature extraction, models, evaluation)  
- `Comp_Vision.docx` → Project documentation/notes  
- `.gitattributes` → Git attributes for text/binary normalization  

---

## 🎯 Objectives

- Implement **HOG** and **SIFT** feature extraction pipelines  
- Train and evaluate **Support Vector Machines (SVM)** on classical features  
- Build and train a **baseline CNN** for end-to-end recognition  
- Compare performance, training cost, and limitations of both approaches  

---

## 📊 Dataset

- **CIFAR-10** → 60,000 images (32×32 pixels, RGB) across **10 classes** (50,000 train / 10,000 test).  
  Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.  

🔗 [Official CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 🏗️ Workflow

1. **Data Loading** → Import CIFAR-10 train/test sets  
2. **Preprocessing** → Normalization, optional grayscale conversion for HOG/SIFT  
3. **Feature Extraction (Classical)** →  
   - HOG (Histogram of Oriented Gradients)  
   - SIFT (Scale-Invariant Feature Transform) + Bag-of-Visual-Words (BoVW)  
4. **Model Training** →  
   - Classical: SVM (linear / RBF kernel)  
   - Deep Learning: CNN trained on raw pixel data  
5. **Evaluation** → Accuracy, per-class metrics, confusion matrices  
6. **Visualization** → Feature maps, embeddings, sample predictions  

---

## 🔬 Model Architectures

### 1) Classical (HOG/SIFT + SVM)
- **Feature Extraction**: HOG histograms or SIFT descriptors  
- **Classifier**: SVM (linear or RBF kernel)  
- **Tuning**: Grid-search hyperparameters (C, γ, kernel)  

---

### 2) CNN Baseline

The CNN baseline provides an **end-to-end deep learning approach** without handcrafted features. It learns hierarchical representations directly from raw images.  

**Architecture:**
Input (32×32×3)
→ Conv(32, 3×3) + ReLU
→ Conv(32, 3×3) + ReLU
→ MaxPool(2×2)
→ Dropout(0.25)
→ Conv(64, 3×3) + ReLU
→ Conv(64, 3×3) + ReLU
→ MaxPool(2×2)
→ Dropout(0.25)
→ Flatten
→ Dense(256) + ReLU
→ Dropout(0.5)
→ Dense(10) + Softmax


**Training Configuration:**
- Optimizer → Adam  
- Loss → Categorical Cross-Entropy  
- Batch Size → 64–128  
- Epochs → 20–50 (with early stopping)  
- Regularization → Dropout layers (0.25–0.5)  
- Augmentation → Random horizontal flips, random crops  

**Expected Outcome:**
- CNN generally **outperforms classical methods** on CIFAR-10 with higher accuracy, at the cost of longer training time.  
- Serves as a baseline to show the benefit of feature learning vs. feature engineering.  

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Rachit-Singhal-01/Object-Recognition.git
cd Object-Recognition

### Create a Virtual Environment
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
.\venv\Scripts\activate         # Windows

pip install -r requirements.txt

🛠️ Dependencies

numpy, pandas

scikit-learn

matplotlib, seaborn

opencv-python, scikit-image

torch, torchvision, torchaudio (or tensorflow/keras if TF used)

notebook / jupyterlab

📚 References

Krizhevsky, A. Learning Multiple Layers of Features from Tiny Images. CIFAR-10 dataset, 2009.

Dalal & Triggs. Histograms of Oriented Gradients for Human Detection. CVPR 2005.

Lowe, D. Distinctive Image Features from Scale-Invariant Keypoints. IJCV 2004.
