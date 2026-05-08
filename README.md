# 🎬 MovieLens Recommendation Engine: Matrix Factorization & Vector Search  
**`Machine Learning | Collaborative Filtering | Deep Learning | Vector Search`**

---

## ⚡ The Solution
This project is an end-to-end Machine Learning pipeline designed to generate personalized movie recommendations at scale. Instead of relying on traditional for-loop based approaches, this engine utilizes a **Deep Matrix Factorization neural network** to learn latent user and item representations, enabling efficient and scalable recommendation generation.

### 🧪 Key Insights
By extracting learned embeddings from the trained model and deploying them into a **FAISS vector index**, the recommendation process was optimized from linear-time complexity to **near-instant nearest neighbor search**, making the system suitable for real-time applications.

---

## 🛠️ Tech Stack
![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange?logo=tensorflow)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0099cc)

- **Core:** Python 3.13  
- **ML Frameworks:** TensorFlow / Keras, FAISS  
- **Data Handling:** Pandas, NumPy  
- **Architecture:** Functional API, Vector Database  

---

## 📊 Training Dynamics & Overfitting Prevention
The following metrics highlight the importance of validation monitoring:

| Training State | Epoch | Training Loss (MSE) | Validation Loss (MSE) |
| :--- | :--- | :--- | :--- |
| **Optimal Generalization** | 6 | 0.8645 | 0.9522 |
| Overfit (Memorization) | 15 | **0.7792** | 0.9779 |

**Engineering Decision:** We implemented **EarlyStopping with weight restoration**. Although training loss continued to decrease, validation loss degraded after a point, indicating overfitting. The model restored the best weights to ensure strong generalization instead of memorization.

---

## 🚀 Architectural Features
- **Contiguous Index Mapping:** Converted sparse real-world IDs into continuous indices using `factorize()` to eliminate embedding lookup errors.  
- **Custom Neural Architecture:** Built using Keras Functional API with **user embeddings, item embeddings, bias terms, and global mean**, forming a complete collaborative filtering system.  
- **Vector-Based Inference:** Extracted embeddings and deployed them into a FAISS `IndexFlatIP` index for **fast inner-product similarity search**.  
- **Data Leakage Prevention:** Ensured global statistics (e.g., mean rating) were computed strictly from training data only.  

---

## 📂 Project Structure & Usage
1. **Clone the repo:** `git clone https://github.com/NaramCharan/Collaborative_Filtering_Recommendation_system.git`  
2. **Install dependencies:** `pip install -r requirements.txt`  
3. **Run Analysis:** Execute `Collaborative_Filtering.ipynb` to explore the pipeline, model training, embedding extraction, and FAISS-based recommendations.  

---

## 🗺️ Roadmap
- [ ] **REST API Deployment:** Build a FastAPI backend to serve real-time recommendations.  
- [ ] **Cold Start Problem:** Implement content-based filtering using TF-IDF on movie metadata.  
- [ ] **Implicit Feedback Learning:** Upgrade from MSE to ranking-based loss functions like BPR.  

---

## 📬 Contact
- **Naram Charan** - [LinkedIn](https://www.linkedin.com/in/naramcharan/) | [Email](mailto:charannaram1710@gmail.com)  
