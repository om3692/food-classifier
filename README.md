# 🍏 FreshCheck AI: Food Quality Classifier

A web application that uses Deep Learning (Vision Transformers) to assess the freshness of fruits and vegetables. Upload an image, and the AI detects the item and predicts its quality.

![Project Screenshot](static/logo.png)

## 🚀 Features
* **AI-Powered:** Uses a pre-trained Vision Transformer (ViT) from Hugging Face.
* **Real-time Analysis:** Instantly classifies food as Fresh, Okay, or Rotten.
* **Clean UI:** Simple, responsive interface built with HTML/CSS.
* **FastAPI Backend:** High-performance asynchronous API.

## 🛠️ Tech Stack
* **Frontend:** HTML5, CSS3, JavaScript
* **Backend:** Python, FastAPI
* **AI/ML:** PyTorch, Transformers (Hugging Face)

## 📦 How to Run

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/food-classifier.git](https://github.com/your-username/food-classifier.git)
    cd food-classifier
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Server**
    ```bash
    python app.py
    ```

4.  **Access the App**
    Open your browser and go to `http://127.0.0.1:8000`

## 📂 Project Structure
```text
food-classifier/
├── static/          # Images and assets
├── templates/       # HTML Frontend
├── app.py           # FastAPI Backend Server
├── model_utils.py   # AI Inference Logic
└── requirements.txt # Python Dependencies
