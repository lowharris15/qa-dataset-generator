---

# Q&A Dataset Generator

## 📌 Overview
The **Q&A Dataset Generator** is a **Streamlit-based application** designed to extract text from **PDF and DOCX files**, generate **question-answer (Q&A) pairs**, and save them in a structured JSON format. It supports multiple AI models from **OpenAI, Anthropic, Google (Gemini), and Ollama** for text generation.

## ✨ Features
- 📄 **Extracts text** from **PDFs** and **DOCX** files  
- 🤖 **Generates Q&A pairs** using AI models (**OpenAI, Anthropic, Google, Ollama**)  
- 📦 **Saves output as JSON** with metadata  
- 📊 **Displays sample Q&A pairs and dataset statistics**  
- 🔑 **Secure API key input** for different AI providers  
- 🏗️ **Modular and extensible** design  

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/lowharris15/qa-dataset-generator.git
cd qa-dataset-generator
```

### 2️⃣ Install Dependencies
Ensure you have **Python 3.8+** installed. Then, run:
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up API Keys
The app requires API keys for **OpenAI, Anthropic, Google Gemini, or Ollama**. These can be entered through the Streamlit UI.

### 4️⃣ Run the Application
```bash
streamlit run jsonfinetunedatasets.py
```

---

## 📂 Project Structure
```
📦 qa-dataset-generator
 ┣ 📂 output/                  # Stores generated JSON datasets
 ┣ 📜 jsonfinetunedatasets.py   # Main Streamlit application
 ┣ 📜 requirements.txt          # Required dependencies
 ┗ 📜 README.md                 # Project documentation
```

---

## 🔧 How It Works

### 1️⃣ Upload a Document  
- Upload a **PDF** or **DOCX** file via the **Streamlit UI**  

### 2️⃣ Select AI Provider & Model  
- Choose between **OpenAI, Anthropic, Google Gemini, or Ollama**  
- The app fetches available models dynamically  

### 3️⃣ Enter Context for Q&A Generation  
- The **context** guides the AI in formulating meaningful questions  

### 4️⃣ Generate & Save Q&A Pairs  
- The extracted text is processed in **chunks**  
- AI models generate **five Q&A pairs per chunk**  
- The dataset is **saved as JSON**  

---

## 🗂 Example Output (JSON)
```json
{
  "metadata": {
    "source_file": "sample.pdf",
    "context": "Machine Learning",
    "model": "gpt-4",
    "total_qa_pairs": 25,
    "generation_timestamp": "2025-02-06T14:00:00"
  },
  "qa_pairs": [
    {
      "question": "In the context of Machine Learning, what is a neural network?",
      "answer": "A neural network is a series of algorithms that mimic the human brain to recognize patterns."
    },
    {
      "question": "In the context of Machine Learning, how does gradient descent work?",
      "answer": "Gradient descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of the steepest descent."
    }
  ]
}
```

---

## ⚡ Technologies Used
- **Python**  
- **Streamlit** (Web UI)  
- **PyPDF2** (PDF text extraction)  
- **docx** (DOCX text extraction)  
- **OpenAI API** (GPT models)  
- **Anthropic API** (Claude models)  
- **Google Gemini API**  
- **Ollama** (Local AI models)  

---

## 🔥 Future Enhancements
- ✅ **Support for more document formats** (TXT, HTML, Markdown)  
- ✅ **Batch file processing**  
- ✅ **Fine-tuning AI responses**  

---

## 🛠️ Troubleshooting
### ❌ OpenAI API Key Not Found?
- Ensure your OpenAI API key is correctly entered in the Streamlit UI  
- Verify that your API key is **active and valid**  

### ❌ Text Extraction Fails?
- Try a different document format (PDF vs. DOCX)  
- Ensure the file is not **scanned or encrypted**  

---

## 👨‍💻 Contributing
We welcome contributions! Fork the repo, create a branch, and submit a **Pull Request**. 🚀  

---

## 📄 License
This project is **MIT Licensed**.  

---
