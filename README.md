# DEMO
![Uploading image.png…]()

# 🚀 Blog Writing Agent

An AI-powered blog generation system built using **LangChain** and structured LLM workflows. This project automates blog planning and content generation through multi-step orchestration.

---

## ✨ Features

* 🧠 Generates structured blog outlines from a given topic
* 📝 Produces high-quality blog content using LLMs
* 🔗 Multi-step workflow (Planner → Generator → Refinement)
* 🔍 Research-enhanced prompting
* 🧩 Modular backend and frontend design
* 📊 Experimentation via Jupyter Notebooks

---

## 🏗️ Project Structure

```
blog-writing-agent/
│
├── bwa_backend.py        # Core orchestration logic
├── bwa_frontend.py       # Interface / interaction layer
├── *.ipynb               # Experiment notebooks
├── outputs/              # Generated blog content (recommended)
├── .env.example          # Environment variable template
├── .gitignore            # Ignored files
└── README.md             # Project documentation
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Pushpendrajaiswa/blog-writing-agent.git
cd blog-writing-agent
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Setup

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_key
```

⚠️ Never commit `.env` to GitHub.

---

## ▶️ Usage

Run the backend:

```bash
python bwa_backend.py
```

(Optional) Run frontend:

```bash
python bwa_frontend.py
```

---

## 🧠 How It Works

1. **User Input** → Topic
2. **Planner (LLM)** → Generates structured blog outline
3. **Generator (LLM)** → Writes content section-wise
4. **Optional Enhancements** → Research / refinement

---

## 📚 Notebooks

* `1_bwa_basic.ipynb` → Basic implementation
* `2_bwa_improved_prompting.ipynb` → Prompt engineering
* `3_bwa_research.ipynb` → Research integration
* `4_bwa_research_fine_tuned.ipynb` → Advanced pipeline
* `5_bwa_image.ipynb` → Multimodal experiments

---

## 🛠️ Tech Stack

* Python
* LangChain / LangGraph
* OpenAI API
* Jupyter Notebook

---

## 🚀 Future Improvements

* 🌐 Streamlit / Web UI
* 🧠 Memory & retrieval (RAG)
* 🐳 Docker deployment
* 📈 Evaluation metrics for content quality

---

## 🤝 Contributing

Feel free to fork the repo and submit pull requests.

---

## 📌 Author

**Pushpendra Jaiswal**
🔗 GitHub: https://github.com/Pushpendrajaiswa

---

## ⭐ If you found this useful, give it a star!
