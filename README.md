# 📝 Text Summarizer & FAQ Generator  
### 🚀 AI-Powered Text Understanding, Built with Streamlit  

**Text Summarizer & FAQ Generator** is an intelligent Streamlit-based web app that helps you quickly **summarize lengthy text** and **generate FAQs** using a **custom-trained neural model**.  
Designed for simplicity, speed, and privacy — it runs entirely on your **local model server**, ensuring **no data leaves your system**.

---

## ✨ Key Features  
- 🧠 **AI-Based Processing** — Uses transformer-inspired neural networks for contextual understanding.  
- 📝 **Smart Summarization** — Generates short, medium, or detailed summaries in seconds.  
- ❓ **FAQ Generation** — Creates relevant, human-like Q&A sets from any document or article.  
- 🔄 **Chunk-Based Optimization** — Automatically splits large texts for efficient processing.  
- 🎨 **Modern Streamlit UI** — Clean, responsive design with gradient styling and animations.  
- 💾 **Downloadable Results** — Export summaries and FAQs in plain text format.  
- 🔒 **Local Inference** — Runs fully offline via your custom model server (`localhost:11434`).  
- ⏱️ **Extended Timeout (1000s)** — Handles long or complex documents effortlessly.  

---

## 🧩 Tech Stack  
| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **Model Interface** | REST API (`/api/chat`, `/api/generate`) |
| **Styling** | Custom CSS with gradient animations |
| **Libraries Used** | `streamlit`, `requests`, `json`, `time` |

---

## 🛠️ How to Run  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/text-summarizer-faq-generator.git
cd text-summarizer-faq-generator
