[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/vgbm4cZ0)

# 🏛️ ADGM Corporate Agent

> **Intelligent AI-powered Corporate Agent fully compliant with Abu Dhabi Global Market (ADGM) regulations**

## 🚀 **Core Features**

### **📋 Mandatory Requirements (100% Implemented)**
- ✅ **Document Upload**: Accept `.docx` files via Streamlit UI
- ✅ **Document Classification**: Auto-detect document types (AoA, MoA, UBO, etc.)
- ✅ **Process Detection**: Identify legal process (Incorporation, Licensing, etc.)
- ✅ **ADGM Compliance**: Compare against official checklists
- ✅ **Missing Document Detection**: Notify about required documents
- ✅ **Red-Flag Detection**: Find compliance issues with AI analysis
- ✅ **Inline Comments**: Insert contextual feedback in .docx files
- ✅ **Downloadable Output**: Reviewed documents + structured JSON reports

### **🎯 Enhanced Capabilities**
- 🔄 **Dual RAG Backend**: Switch between In-Memory (Dot Product Similarity, fast) and ChromaDB (Cosine similarity, scalable)
- 🧠 **AI-Powered Analysis**: Google Gemini 2.0 Flash integration
- 📊 **Comprehensive Reporting**: Detailed compliance scores and recommendations
- 🎨 **Smart UI**: Interactive Streamlit interface with real-time feedback
- 📁 **Multi-Document Support**: Process up to 10 documents simultaneously  

---

## 🖼️ **App Preview**

<div align="center">

<img src="https://github.com/user-attachments/assets/ecaccc58-e171-4b97-b0af-427dd87ba692" width="800" alt="Feature Screenshot 1" />
<br/><br/>
<img src="https://github.com/user-attachments/assets/79ba8c8e-eb73-4296-b218-66733a52f837" width="800" alt="Feature Screenshot 2" />
<br/><br/>
<img src="https://github.com/user-attachments/assets/d330d072-cc9a-472d-bc2a-aa814b43e76d" width="800" alt="Feature Screenshot 3" />
<br/><br/>
<img src="https://github.com/user-attachments/assets/4c4c7338-9622-43ad-bb44-a291c345389f" width="800" alt="Feature Screenshot 4" />

</div>

---

### 🗂️ Before Processing

<div align="center">
<img src="https://github.com/user-attachments/assets/70fa10e4-33aa-4b29-877c-72beb1a16594" width="800" alt="Before Processing" />
</div>



### ✅ After Processing

<div align="center">
<img src="https://github.com/user-attachments/assets/e852cecc-3f8a-40e3-8fdf-8b35820addc5" width="800" alt="After Processing" />
</div>

---

## 🔄 **Dual RAG Architecture**
### **📦 In-Memory Backend (Default)**
- **Speed**: ~1ms retrieval (lightning fast)
- **Memory**: ~2MB (lightweight)
- **Use Case**: Perfect for current 14-document scale
- **Setup**: Zero configuration needed

### **🗄️ ChromaDB Backend (Optional)**
- **Speed**: ~5-10ms retrieval (scalable)
- **Memory**: ~50MB+ (persistent storage)
- **Use Case**: Scale to 1000+ documents
- **Setup**: `pip install chromadb>=0.4.0`

### **⚙️ Easy Switching**
```bash
# Use In-Memory (Default - Fast)
USE_CHROMADB=false

# Use ChromaDB (Scalable)  
USE_CHROMADB=true
```

## 🛠️ **Installation & Setup**

### **1. Clone Repository**
```bash
git clone https://github.com/2CentsCapitalHR/ai-engineer-task-kushagra-a-singh.git
cd ai-engineer-task-kushagra-a-singh
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Configure Environment**
Create `.env` file:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_MODEL=gemini-2.0-flash

# RAG Configuration
USE_CHROMADB=false
RAG_TOP_K=3
RAG_SIMILARITY_THRESHOLD=0.1

# LLM Configuration  
GEMINI_TEMPERATURE=0.1
GEMINI_MAX_TOKENS=1024

# Performance Configuration
BATCH_SIZE=5
MAX_WORKERS=3

# Analysis Features
ENABLE_STRUCTURE_ANALYSIS=true
ENABLE_COMPLIANCE_ANALYSIS=true
ENABLE_RED_FLAG_ANALYSIS=true

# UI Configuration
SHOW_DEBUG_INFO=false
ENABLE_DOWNLOAD_ALL=true
```

### **4. Run Application**
```bash
streamlit run app.py
```

## 📚 **Supported Document Types**

### **🏢 Company Formation**
- Articles of Association (AoA)
- Memorandum of Association (MoA)
- Board Resolution for Incorporation
- Shareholder Resolution
- UBO Declaration Form
- Register of Members & Directors

### **🏪 Branch Company Setup**
- Parent Company Documents
- Power of Attorney
- Local Representative Documents
- Business Plan

### **👥 Employment & HR**
- Standard Employment Contracts
- HR Policies
- Compliance Frameworks

### **📋 Licensing & Compliance**
- Licensing Applications
- Compliance Policies
- Data Protection Policies
- Annual Filings

## 🔍 **How It Works**

### **1. Document Upload**
- Drag & drop `.docx` files
- Automatic file validation
- Support for multiple documents

### **2. AI Analysis Pipeline**
- **Text Extraction**: Parse document content
- **Type Detection**: Identify document category
- **RAG Retrieval**: Find relevant ADGM regulations
- **Compliance Check**: AI-powered red-flag detection
- **Structure Validation**: Verify required sections

### **3. Output Generation**
- **Reviewed Documents**: `.docx` with inline comments
- **JSON Summary**: Structured analysis results
- **Compliance Report**: Detailed findings and recommendations

## 📊 **Sample Output**

### **JSON Summary Structure**
```json
{
  "process": "Company Incorporation",
  "documents_uploaded": 4,
  "required_documents": 5,
  "missing_documents": ["Register of Members and Directors"],
  "compliance_score": 85.5,
  "issues_found": [
    {
      "document": "Articles of Association",
      "section": "Clause 3.1",
      "issue": "Jurisdiction clause does not specify ADGM",
      "severity": "High",
      "suggestion": "Update jurisdiction to ADGM Courts",
      "law_citation": "ADGM Companies Regulations 2020, Article 6"
    }
  ]
}
```

## 🧠 Smart Batching & Robust Parsing
- Combines multiple small sections into one Gemini call while preserving per‑section results
- Enforces JSON‑only responses with strict prompt instructions
- Falls back gracefully and, on rare parser failures, inserts a single consolidated technical notice (not repeated) in the reviewed doc

## 🎯 Configuration Options

### Environment Variables
```bash
# RAG Backend
USE_CHROMADB=false              # true/false
RAG_TOP_K=3                     # Retrieval count
RAG_SIMILARITY_THRESHOLD=0.1    # Similarity threshold

# LLM Settings
GEMINI_TEMPERATURE=0.1          # 0.0-2.0
GEMINI_MAX_TOKENS=1024          # Max output tokens

# Performance (UI-level)
BATCH_SIZE=5                    # Documents per batch
MAX_WORKERS=3                   # Parallel threads

# Enhanced system (batching & free-plan tuning)
GEMINI_FREE_PLAN=true           # Use conservative defaults
GEMINI_MAX_SECTIONS_PER_BATCH=5 # Lower to 4 for more reliability if needed
GEMINI_MAX_TOTAL_CHARS=4000     # Lower to 3500 for more reliability
GEMINI_MAX_REQUESTS_PER_MINUTE=15
GEMINI_RETRY_ATTEMPTS=3
```

## 🔍 How It Works

### 2. AI Analysis Pipeline
- **Text Extraction**: Parse document content
- **Type Detection**: Identify document category
- **RAG Retrieval**: Find relevant ADGM regulations
- **Smart Batching**: Merge multiple sections with explicit markers (`--- SECTION i START/END ---`) to trigger batched handling
- **Strict JSON Parsing**: Enforce JSON‑only output and parse per section
- **Structure Validation**: Verify required sections

## 📈 **Future Enhancements**

### **Planned Features**
- 🔍 Advanced similarity metrics
- 📱 Mobile-responsive UI
- 🌐 Multi-language support
- 📊 Advanced analytics dashboard

### **Scaling Strategy**
- Start with In-Memory (current)
- Switch to ChromaDB at 50+ documents
- Consider cloud deployment at 1000+ documents


---
