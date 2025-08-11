[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/vgbm4cZ0)

# 🏛️ ADGM Corporate Agent v2.0

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
USE_CHROMADB=false              # true/false
RAG_TOP_K=3                     # Documents to retrieve
RAG_SIMILARITY_THRESHOLD=0.1    # Similarity threshold

# Performance
BATCH_SIZE=5                    # Documents per batch
MAX_WORKERS=3                   # Parallel processing
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

## 🎯 **Configuration Options**

### **Environment Variables**
```bash
# RAG Backend
USE_CHROMADB=false              # true/false
RAG_TOP_K=3                     # Retrieval count
RAG_SIMILARITY_THRESHOLD=0.1    # Similarity threshold

# LLM Settings
GEMINI_TEMPERATURE=0.1          # 0.0-2.0
GEMINI_MAX_TOKENS=1024          # Max output tokens

# Performance
BATCH_SIZE=5                    # Documents per batch
MAX_WORKERS=3                   # Parallel threads

# Features
ENABLE_STRUCTURE_ANALYSIS=true  # Structure validation
ENABLE_COMPLIANCE_ANALYSIS=true # Compliance checking
ENABLE_RED_FLAG_ANALYSIS=true   # Issue detection
```

## **Testing without UI**

### **Quick Test**
```bash
python demo.py
```

### **Test Dual RAG**
```bash
# Test both backends
python -c "
import os
os.environ['USE_CHROMADB'] = 'false'
from adgm_rag import retrieve_relevant_snippets
print('In-Memory:', len(retrieve_relevant_snippets('ADGM incorporation')))
"
```

## 🏆 **Performance Metrics**

### **Current Scale (14 Documents)**
- **In-Memory**: ~1ms retrieval, ~2MB memory
- **ChromaDB**: ~5-10ms retrieval, ~50MB memory
- **Recommendation**: Stick with In-Memory for now

### **Scalability**
- **In-Memory**: Optimal up to 50 documents
- **ChromaDB**: Scales to 10,000+ documents
- **Switch Point**: When you exceed 50 documents

## 🔧 **Technical Architecture**

### **Core Components**
- **Streamlit UI**: Interactive web interface
- **Document Parser**: `python-docx` integration
- **RAG Engine**: Dual backend (In-Memory/ChromaDB)
- **LLM Integration**: Google Gemini 2.0 Flash
- **Embedding Model**: SentenceTransformers (all-MiniLM-L6-v2)

### **Data Flow**
```
Document Upload → Text Extraction → RAG Retrieval → AI Analysis → Output Generation
```

### **Performance Tips**
- Keep `USE_CHROMADB=false` for current scale
- Use `BATCH_SIZE=5` for optimal processing
- Enable `SHOW_DEBUG_INFO=false` in production

## 📈 **Future Enhancements**

### **Planned Features**
- 🔍 Advanced similarity metrics
- 📱 Mobile-responsive UI
- 🌐 Multi-language support
- 🔐 Enhanced security features
- 📊 Advanced analytics dashboard

### **Scaling Strategy**
- Start with In-Memory (current)
- Switch to ChromaDB at 50+ documents
- Consider cloud deployment at 1000+ documents


---
