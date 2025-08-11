# 🔧 Developer Guide - ADGM Corporate Agent

> **Technical documentation for developers and contributors**

## 🏗️ **Architecture Overview**

### **Core Modules**
```
📁 Project Structure
├── 🚀 app.py                    # Main Streamlit application
├── 🧠 adgm_rag.py              # Unified RAG system (dual backend)
├── 🗄️ adgm_rag_chromadb.py     # ChromaDB backend implementation
├── 📋 adgm_checklists.py       # ADGM compliance checklists
├── 📄 doc_utils.py             # Document processing utilities
├── ⚙️ config.py                # Centralized configuration
├── 📊 requirements.txt          # Dependencies
└── 🔑 .env                     # Environment variables
```

### **Data Flow**
```
Document Upload → Text Extraction → RAG Retrieval → AI Analysis → Output Generation
     ↓                ↓              ↓            ↓            ↓
  Streamlit UI → python-docx → SentenceTransformers → Gemini → DOCX+JSON
```

## 🔄 **Dual RAG Implementation**

### **Backend Selection Logic**
```python
# In adgm_rag.py
def retrieve_relevant_snippets(query, top_k=None, category_filter=None):
    if USE_CHROMADB:
        # Use ChromaDB backend
        chromadb_rag = _get_chromadb_rag()
        if chromadb_rag:
            return chromadb_rag.retrieve_relevant_snippets(query, top_k, category_filter)
    
    # Fallback to in-memory
    return _retrieve_relevant_snippets_inmemory(query, top_k, category_filter)
```

### **Configuration Switch**
```python
# In config.py
USE_CHROMADB = os.getenv("USE_CHROMADB", "false").lower() == "true"

# In .env
USE_CHROMADB=false  # or true
```

## 🧠 **RAG System Details**

### **In-Memory Backend**
- **Embeddings**: Pre-computed on module load
- **Similarity**: Dot product (normalized vectors)
- **Storage**: Python list of dictionaries
- **Performance**: ~1ms retrieval

### **ChromaDB Backend**
- **Embeddings**: Generated automatically by ChromaDB
- **Similarity**: True cosine similarity
- **Storage**: Persistent vector database
- **Performance**: ~5-10ms retrieval

### **Reference Data**
```python
# 14 official ADGM documents
REFERENCE_SNIPPETS = [
    {
        "title": "ADGM Company Incorporation Checklist",
        "url": "https://...",
        "category": "Company Formation",
        "text": "Required documents for...",
        "embedding": [0.1, 0.2, ...]  # Only in in-memory mode
    }
]
```

## 🤖 **LLM Integration**

### **Gemini Configuration**
```python
# In adgm_rag.py
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

response = model.generate_content(
    enhanced_prompt,
    generation_config={
        "temperature": temperature,
        "max_output_tokens": 1024,
        "top_p": 0.8,
        "top_k": 40,
    }
)
```

### **Prompt Engineering**
```python
enhanced_prompt = f"""
You are an expert ADGM legal compliance analyst. Use the following official ADGM sources:

OFFICIAL ADGM SOURCES (Retrieved via {'ChromaDB Semantic Search' if USE_CHROMADB else 'In-Memory Search'}):
{context}

DOCUMENT CONTENT TO ANALYZE:
{prompt}

ANALYSIS REQUIREMENTS:
1. Check jurisdiction compliance (must be ADGM, not UAE Federal/Dubai/Abu Dhabi courts)
2. Verify required clauses are present and properly worded
3. Ensure formatting meets ADGM standards
4. Confirm compliance with current ADGM regulations
5. Identify ambiguous or non-binding language
6. Check for outdated legal references

Respond in this exact JSON format:
{{
    "red_flag": "Description of compliance issue or null if compliant",
    "law_citation": "Exact ADGM regulation citation",
    "suggestion": "Specific compliant alternative wording",
    "severity": "High/Medium/Low",
    "category": "jurisdiction/missing_clauses/formatting/compliance/ambiguity/outdated",
    "confidence": "High/Medium/Low"
}}
"""
```

## 📄 **Document Processing**

### **Text Extraction**
```python
# In doc_utils.py
def extract_text_sections(docx_file):
    doc = Document(docx_file)
    sections = {
        "paragraphs": [],
        "headers": [],
        "tables": [],
        "clauses": []
    }
    
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            sections["paragraphs"].append(paragraph.text.strip())
    
    return sections
```

### **Document Type Detection**
```python
def detect_document_type(text, all_types, use_ai_fallback=True):
    # Pattern-based detection
    for doc_type, patterns in DOCUMENT_PATTERNS.items():
        score = sum(1 for pattern in patterns if pattern.lower() in text.lower())
        if score >= 2:  # Threshold for detection
            return doc_type
    
    # AI fallback if enabled
    if use_ai_fallback:
        return _ai_document_type_detection(text, all_types)
    
    return "Unknown"
```

## ⚙️ **Configuration Management**

### **Environment Variables**
```bash
# Required
GOOGLE_API_KEY=your_key_here
GOOGLE_MODEL=gemini-2.0-flash

# RAG Configuration
USE_CHROMADB=false              # Backend selection
RAG_TOP_K=3                     # Retrieval count
RAG_SIMILARITY_THRESHOLD=0.1    # Similarity threshold

# Performance
BATCH_SIZE=5                    # Documents per batch
MAX_WORKERS=3                   # Parallel processing

# Features
ENABLE_STRUCTURE_ANALYSIS=true  # Structure validation
ENABLE_COMPLIANCE_ANALYSIS=true # Compliance checking
ENABLE_RED_FLAG_ANALYSIS=true   # Issue detection
```

### **Configuration Functions**
```python
def get_config_summary() -> Dict[str, Any]:
    return {
        "rag_backend": "ChromaDB" if USE_CHROMADB else "In-Memory",
        "rag_top_k": RAG_TOP_K,
        "max_files": MAX_FILES_PER_UPLOAD,
        "gemini_temperature": GEMINI_TEMPERATURE,
        # ... more config
    }

def get_rag_config() -> Dict[str, Any]:
    return {
        "backend": "ChromaDB" if USE_CHROMADB else "In-Memory",
        "top_k": RAG_TOP_K,
        "similarity_threshold": RAG_SIMILARITY_THRESHOLD,
        "persist_directory": str(CHROMA_DB_DIR) if USE_CHROMADB else None,
    }
```

## 🧪 **Testing & Development**

### **Running Tests**
```bash
# Test the dual RAG system
python -c "
import os
os.environ['USE_CHROMADB'] = 'false'
from adgm_rag import retrieve_relevant_snippets
print('In-Memory:', len(retrieve_relevant_snippets('ADGM incorporation')))

os.environ['USE_CHROMADB'] = 'true'
print('ChromaDB:', len(retrieve_relevant_snippets('ADGM incorporation')))
"

# Run demo
python demo.py
```

### **Development Workflow**
1. **Setup**: Install dependencies with `pip install -r requirements.txt`
2. **Configure**: Set `GOOGLE_API_KEY` in `.env`
3. **Test**: Run `python demo.py` to verify functionality
4. **Develop**: Make changes and test with both backends
5. **Validate**: Ensure both RAG backends work correctly

## 🔍 **Debugging & Monitoring**

### **Logging Configuration**
```python
import logging
logger = logging.getLogger(__name__)

# In functions
logger.info(f"Using {rag_config['backend']} retrieval for query: {query[:50]}...")
logger.error(f"Error in RAG retrieval: {e}")
```

### **Performance Monitoring**
```python
import time

start_time = time.time()
results = retrieve_relevant_snippets(query, top_k=3)
retrieval_time = time.time() - start_time

logger.info(f"Retrieval completed in {retrieval_time*1000:.2f}ms")
```

### **Common Issues**

#### **ChromaDB Import Error**
```bash
# Solution
pip install chromadb>=0.4.0
```

#### **API Key Issues**
```bash
# Check .env file
cat .env | grep GOOGLE_API_KEY
```

#### **Memory Issues**
```bash
# Use in-memory backend for small datasets
USE_CHROMADB=false
```

## 🚀 **Performance Optimization**

### **Current Optimizations**
- **Lazy Loading**: ChromaDB only initialized when needed
- **Embedding Caching**: Pre-computed embeddings for in-memory mode
- **Batch Processing**: Parallel document analysis
- **Async Operations**: Non-blocking LLM calls

### **Future Optimizations**
- **Vector Indexing**: Advanced similarity search algorithms
- **Model Caching**: LLM response caching
- **Connection Pooling**: Database connection optimization
- **CDN Integration**: Static asset optimization

## 📚 **API Reference**

### **Core Functions**
```python
# RAG Retrieval
retrieve_relevant_snippets(query, top_k=3, category_filter=None)

# LLM Analysis
gemini_legal_analysis(prompt, rag_snippets=None, temperature=0.1)

# Document Processing
extract_text_sections(docx_file)
detect_document_type(text, all_types)
validate_document_structure(doc_type, metadata)
```

### **Configuration Functions**
```python
get_config_summary()      # System configuration overview
get_rag_config()          # RAG-specific configuration
validate_config()          # Configuration validation
```

## 🔄 **Contributing Guidelines**

### **Code Standards**
- **Type Hints**: Use Python type hints for all functions
- **Docstrings**: Include comprehensive docstrings
- **Error Handling**: Implement proper exception handling
- **Logging**: Use structured logging for debugging

### **Testing Requirements**
- **Unit Tests**: Test individual functions
- **Integration Tests**: Test end-to-end workflows
- **Performance Tests**: Verify both RAG backends
- **Error Tests**: Test error conditions and fallbacks

### **Pull Request Process**
1. **Fork** the repository
2. **Create** feature branch
3. **Implement** changes with tests
4. **Test** both RAG backends
5. **Submit** pull request with description

---

**For technical questions, check the code comments or create an issue on GitHub.** 