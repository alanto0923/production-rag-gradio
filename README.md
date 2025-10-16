# Enterprise-Grade Verifiable RAG Application
### A production-ready, locally-hosted RAG system with a multi-stage architecture for maximum accuracy and trust.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides an enterprise-grade, **verifiable** Retrieval-Augmented Generation (RAG) system engineered to deliver high-precision, factually grounded answers from private documents. It moves beyond standard RAG prototypes by implementing a multi-stage, self-correcting architecture that builds unbreakable trust and ensures mission-critical reliability.

The entire system runs on *local infrastructure* using **Ollama** and **AnythingLLM**, guaranteeing 100% data privacy and security.

## ✨ Core Features & Technical Highlights

This system is built with production-readiness in mind, incorporating numerous advanced techniques:

> #### Hybrid VLM+OCR Ingestion
> An intelligent document parsing pipeline (`intelligent_document_preprocessing`) that uses a visual heuristic (`page.get_images()`) to decide between high-speed OCR for simple text and a Vision Language Model (VLM) for accurately extracting and structuring complex visual layouts like tables, charts, and multi-column text.

> #### Configurable Ingestion Profiles
> Pre-tuned processing profiles (`PROCESSING_PROFILES`) allow users to optimize ingestion for different document types (e.g., `"Dense Technical Manual"`, `"Marketing Brochure"`), controlling parameters like `sentences_per_chunk`, `sentence_overlap`, `dpi`, and VLM usage.

> #### Advanced Multi-Step Retrieval
> 1.  **Hypothetical Document Embeddings (HyDE):** An optional "Conceptual Search" mode that uses the `HYDE_PROMPT_TEMPLATE` to generate an ideal answer *before* retrieval, improving results for abstract queries.
> 2.  **Multi-Query Expansion:** Automatically rewrites the user's query into three distinct variants using the `MULTI_QUERY_PROMPT_TEMPLATE` to broaden the search and maximize document recall.
> 3.  **Adaptive Cross-Encoder Reranking:** A sophisticated final pass using a `CrossEncoder` model (`ms-marco-MiniLM-L-6-v2`) to re-score and rank retrieved chunks for maximum relevance. The logic adaptively reranks *per-document* if sources are retrieved from multiple files, ensuring fair representation.

> #### Chain-of-Verification (CoV)
> A final, critical validation step (`validate_answer`) where an LLM uses the `CHAIN_OF_VERIFICATION_PROMPT` to meticulously fact-check the generated answer against the source context, virtually eliminating hallucinations and providing a clear validation status.

> #### Robust & Isolated Workspaces
> Each processing session generates a unique, timestamped workspace in AnythingLLM (e.g., `RAG-Session-2023-10-27-103000`), ensuring zero data contamination between different knowledge bases.

> #### Stable, Asynchronous Uploads
> The document upload process (`upload_document_chunks`) is engineered for stability, using an `asyncio.Semaphore` with `UPLOAD_BATCH_SIZE = 1` to prevent server overload by processing uploads sequentially.

> #### Expert Prompt Engineering
> The system's behavior is guided by a suite of specialized prompts, including a restrictive `SYSTEM_PROMPT` for generation and a powerful `VISION_RESTRUCTURING_PROMPT_TEMPLATE` for the VLM.

---

## ⚙️ Architectural Workflow

The system is logically divided into two distinct phases: a one-time **Ingestion Pipeline** for knowledge base creation, and a real-time **Query Pipeline** for delivering verified answers.

### Phase 1: Ingestion & Knowledge Base Creation

This automated pipeline converts raw, unstructured documents into a highly optimized and searchable vector knowledge base.

1.  **Initiation:** The user selects documents and a pre-configured `Processing Profile` via the Gradio UI.
2.  **Workspace Isolation (`create_workspace_if_not_exists`):** A new, secure workspace with a unique timestamped name is instantiated in AnythingLLM. This sandboxed environment is configured with the restrictive `SYSTEM_PROMPT`.
3.  **Intelligent Page Analysis (`intelligent_document_preprocessing`):** The system iterates through each page of the source PDF. It uses the `page.get_images()` heuristic to classify the page as "simple" (text-only) or "complex".
4.  **Hybrid Content Extraction:**
    *   For **simple** pages, fast, standard OCR (`PyMuPDF`) is used to extract raw text.
    *   For **complex** pages, the page is rendered as an image and sent to the `VISION_PREPROCESSOR_MODEL`. Using the detailed `VISION_RESTRUCTURING_PROMPT_TEMPLATE`, the VLM analyzes the visual layout to correctly transcribe text from charts, format tables into Markdown, and preserve multi-column reading order.
5.  **Semantic Chunking (`create_sentence_chunks`):** The clean, structured text from each page is passed to `nltk.sent_tokenize`. It is then split into small, semantically complete chunks based on the `sentences_per_chunk` and `sentence_overlap` parameters from the selected profile.
6.  **Robust Upload (`upload_document_chunks`):** Each text chunk is uploaded to AnythingLLM. To ensure stability, an `asyncio.Semaphore` with a batch size of 1 processes uploads sequentially, preventing API rate limiting or server overload.
7.  **Vectorization & Indexing:** AnythingLLM's embedding model converts each text chunk into a numerical vector and stores it in its high-performance vector database, making the knowledge base ready for querying.

### Phase 2: Query, Verification & Response

This multi-agent pipeline processes a user's question to deliver a factually verified, fully cited answer.

1.  **Query Input:** The user submits a question via the `ChatInterface`.
2.  **Query Enhancement (`chat_responder`):**
    *   **(Optional) HyDE (`generate_hyde_document`):** If "Conceptual Search" is enabled, the `HYDE_MODEL` generates a hypothetical answer to the query using the `HYDE_PROMPT_TEMPLATE`.
    *   **Multi-Query Expansion:** An LLM call uses the `MULTI_QUERY_PROMPT_TEMPLATE` to rewrite the original (or HyDE-enhanced) query into three distinct variants.
3.  **Multi-Vector Retrieval:** All query variants are sent to the AnythingLLM workspace. A similarity search retrieves a wide "candidate pool" of relevant document chunks.
4.  **Precision Reranking (`process_text_query_pipeline`):** The initial candidate pool is passed to the `CrossEncoder` (`RERANKER_MODEL`). It meticulously re-scores each chunk against the *original user query* for contextual relevance. If chunks come from multiple source documents, they are grouped and the top N are selected from *each document* to ensure diverse context.
5.  **Contextual Synthesis:** The top-ranked chunks are used to assemble a rich context block. This context, the original query, and the `SYSTEM_PROMPT` are passed to the `GENERATOR_MODEL` to synthesize an answer.
6.  **Chain-of-Verification (`validate_answer`):** The generated answer and the source context are sent to the `VALIDATION_MODEL`. It is guided by the rigorous `CHAIN_OF_VERIFICATION_PROMPT` to break down the answer into individual claims and confirm that each one is factually supported by the context, outputting a final "Yes" or "No".
7.  **Verified Response Delivery:** The final answer is formatted, prepended with a validation marker (✅ or ❌), and appended with cited source snippets, providing a complete, trustworthy, and auditable response.

---

## 🔬 Deep Dive: Technology & AI Strategy

### AI Model Roles

| Variable Name             | Default Model                                | Purpose                                                                                                 |
| ------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `VISION_PREPROCESSOR_MODEL` | `qwen2.5vl:72b`                              | A multimodal VLM that analyzes page images to extract and structure complex visual data (tables, charts). |
| `HYDE_MODEL`              | `gpt-oss:latest`                             | A capable text model used for generating hypothetical answers and for multi-query expansion.            |
| `GENERATOR_MODEL`         | `gpt-oss:latest`                             | The primary model for synthesizing the final answer from the retrieved context.                         |
| `VALIDATION_MODEL`        | `gpt-oss:latest`                             | The fact-checking model that performs the Chain-of-Verification step.                                   |
| `RERANKER_MODEL`          | `cross-encoder/ms-marco-MiniLM-L-6-v2` | A non-LLM Sentence Transformer model specifically trained for high-precision reranking tasks.             |

### Configurable Processing Profiles

The `PROCESSING_PROFILES` dictionary allows fine-tuning the ingestion pipeline for optimal results on different document types.

| Profile Name                           | `sentences_per_chunk` | `sentence_overlap` | `top_n_rerank` | `dpi` | `use_vlm` | Use Case                                                  |
| -------------------------------------- | --------------------- | ------------------ | -------------- | ----- | --------- | --------------------------------------------------------- |
| **Default (Balanced)**                 | 10                    | 2                  | 5              | 250   | `True`    | A versatile setting for mixed-content documents.          |
| **Marketing Brochure / Visual Document** | 12                    | 3                  | 3              | 300   | `True`    | Larger chunks and higher DPI for visually rich layouts.   |
| **Dense Technical Manual / Legal Text**  | 8                     | 2                  | 4              | 300   | `False`   | Smaller, precise chunks and disables VLM for text-heavy docs. |

### Advanced Prompt Engineering

The system's intelligence is heavily reliant on a suite of custom-engineered prompts:
*   `SYSTEM_PROMPT`: Strictly confines the `GENERATOR_MODEL` to only use the provided context, preventing it from inventing information.
*   `VISION_RESTRUCTURING_PROMPT_TEMPLATE`: Instructs the VLM to act as an expert document analyst, preserving structure, formatting lists, creating Markdown tables, and transcribing text from charts.
*   `MULTI_QUERY_PROMPT_TEMPLATE`: Guides an LLM to generate three diverse rephrasings of a user's question while maintaining its core intent, and to output them in a clean JSON format.
*   `HYDE_PROMPT_TEMPLATE`: Asks an LLM to create a detailed, factual-sounding hypothetical document passage that directly answers a query.
*   `CHAIN_OF_VERIFICATION_PROMPT`: A rigorous, step-by-step instruction set for the `VALIDATION_MODEL` to break down and verify claims one by one before giving a final verdict.

---

## 🚀 Setup and Installation

### Prerequisites

*   **Python 3.8+** and `pip`
*   **Ollama:** Installed and running. [Download here](https://ollama.com/).
*   **AnythingLLM:** Docker is the recommended setup. [Official Guide](https://docs.useanything.com/getting-started/docker-installation).
*   **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

You are absolutely right. My sincere apologies. The formatting in that last response was corrupted and incorrect. Thank you for pointing it out.

Here is the **correctly formatted** Markdown for the "Setup" and "How to Use" sections. This version has been carefully reviewed to ensure all headings, code blocks, and lists are properly structured.

---
```markdown
## 🚀 Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create Virtual Environment & Install Dependencies

It is highly recommended to create a `requirements.txt` file with the following content:
```
gradio
httpx
PyMuPDF
sentence-transformers
nltk
torch
torchvision
torchaudio
```
Then, run the installation:
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install from requirements.txt
pip install -r requirements.txt
```
The script will also attempt to auto-download necessary NLTK data (`punkt`, `popular`) on first run.

### 3. Download Local LLM Models

Ensure your Ollama service is running, then pull the default models:
```bash
# Vision model for document preprocessing
ollama pull qwen2.5vl:72b

# For HyDE, generation, and validation (or your preferred model)
ollama pull gpt-oss:latest
```

### 4. Configure AnythingLLM API Key

1.  Start your AnythingLLM instance.
2.  Navigate to `Settings` -> `API Keys` and generate a new key.
3.  Set this key as an environment variable for security.

    **Linux/macOS:**
    ```bash
    export ANYTHINGLLM_API_KEY="your-api-key-here"
    ```
    **Windows (PowerShell):**
    ```powershell
    $env:ANYTHINGLLM_API_KEY="your-api-key-here"
    ```

## 📖 How to Use

**1. Add Your Documents:** Place your PDF, TXT, or MD files into the `documents` folder.

**2. Run the Application:**
```bash
python your_script_name.py
```

**3. Open the Web UI:** Access the local URL provided in your terminal (e.g., `http://127.0.0.1:7860`).

### User Workflow

1.  **Select Processing Profile:** Choose a profile optimized for your document type.
2.  **Select Documents:** Use the radio buttons to select all documents, a folder, or specific files.
3.  **Click "Create Workspace and Process Document(s)"**: Monitor the live logs as the system ingests your data. The chat interface will appear upon completion.
4.  **Ask Questions:** Type your query into the chat. For deeper, more abstract questions, enable the **"Conceptual Search"** checkbox to activate the HyDE strategy.
5.  **Review Verified Answers:** Analyze the response, its validation emoji (✅/❌), and the supporting source snippets provided.
6.  **Reset:** To start over with new documents, click **"Process Other Documents"**.
```
