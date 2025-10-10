import os
import asyncio
import httpx
import fitz
from sentence_transformers import CrossEncoder
import gradio as gr
import nltk
import base64
from datetime import datetime
from collections import defaultdict

# --- NLTK Resource Downloader ---
# Ensures that the required NLTK data packages are available.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("[INFO] NLTK 'punkt' model not found. Downloading...")
    nltk.download('punkt')
    print("[SUCCESS] 'punkt' model downloaded.")


try:
    nltk.data.find('corpora/stopwords') # Check for a common resource in the "popular" package
except LookupError:
    print("[INFO] NLTK 'popular' package not found. Downloading...")
    nltk.download('popular')
    print("[SUCCESS] 'popular' package downloaded.")


# --- 1. CORE CONFIGURATION ---
ANYTHINGLLM_API_URL = "http://localhost:3001/api/v1"
ANYTHINGLLM_API_KEY = os.getenv("ANYTHINGLLM_API_KEY", "3T2THCK-RYKM4F9-GMCMW4B-BG2NRBW") # Example: Use your actual API key
LOCAL_LLM_URL = "http://localhost:11434/api/generate"    # Ollama API
DOCUMENTS_FOLDER = "documents"
TEMP_FOLDER = "temp_processed"


# --- 2. RAG STRATEGY & MODEL CONFIGURATION ---
    
# --- START: ADD THIS ENTIRE CODE BLOCK ---

# Defines pre-configured sets of hyperparameters for different document types.
# The user will select one of these profiles in the UI.
PROCESSING_PROFILES = {
    # A balanced profile that works well for a mix of document types.
    "Default (Balanced)": {
        "sentences_per_chunk": 10, # Group 10 sentences into a chunk.
        "sentence_overlap": 2,     # The next chunk will start 2 sentences back.
        "top_n_rerank": 5,
        "dpi": 250,
        "use_vlm": True
    },
    
    # Tuned for visually rich documents where layout is critical.
    "Marketing Brochure / Visual Document": { # (12, 3, 3, 300, True)
        "sentences_per_chunk": 12,
        "sentence_overlap": 3,
        "top_n_rerank": 3,
        "dpi": 300,
        "use_vlm": True
    },

    # Tuned for precision in dense, text-heavy documents. SKIPS THE VLM.
    "Dense Technical Manual / Legal Text": { # (8, 2, 4, 300, False)
        "sentences_per_chunk": 8,  # Smaller, more precise chunks.
        "sentence_overlap": 2,     # Minimal overlap for factual density.
        "top_n_rerank": 4,
        "dpi": 300,
        "use_vlm": False
    }
}

# --- END OF CODE BLOCK TO ADD ---

WORKSPACE_CONFIG = {"name": "Hybrid RAG Workspace"}
# Ensure these models are available in your local Ollama instance. Run `ollama list` to check.
# This model MUST be a multimodal LLM available in your Ollama instance (e.g., llava)
VISION_PREPROCESSOR_MODEL = "qwen2.5vl:72b" # This model MUST be a multimodal LLM (e.g., llava)
HYDE_MODEL = "gpt-oss:latest"  # Model for generating hypothetical answers
GENERATOR_MODEL = "gpt-oss:latest"  # The model that generates the final answer
VALIDATION_MODEL = "gpt-oss:latest"
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
TOP_N_RERANK = 5
UPLOAD_BATCH_SIZE = 1



# --- 3. PROMPT CONFIGURATION ---
SYSTEM_PROMPT = """You are a restricted information-extraction AI. Your task is to answer the user's question using *only* the provided context. Do not add any information that is not explicitly stated in the text. Do not rephrase or summarize unless asked. Quote directly from the text where possible. If the answer is not in the context, you must respond with the exact phrase: "Based on the provided documents, I cannot answer that question."
"""

# --- START: MODIFICATION ---
    
# This new prompt is designed for the OCR + VLM restructuring strategy.

VISION_RESTRUCTURING_PROMPT_TEMPLATE = """You are an expert document analyst. Your mission is to convert a page image and its raw OCR text into perfectly structured Markdown.

You will receive:
1. An image of a document page.
2. The raw, jumbled text extracted from that page.

Your primary goal is to use the visual layout of the image to re-order and format the raw text.

**Core Rules to Follow:**
1.  **Structure is Paramount:** Your most important task is to preserve the document's structure. This includes:
    -   Correctly ordering text from columns.
    -   Formatting bullet points and numbered lists.
    -   Converting data tables into Markdown tables.
2.  **Transcribe from Visuals:** For complex visual elements like **charts, diagrams, screenshots, or flowcharts**, you MUST meticulously transcribe any and all text visible within them. This text is critical and should be formatted logically (e.g., as a list).
3.  **Use Raw Text as Reference:** The provided raw text is a good reference for content, but the visual layout of the image is your ultimate source of truth for structure and order.
4.  **Clean Up, Don't Invent:** Clean up minor OCR artifacts like extra line breaks, but do not add or invent information not present in the document.

Now, analyze the provided page image and its raw text. Produce the final, clean Markdown. Respond ONLY with the Markdown content.
"""
# --- END: MODIFICATION ---

# --- START: ADD THIS NEW PROMPT ---
MULTI_QUERY_PROMPT_TEMPLATE = """You are an expert query analyst. Your task is to take a user's question and generate three alternative versions of it. The goal is to rephrase the question in different ways to improve the chances of finding relevant documents. Maintain the core intent of the original question.

Original Question: {query}

Provide your response as a JSON list of strings only, with nothing before or after. For example:
["query 1", "query 2", "query 3"]

Rewritten Questions:
"""
# --- END OF NEW PROMPT ---

HYDE_PROMPT_TEMPLATE = """You are an expert at generating highly relevant, hypothetical document passages that directly answer a user's query. Your goal is to create a detailed, yet concise, hypothetical answer that would contain the information needed to respond to the user's question, even if you don't know the real answer.


The hypothetical document should be self-contained and sound like a factual excerpt from a document.


User Query: {query}


Hypothetical Document:"""


# --- START: REPLACE THE OLD VALIDATION PROMPT WITH THIS ---
CHAIN_OF_VERIFICATION_PROMPT = """You are an expert fact-checker performing a detailed verification. Follow these steps precisely:
1. Break down the 'Proposed Answer' into a list of individual claims.
2. For each claim, meticulously check if it is directly stated, a reasonable summary, or a logical inference that is undeniably supported by the 'Source Context'.
3. For each claim, state 'SUPPORTED' or 'NOT SUPPORTED'.
4. Finally, based on your analysis, provide a final verdict. If ALL claims are SUPPORTED, respond with 'Yes'. If ANY claim is NOT SUPPORTED, respond with 'No'.

Source Context:
{context}

Proposed Answer:
{answer}

Begin your step-by-step analysis. On the final line, write only the single-word verdict: 'Yes' or 'No'.
"""
# --- END OF REPLACEMENT ---

# --- 4. SETUP & CLIENTS ---
json_headers = {"Authorization": f"Bearer {ANYTHINGLLM_API_KEY}", "Content-Type": "application/json"}
upload_headers = {"Authorization": f"Bearer {ANYTHINGLLM_API_KEY}"}
async_client = httpx.AsyncClient(timeout=60.0)

try:
    reranker = CrossEncoder(RERANKER_MODEL)
    print(f"[INFO] Reranker model '{RERANKER_MODEL}' loaded successfully.")
except Exception as e:
    print(f"[FATAL ERROR] Could not load reranker model: {e}")
    reranker = None


# --- 5. CORE BACKEND LOGIC ---
async def create_workspace_if_not_exists(config, prompt):
    """Checks for an existing workspace by name, or creates and configures a new one."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            list_url = f"{ANYTHINGLLM_API_URL}/workspaces"
            response = await client.get(list_url, headers=json_headers)
            response.raise_for_status()
            workspaces = response.json().get('workspaces', [])
           
            for ws in workspaces:
                if ws['name'] == config['name']:
                    print(f"[INFO] Workspace '{config['name']}' already exists with slug '{ws['slug']}'.")
                    return ws['slug']
           
            print(f"[INFO] Workspace '{config['name']}' not found. Creating it...")
            create_url = f"{ANYTHINGLLM_API_URL}/workspace/new"
            create_payload = {"name": config['name']}
            response = await client.post(create_url, headers=json_headers, json=create_payload)
            response.raise_for_status()
            workspace = response.json().get("workspace")
            workspace_slug = workspace['slug']
            print(f"[SUCCESS] Workspace '{workspace_slug}' created.")
           
            update_url = f"{ANYTHINGLLM_API_URL}/workspace/{workspace_slug}/update"
            
            # --- CORRECTION 1: CRITICAL API BUG ---
            # The API documentation (pages 51-52) shows that the /update endpoint does not accept
            # a "chatModel" key. Sending it is incorrect. The payload is now corrected to only
            # include valid keys like "openAiPrompt".
            update_payload = {"openAiPrompt": prompt}
            
            await client.post(update_url, headers=json_headers, json=update_payload)
            print("[SUCCESS] Configured workspace with a system prompt.")
            return workspace_slug
           
    except Exception as e:
        print(f"[FATAL ERROR] during workspace setup for '{config['name']}': {e}")
        return None
        
async def get_current_vector_count():
    """Calls the AnythingLLM API to get the current total vector count."""
    try:
        url = f"{ANYTHINGLLM_API_URL}/system/vector-count"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=json_headers)
            response.raise_for_status()
            data = response.json()
            # The API returns 'vectorCount' with a capital C
            return data.get('vectorCount', 0)
    except Exception as e:
        print(f"[WARNING] Could not retrieve vector count: {e}")
        return -1 # Return an error indicator
        
    
async def make_local_llm_request(prompt, model, system_prompt=None, stream=False):
    """Async helper to call a local text-based LLM, with optional system prompt."""
    try:
        payload = {"model": model, "prompt": prompt, "stream": stream}
        # Add system prompt to payload if provided (for Ollama)
        if system_prompt:
            payload["system"] = system_prompt
            
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(LOCAL_LLM_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get('response', '').strip()
    except Exception as e:
        print(f"\n[ERROR] Could not connect to local LLM: {e}")
        return None

# You may need to modify this function if your local LLM client requires a different payload structure for multimodal inputs
async def make_local_multimodal_request(prompt: str, image_path: str, model: str):
    """Async helper to call a local multimodal LLM with an image."""
    try:
        # Encode image to base64
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # This payload structure is common for models like LLaVA with Ollama
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [encoded_image],
            "stream": False
        }
       
        async with httpx.AsyncClient(timeout=300.0) as client: # Increase timeout for VLM
            response = await client.post(LOCAL_LLM_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get('response', '').strip()

    except Exception as e:
        print(f"\n[ERROR] Could not connect to local Multimodal LLM: {e}")
        return None         
    
async def intelligent_document_preprocessing(pdf_path: str, temp_dir: str, dpi: int = 300, use_vlm: bool = True):
    """
    Conditionally processes a PDF. If use_vlm is False, it always uses fast OCR.
    If use_vlm is True, it analyzes each page and ONLY uses the VLM on pages
    that contain images or other complex visual elements.
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    processed_pages = []
    
    # --- The "Fast Path" for profiles where VLM is completely disabled ---
    if not use_vlm:
        print(f"[*] Starting fast OCR extraction (VLM disabled by profile) for: {pdf_path}...")
        try:
            with fitz.open(pdf_path) as doc:
                for i, page in enumerate(doc):
                    page_num = i + 1
                    raw_text = page.get_text("text")
                    if raw_text.strip():
                        processed_pages.append((page_num, raw_text))
            print(f"[SUCCESS] Fast extracted {len(processed_pages)} pages.")
            return processed_pages
        except Exception as e:
            print(f"[FATAL ERROR] during fast OCR extraction: {e}")
            raise e

    # --- The "Smart Path" where VLM is used conditionally per page ---
    print(f"[*] Starting Smart VLM Processing (analyzing pages individually) for: {pdf_path}...")
    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                page_num = i + 1

                # --- NEW: Pre-analysis heuristic ---
                # We check if the page contains any images. This is a fast and effective
                # proxy for "visual complexity". If no images, we use fast OCR.
                is_complex_page = bool(page.get_images())

                # --- Run VLM only if the page is complex ---
                if is_complex_page:
                    print(f"  - Page {page_num}/{len(doc)} is complex, using VLM...")
                    # Step 1: Render image
                    image_path = os.path.join(temp_dir, f"page_{page_num}.png")
                    pix = page.get_pixmap(dpi=dpi)
                    pix.save(image_path)
                    
                    # Step 2: Perform standard OCR
                    raw_text = page.get_text("text")
                    
                    # Step 3: Call VLM to restructure
                    prompt = VISION_RESTRUCTURING_PROMPT_TEMPLATE.format(raw_text=raw_text)
                    structured_content = await make_local_multimodal_request(prompt, image_path, VISION_PREPROCESSOR_MODEL)
                    
                    if structured_content:
                        processed_pages.append((page_num, structured_content))
                    else:
                        print(f"[WARNING] VLM returned no content for page {page_num}. Falling back to raw text.")
                        processed_pages.append((page_num, raw_text))
                    
                    # Step 4: Clean up
                    os.remove(image_path)
                
                # --- Use Fast OCR for simple pages ---
                else:
                    print(f"  - Page {page_num}/{len(doc)} is simple, using fast OCR...")
                    raw_text = page.get_text("text")
                    if raw_text.strip():
                        processed_pages.append((page_num, raw_text))

        print(f"[SUCCESS] Smart processed {len(processed_pages)} pages.")
        return processed_pages
    except Exception as e:
        print(f"[FATAL ERROR] during intelligent preprocessing: {e}")
        raise e
    
async def validate_answer(answer, context, model):
    """
    Uses an LLM to perform a chain-of-verification analysis to determine if
    the answer is supported by the context.
    """
    if not answer or not context:
        return False
    print("\n[*] Performing Chain-of-Verification for final answer...")
    
    # Use the new, more advanced prompt
    prompt = CHAIN_OF_VERIFICATION_PROMPT.format(context=context, answer=answer)
    decision_text = await make_local_llm_request(prompt, model)
    
    if decision_text:
        print(f"[INFO] Validator Analysis:\n{decision_text}")
        try:
            # The verdict is the last non-empty line of the response.
            # This is a robust way to parse the final "Yes" or "No".
            last_line = [line for line in decision_text.strip().split('\n') if line.strip()][-1]
            if "yes" in last_line.lower():
                print("[SUCCESS] Validation Passed.")
                return True
        except IndexError:
            # This happens if the model returns an empty string.
            print("[WARNING] Validation failed: Model returned an empty response.")
            return False

    print("[WARNING] Validation Failed.")
    return False
async def generate_hyde_document(query: str, model: str):
    """Uses an LLM to generate a hypothetical document based on the user's query."""
    print("[*] Generating hypothetical document (HyDE)...")
    prompt = HYDE_PROMPT_TEMPLATE.format(query=query)
    hyde_doc = await make_local_llm_request(prompt, model)
    if hyde_doc:
        print(f"[*] Generated HyDE Document: {hyde_doc[:100]}...") # Print first 100 chars
        return hyde_doc
    print("[WARNING] Failed to generate HyDE document.")
    return query # Fallback to original query


    
async def process_text_query_pipeline(original_query, rewritten_queries, text_slug, top_n: int = 5):
    """
    Performs retrieval, reranks results, and generates a validated answer.
    NEW: If results are from multiple documents, it reranks and selects the top_n
    chunks from EACH document to ensure fair representation.
    """
    yield "Step 2/5: Retrieving documents from multiple perspectives..."
    
    all_retrieved_sources = []
    try:
        async with httpx.AsyncClient() as client:
            for q in rewritten_queries:
                chat_payload = {"message": q, "mode": "query"}
                response = await client.post(
                    f"{ANYTHINGLLM_API_URL}/workspace/{text_slug}/chat", 
                    json=chat_payload, 
                    headers=json_headers, 
                    timeout=60.0
                )
                response.raise_for_status()
                all_retrieved_sources.extend(response.json().get('sources', []))
    except Exception as e:
        yield {"answer": f"Error during multi-query retrieval: {e}", "is_validated": False, "sources": []}
        return

    unique_sources_dict = {doc['id']: doc for doc in all_retrieved_sources}
    retrieved_sources = list(unique_sources_dict.values())

    if not retrieved_sources:
        yield {"answer": "Based on the documents, I could not find relevant information using multiple perspectives.", "is_validated": False, "sources": []}
        return

    yield "Step 2.5/5: Reranking results for relevance..."
    
    top_docs = []
    rerank_query = rewritten_queries[0]
    
    # --- NEW LOGIC STARTS HERE ---
    
    # --- 1. Group sources by their original document ---
    docs_by_origin = defaultdict(list)
    for doc in retrieved_sources:
        try:
            # Assumes title format like "base-filename_p1_chunk0.txt"
            base_filename = doc['title'].split('_p')[0]
            docs_by_origin[base_filename].append(doc)
        except Exception:
            docs_by_origin["unknown_source"].append(doc) # Fallback

    # --- 2. Conditionally rerank based on the number of source documents ---
    # If chunks from more than one original document were retrieved, process them group by group.
    if len(docs_by_origin) > 1:
        print(f"[INFO] Retrieved chunks from {len(docs_by_origin)} different documents. Reranking each document's results individually.")
        
        for base_filename, doc_list in docs_by_origin.items():
            if reranker and doc_list:
                # Rerank the chunks within this specific document group
                query_and_docs = [(rerank_query, doc['text']) for doc in doc_list]
                scores = reranker.predict(query_and_docs)
                for doc, score in zip(doc_list, scores):
                    doc['rerank_score'] = score
                
                reranked_group = sorted(doc_list, key=lambda x: x.get('rerank_score', 0), reverse=True)
                # Add the top N chunks from THIS document to the final list
                top_docs.extend(reranked_group[:top_n])
            else:
                top_docs.extend(doc_list[:top_n]) # Fallback if no reranker

    # Otherwise, use the original logic for a single document source.
    else:
        print("[INFO] Retrieved chunks from a single document source. Performing global rerank.")
        if reranker and retrieved_sources:
            query_and_docs = [(rerank_query, doc['text']) for doc in retrieved_sources]
            scores = reranker.predict(query_and_docs)
            for doc, score in zip(retrieved_sources, scores):
                doc['rerank_score'] = score
            reranked_docs = sorted(retrieved_sources, key=lambda x: x.get('rerank_score', 0), reverse=True)
            top_docs = reranked_docs[:top_n]
        else:
            top_docs = retrieved_sources[:top_n]

    # --- NEW LOGIC ENDS HERE ---
   
    # --- Context Assembly (remains the same) ---
    print("[*] Retrieving full-page context for top documents...")
    context_pages = {} 
    for doc in top_docs:
        try:
            title_parts = doc['title'].split('_p')
            base_filename = title_parts[0]
            page_num = title_parts[1].split('_')[0]
            full_page_path = os.path.join(TEMP_FOLDER, f"{base_filename}_p{page_num}_full.txt")
            if os.path.exists(full_page_path):
                with open(full_page_path, 'r', encoding='utf-8') as f:
                    context_pages[doc['title']] = f.read() # Use unique key
            else:
                print(f"[WARNING] Could not find full context file: {full_page_path}")
                context_pages[doc['title']] = doc['text']
        except Exception:
            print(f"[WARNING] Could not parse page number from title: {doc['title']}. Using chunk text as context.")
            context_pages[doc['title']] = doc['text']

    final_context = "\n\n---\n\n".join(context_pages.values())
    if not final_context:
        yield {"answer": "Could not assemble context from retrieved documents.", "is_validated": False, "sources": top_docs}
        return

    # --- Synthesis and Validation (remains the same) ---
    yield "Step 3/5: Synthesizing answer from context..."
    generation_prompt = f"""{SYSTEM_PROMPT}\n\n<context>\n{final_context}\n</context>\n\n<question>\n{original_query}\n</question>\n"""
    full_response = await make_local_llm_request(prompt=generation_prompt, model=GENERATOR_MODEL, stream=False)

    yield "Step 4/5: Validating answer for accuracy..."
    is_validated = await validate_answer(full_response, final_context, VALIDATION_MODEL)

    yield {
        "answer": full_response,
        "is_validated": is_validated,
        "sources": top_docs
    }

# --- CORRECTED FUNCTION ---
async def upload_document_chunks(chunks_with_pages, base_filename, workspace_slug):
    """
    Uploads text chunks as individual files, processing them in sequential batches
    to ensure maximum stability with sensitive servers, as required by the API.
    """
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)

    # This semaphore uses your UPLOAD_BATCH_SIZE = 1 setting to ensure
    # only one file is uploaded at a time. This is the key to stability.
    semaphore = asyncio.Semaphore(UPLOAD_BATCH_SIZE)
    
    # First, prepare all file paths on disk before starting the upload.
    all_filepaths = []
    for page_num, chunk_list in chunks_with_pages.items():
        for i, chunk in enumerate(chunk_list):
            processed_filename = f"{base_filename}_p{page_num}_chunk{i}.txt"
            processed_filepath = os.path.join(TEMP_FOLDER, processed_filename)
            with open(processed_filepath, "w", encoding="utf-8") as f:
                f.write(chunk)
            all_filepaths.append(processed_filepath)

    total_tasks = len(all_filepaths)
    successful_uploads = 0
    print(f"[*] Beginning upload of {total_tasks} text chunks in controlled, sequential batches...")

    # This loop processes the files one by one, waiting for each to finish.
    for i in range(0, total_tasks, UPLOAD_BATCH_SIZE):
        batch_filepaths = all_filepaths[i:i + UPLOAD_BATCH_SIZE]
        
        # Create the task for only the current file.
        batch_tasks = [
            asyncio.create_task(upload_original_document(fp, [workspace_slug], semaphore))
            for fp in batch_filepaths
        ]
        
        # This message will show you the progress in the console.
        print(f"  - Uploading batch {i//UPLOAD_BATCH_SIZE + 1}/{(total_tasks + UPLOAD_BATCH_SIZE - 1)//UPLOAD_BATCH_SIZE}...")
        
        # This line WAITS for the current file upload to complete before the loop continues.
        results = await asyncio.gather(*batch_tasks)
        successful_uploads += sum(1 for r in results if r)

    print(f"[SUCCESS] Upload process completed. {successful_uploads}/{total_tasks} chunks uploaded successfully.")

# --- END OF CORRECTED FUNCTION ---

# --- 6. DATA HANDLING & PRE-PROCESSING UTILITIES ---

def create_sentence_chunks(text: str, sentences_per_chunk: int, sentence_overlap: int):
    """
    Splits text into semantic chunks based on a sliding window of sentences.
    """
    print(f"[*] Creating sentence chunks (size: {sentences_per_chunk}, overlap: {sentence_overlap})...")
    
    # 1. Split the entire text into sentences
    sentences = nltk.sent_tokenize(text)
    
    # 2. Group sentences into chunks using a sliding window
    chunks = []
    if not sentences:
        return []

    # Calculate the step size for the sliding window
    step_size = sentences_per_chunk - sentence_overlap
    if step_size < 1:
        step_size = 1 # Ensure we always move forward

    for i in range(0, len(sentences), step_size):
        # Define the window for the current chunk
        start_index = i
        end_index = i + sentences_per_chunk
        
        # Get the sentences for the current chunk
        chunk_sentences = sentences[start_index:end_index]
        
        # Join the sentences back into a single string
        chunk_text = " ".join(chunk_sentences)
        
        # Add the chunk to our list if it's not empty
        if chunk_text.strip():
            chunks.append(chunk_text)

    print(f"[SUCCESS] Split text into {len(chunks)} sentence-based chunks.")
    return chunks
   
async def upload_original_document(filepath, workspace_slugs, semaphore):
    """Uploads a single document, respecting the concurrency limit set by the semaphore."""
    if not os.path.exists(filepath):
        print(f"[ERROR] Document not found at path: {filepath}")
        return False

    upload_url = f"{ANYTHINGLLM_API_URL}/document/upload"
    slug_string = ",".join(workspace_slugs)
    form_data = {'addToWorkspaces': slug_string}
    filename = os.path.basename(filepath)

    # The semaphore ensures only a limited number of this function's instances run concurrently
    async with semaphore:
        # A small delay can sometimes help with server stability under load
        await asyncio.sleep(0.1) 
        print(f"[*] Uploading '{filename}'...")
        try:
            with open(filepath, 'rb') as f:
                files_payload = {'file': (filename, f)}
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(upload_url, headers=upload_headers, files=files_payload, data=form_data)
            
            response.raise_for_status()

            if response.json().get("success"):
                return True
            else:
                print(f"[ERROR] API failure for '{filename}': {response.json().get('error')}")
                return False
        except httpx.HTTPStatusError as e:
            print(f"[FATAL SERVER ERROR] during upload of '{filename}': {e.response.text}")
            return False
        except Exception as e:
            print(f"[FATAL CLIENT ERROR] during upload of '{filename}': {e}")
            return False


# --- 7. GRADIO UI HANDLERS & LOGIC ---
# IMPROVEMENT 1: Real-time streaming feedback
# CORRECTED: This generator now yields a tuple of 6 values to update all relevant UI components.
async def process_documents_for_ui(document_names, profile, app_state_dict):
    """
    Processes a list of documents into a NEW, unique workspace for this session.
    Uses a dynamic processing profile for optimization.
    """
    if not document_names:
        raise gr.Error("No documents were selected for processing.")

    # --- UI State: Disable controls and start logging ---
    yield "Starting processing...", None, gr.update(), gr.update()
    log_output = ""

    # --- STEP 1: Create a NEW, UNIQUE workspace for this job ---
    try:
        log_output += "Step 1/5: Creating a new, isolated workspace for this session...\n"
        yield log_output, None, gr.update(), gr.update()
           
        # Generate a unique name for the workspace using a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        workspace_name = f"RAG-Session-{timestamp}"
        
        # Create the new workspace
        workspace_slug = await create_workspace_if_not_exists({"name": workspace_name}, SYSTEM_PROMPT)
        if not workspace_slug:
            raise RuntimeError("Failed to create a new workspace. Check AnythingLLM connection.")
        
        # Update the app state with the NEW slug for this session
        app_state_dict["workspace_slug"] = workspace_slug
        log_output += f"[SUCCESS] Created workspace '{workspace_name}' with slug '{workspace_slug}'.\n---\n"
        yield log_output, None, gr.update(), gr.update()

    except Exception as e:
        yield f"{log_output}\n[FATAL ERROR] {e}", None, gr.update(), gr.update()
        return
        
    total_chunks_processed = 0
    initial_vector_count = await get_current_vector_count()

    # --- STEP 2-4: Main loop to process each document ---
    for idx, document_name in enumerate(document_names):
        try:
            log_output += f"--- Processing Document {idx + 1}/{len(document_names)}: {document_name} ---\n"
            
            document_path = os.path.join(DOCUMENTS_FOLDER, document_name)
            base_filename = os.path.splitext(os.path.basename(document_name))[0]
            
            # STEP 2: OCR + VLM Restructuring (using the profile's DPI)
            log_output += "Step 2/5: Starting OCR + VLM content restructuring...\n"
            yield log_output, None, gr.update(), gr.update()
            temp_img_dir = os.path.join(TEMP_FOLDER, "images")
            pages = await intelligent_document_preprocessing(
                    document_path, temp_img_dir, dpi=profile["dpi"], use_vlm=profile["use_vlm"]
                    )
            if not pages:
                log_output += f"[WARNING] No content extracted from {document_name}. Skipping.\n"
                continue
            
            for page_num, structured_text in pages:
                full_page_filename = f"{base_filename}_p{page_num}_full.txt"
                with open(os.path.join(TEMP_FOLDER, full_page_filename), "w", encoding="utf-8") as f: f.write(structured_text)
            log_output += f"[SUCCESS] Restructured {len(pages)} pages.\n"

            # STEP 3: Semantic Chunking (using the profile's sentence window)
            log_output += "Step 3/5: Performing sentence-based semantic chunking...\n"
            yield log_output, None, gr.update(), gr.update()
            chunks_by_page = {
                page_num: create_sentence_chunks(
                    text, 
                    sentences_per_chunk=profile["sentences_per_chunk"], 
                    sentence_overlap=profile["sentence_overlap"]
                ) for page_num, text in pages
            }
            current_doc_chunks = sum(len(c) for c in chunks_by_page.values())
            total_chunks_processed += current_doc_chunks
            log_output += f"[SUCCESS] Split content into {current_doc_chunks} chunks.\n"

            # STEP 4: Uploading Chunks
            log_output += "Step 4/5: Uploading processed text chunks...\n"
            yield log_output, None, gr.update(), gr.update()
            await upload_document_chunks(chunks_by_page, base_filename, workspace_slug)
            log_output += f"[SUCCESS] Upload for {document_name} completed.\n"

        except Exception as e:
            log_output += f"\n[FATAL ERROR] processing {document_name}: {e}\n--- SKIPPING ---\n"
            continue
    
    # --- STEP 5: Final Verification ---
    log_output += f"\n--- Verifying total embeddings... ---\n"
    yield log_output, None, gr.update(), gr.update()

    expected_vector_count = initial_vector_count + total_chunks_processed
    max_wait_seconds, poll_interval_seconds, elapsed_time = 600, 5, 0

    while elapsed_time < max_wait_seconds:
        await asyncio.sleep(poll_interval_seconds)
        elapsed_time += poll_interval_seconds
        current_vector_count = await get_current_vector_count()
        
        if current_vector_count >= expected_vector_count:
            log_output += f"[SUCCESS] Embedding complete! Total vectors: {current_vector_count}.\n"
            log_output += "Setup Complete! The chat is now active.\n"
            # Pass the profile to the chat responder via the app_state
            app_state_dict["profile"] = profile
            yield log_output, "Multiple Documents Processed", gr.update(visible=False), gr.update(visible=True)
            return

        chunks_embedded = max(0, current_vector_count - initial_vector_count)
        progress_msg = f"Waiting for embeddings... ({chunks_embedded}/{total_chunks_processed}). Elapsed: {elapsed_time}s"
        yield f"{log_output}{progress_msg}", None, gr.update(), gr.update()

    log_output += f"[FATAL ERROR] Timeout after {max_wait_seconds}s. Embedding did not complete."
    error_message = f"{log_output}\n\n[FATAL ERROR] Timeout after {max_wait_seconds}s. Embedding did not complete. The server may be under heavy load or the number of documents is very large. Try again later."
    yield error_message, None, gr.update(visible=True), gr.update(visible=False)

# IMPROVEMENT 3: UI Reset function
def reset_ui():
    """Hides the chat interface, shows the document selection UI, and clears chat history and state."""
    # Return None to clear ChatInterface history and the processed_doc_path state
    return None, None, gr.update(visible=True), gr.update(visible=False)


async def chat_responder(query, chat_history, app_state_dict, current_doc_path, use_deep_search):
    """
    The main generator function that responds to user queries, now featuring
    Multi-Query Rewriting and calling the pipeline that uses Chain-of-Verification.
    """
    if not current_doc_path or "workspace_slug" not in app_state_dict:
        yield "Error: No document has been processed or the workspace is not properly configured."
        return

    workspace_slug = app_state_dict["workspace_slug"]
    response_data = None
    
    # --- STEP 1: HYDE (Optional) ---
    rewritten_query = query
    if use_deep_search:
        yield "Step 1/5: Generating hypothetical answer (HyDE)..."
        rewritten_query = await generate_hyde_document(query, HYDE_MODEL)
    else:
        yield "Step 1/5: Analyzing query..."

    # --- STEP 1.5: MULTI-QUERY REWRITING (New) ---
    yield "Step 1.5/5: Rephrasing query for broader context search..."
    all_queries = [rewritten_query] # Start with the original or HyDE-enhanced query
    try:
        multi_query_prompt = MULTI_QUERY_PROMPT_TEMPLATE.format(query=rewritten_query)
        # Use a capable model (like HYDE_MODEL) to generate query variations
        rewritten_queries_str = await make_local_llm_request(multi_query_prompt, HYDE_MODEL)
        
        # Robustly parse the JSON list of new queries and add them to our list
        import json
        # Ensure the response is treated as a raw string for JSON parsing
        parsed_queries = json.loads(rewritten_queries_str)
        all_queries.extend(parsed_queries)
        print(f"[INFO] Generated {len(all_queries)} queries for retrieval: {all_queries}")
    except Exception as e:
        print(f"[WARNING] Could not generate or parse multi-queries: {e}. Proceeding with single query.")
    
    # --- RAG PIPELINE (Now takes a list of queries) ---
    # Retrieve the profile and top_n setting that were saved for this session
    profile = app_state_dict.get("profile", PROCESSING_PROFILES["Default (Balanced)"])
    top_n = profile["top_n_rerank"]

    text_pipeline_generator = process_text_query_pipeline(
        original_query=query,
        rewritten_queries=all_queries, # Pass the full list of queries
        text_slug=workspace_slug,
        top_n=top_n
    )
    
    # Process the pipeline's output, yielding status updates
    async for update in text_pipeline_generator:
        if isinstance(update, str):
            yield update
        else:
            response_data = update

    if not response_data:
        yield "An unexpected error occurred and no response was generated."
        return

    yield "Step 5/5: Formatting final response..."

    # --- FINAL "PRETTY" MARKDOWN FORMATTING ---
    answer = response_data.get('answer', "No answer could be generated.")
    is_validated = response_data.get('is_validated', False)
    initial_sources = response_data.get('sources', [])
    
    # 1. Sanitize the main answer from the LLM.
    clean_answer = answer.strip() if answer else "No answer could be generated."
    
    # 2. Assemble the core response with clear validation status.
    validation_emoji = "✅" if is_validated else "❌"
    formatted_response = (
        f"{clean_answer}\n\n"
        f"---\n\n"
        f"**Validation:** {validation_emoji} _Answer supported by sources._"
    )
    
    final_sources = initial_sources
    
    # 3. Re-evaluate sources against the final, clean answer for best relevance.
    if reranker and clean_answer and initial_sources:
        print("[*] Re-evaluating sources against the final answer for accuracy...")
        try:
            answer_and_sources_pairs = [(clean_answer, source['text']) for source in initial_sources]
            relevance_scores = reranker.predict(answer_and_sources_pairs)
            for source, score in zip(initial_sources, relevance_scores):
                source['answer_relevance_score'] = score
            final_sources = sorted(initial_sources, key=lambda x: x.get('answer_relevance_score', 0), reverse=True)
            print("[SUCCESS] Sources re-ranked based on answer relevance.")
        except Exception as e:
            print(f"[WARNING] Could not re-rank sources against the answer: {e}")

    # 4. Assemble the "Sources" section with clean, readable formatting.
    if final_sources:
        formatted_response += "\n\n### Supporting Sources"
        for i, source in enumerate(final_sources, 1):
            
            # --- Source Title Formatting ---
            raw_title = source.get('title', 'Unknown Source')
            try:
                # Attempt to parse a user-friendly name, e.g., "chapter25 (Page 1)"
                base_filename = raw_title.split('_p')[0].replace('_', ' ').replace('-', ' ')
                page_part = raw_title.split('_p', 1)[1]
                page_num_str = page_part.split('_', 1)[0]
                source_display_title = f"**{i}. Source:** *{base_filename} (Page {page_num_str})*"
            except Exception:
                # Fallback for filenames that don't match the expected pattern
                source_display_title = f"**{i}. Source:** *{raw_title}*"

            # --- Snippet Sanitation and Formatting ---
            raw_snippet = source.get('text', 'No snippet available.')
            # Sanitize: remove extra newlines and whitespace for a clean, single-paragraph look.
            clean_snippet = ' '.join(raw_snippet.split()).strip()
            # Truncate and format using a Markdown blockquote for visual separation.
            display_snippet = (clean_snippet[:280] + '...') if len(clean_snippet) > 280 else clean_snippet
            
            formatted_response += f"\n\n{source_display_title}\n> {display_snippet}"
    
    yield formatted_response
    
# --- 8. UI DEFINITION & APP LAUNCH ---


def get_available_documents(folder=DOCUMENTS_FOLDER):
    """Scans a given folder and returns a list of supported files."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    supported_extensions = ['.pdf', '.txt', '.md']
    return [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in supported_extensions
    ]

def get_available_folders():
    """Scans the documents folder and returns a list of subdirectories."""
    if not os.path.exists(DOCUMENTS_FOLDER):
        os.makedirs(DOCUMENTS_FOLDER)
    return [
        d for d in os.listdir(DOCUMENTS_FOLDER)
        if os.path.isdir(os.path.join(DOCUMENTS_FOLDER, d))
    ]

def update_selection_ui(selection_mode):
    """Updates the visibility of UI components based on the user's radio button choice."""
    if selection_mode == "Select all documents":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    elif selection_mode == "Select a folder of documents":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    elif selection_mode == "Select one document":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    elif selection_mode == "Select multiple documents":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

async def process_button_handler(
    profile_name, selection_mode, folder_selection, single_doc_selection, multi_doc_selection, app_state_dict
):
    """
    Controller that gathers the document list and selected processing profile,
    then calls the main processing generator.
    """
    documents_to_process = []
    if selection_mode == "Select all documents":
        documents_to_process = get_available_documents()
    elif selection_mode == "Select a folder of documents":
        if folder_selection:
            folder_path = os.path.join(DOCUMENTS_FOLDER, folder_selection)
            documents_to_process = [os.path.join(folder_selection, doc) for doc in get_available_documents(folder_path)]
    elif selection_mode == "Select one document":
        if single_doc_selection:
            documents_to_process = [single_doc_selection]
    elif selection_mode == "Select multiple documents":
        if multi_doc_selection:
            documents_to_process = multi_doc_selection
    
    # Get the selected profile dictionary
    selected_profile = PROCESSING_PROFILES.get(profile_name, PROCESSING_PROFILES["Default (Balanced)"])

    # Yield from the main processor, passing the documents and the selected profile
    async for update in process_documents_for_ui(documents_to_process, selected_profile, app_state_dict):
        yield update

def build_ui(app_state):
    """Builds the Gradio Blocks UI with dynamic document and profile selection."""
    with gr.Blocks(theme=gr.themes.Soft(), title="Production-Grade Hybrid RAG") as demo:
        # Use an empty dictionary for the initial state. It will be populated by the processing function.
        app_state_obj = gr.State({}) 
        processed_doc_path = gr.State(None)

        gr.Markdown("# Production-Grade Hybrid RAG Application")
        
        with gr.Row(visible=True) as setup_row:
            with gr.Column():
                gr.Markdown("### 1. Select Processing Options and Documents")
                
                # NEW: Profile Selection Dropdown
                profile_dropdown = gr.Dropdown(
                    choices=list(PROCESSING_PROFILES.keys()),
                    value="Default (Balanced)",
                    label="Select Processing Profile",
                    info="Choose a profile that best matches your document type for optimized results."
                )

                # Document Selection Mode
                selection_mode = gr.Radio(
                    ["Select all documents", "Select a folder of documents", "Select one document", "Select multiple documents"],
                    label="Choose Document Selection Mode",
                    value="Select one document"
                )

                # Input controls
                folder_dropdown = gr.Dropdown(choices=get_available_folders(), label="Select a Folder", visible=False, interactive=True)
                single_doc_dropdown = gr.Dropdown(choices=get_available_documents(), label="Select a Document", visible=True, interactive=True)
                multi_doc_checkbox = gr.CheckboxGroup(choices=get_available_documents(), label="Select Documents", visible=False, interactive=True)
                
                process_button = gr.Button("Create Workspace and Process Document(s)", variant="primary")
                status_logs = gr.Textbox(label="Processing Logs", lines=10, interactive=False, max_lines=20)
       
        with gr.Row(visible=False) as chat_row:
            with gr.Column():
                gr.Markdown("### 2. Ask Questions About Processed Documents")
                deep_search_toggle = gr.Checkbox(label="Conceptual Search", info="Slower. Rephrases your question...")
                chat_ui = gr.ChatInterface(
                    fn=chat_responder,
                    additional_inputs=[app_state_obj, processed_doc_path, deep_search_toggle]
                )
                reset_button = gr.Button("Process Other Documents")

        # --- Event Listeners ---
        selection_mode.change(
            fn=update_selection_ui,
            inputs=selection_mode,
            outputs=[folder_dropdown, single_doc_dropdown, multi_doc_checkbox]
        )
        
        process_button.click(
            fn=process_button_handler,
            inputs=[profile_dropdown, selection_mode, folder_dropdown, single_doc_dropdown, multi_doc_checkbox, app_state_obj],
            outputs=[status_logs, processed_doc_path, setup_row, chat_row]
        )
       
        reset_button.click(
            fn=reset_ui,
            inputs=[],
            outputs=[chat_ui, processed_doc_path, setup_row, chat_row]
        )
    return demo

# Cell 3: Initialize and Launch the Application

# This is the main execution block for the notebook
if __name__ == "__main__":
    if reranker is None:
        print("[FATAL] Reranker model failed to load. The application cannot continue.")
        exit() # Exit the script
   
    else:
        # Build the UI. The initial state is now an empty dictionary.
        app_ui = build_ui({})

        print("--- Launching Gradio UI ---")
        app_ui.launch(share=True)





