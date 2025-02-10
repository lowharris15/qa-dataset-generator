import os
import streamlit as st
import PyPDF2
import json
import traceback
import anthropic
from openai import OpenAI
import google.generativeai as genai
import requests
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import docx
import datetime
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIManager:
    def __init__(self, provider: str):
        self.provider = provider.split()[0].lower()
        self.available_models = {}
        self.client = None
        self.setup_client()

    def setup_client(self):
        """Initialize the appropriate AI client based on provider."""
        try:
            if self.provider == "anthropic":
                if not st.session_state.get('anthropic_api_key'):
                    raise ValueError("Anthropic API key not found")
                self.client = anthropic.Anthropic(api_key=st.session_state.anthropic_api_key)
            
            elif self.provider == "openai":
                if not st.session_state.get('openai_api_key'):
                    raise ValueError("OpenAI API key not found")
                self.client = OpenAI(api_key=st.session_state.openai_api_key)
            
            elif self.provider == "google":
                if not st.session_state.get('google_api_key'):
                    raise ValueError("Google API key not found")
                genai.configure(api_key=st.session_state.google_api_key)
            
            elif self.provider == "ollama":
                self.base_url = "http://localhost:11434"
            
            logger.info(f"Successfully set up client for {self.provider}")
        
        except Exception as e:
            logger.error(f"Error setting up client for {self.provider}: {str(e)}")
            raise

    def get_available_models(self) -> Dict[str, Dict[str, str]]:
        """Get available models for the current provider."""
        try:
            if self.provider == "openai" and self.client:
                models = self.client.models.list()
                self.available_models = {
                    model.id: {"id": model.id}
                    for model in models
                    if any(prefix in model.id for prefix in ["gpt-4", "gpt-3.5"])
                }
                
            elif self.provider == "anthropic" and self.client:
                self.available_models = {
                    "claude-3-opus-20240229": {"id": "claude-3-opus-20240229"},
                    "claude-3-sonnet-20240229": {"id": "claude-3-sonnet-20240229"},
                    "claude-3-haiku-20240307": {"id": "claude-3-haiku-20240307"},
                    "claude-2.1": {"id": "claude-2.1"}
                }
                
            elif self.provider == "google":
                self.available_models = {
                    "gemini-pro": {"id": "gemini-pro"},
                    "gemini-pro-vision": {"id": "gemini-pro-vision"}
                }
                
            elif self.provider == "ollama":
                try:
                    response = requests.get(f"{self.base_url}/api/tags")
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        self.available_models = {
                            model["name"]: {"id": model["name"]}
                            for model in models
                        }
                except Exception as e:
                    logger.error(f"Error getting Ollama models: {str(e)}")
            
            return dict(sorted(self.available_models.items()))  # Return models in alphabetical order
        
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return {}

    def generate_qa_pairs(self, text: str, context: str, model_id: str) -> List[Dict[str, str]]:
        try:
            prompt = f"""Input Text:
{text}

Context:
{context}

Request:
Provide five question and answer pairs based on the text above. The questions must begin with "In the context of". The answers should borrow verbatim from the text above. In providing each question, consider that the reader does not see or have access to any of the other questions for context. Vary the style and format of questions that improves training. Respond in plain text on a new line for each question and answer. Do not include question numbers.

Example Format:
In the context of [topic], what is [specific aspect]?
[Answer borrowed verbatim from the text]

In the context of [topic], how does [specific aspect] work?
[Answer borrowed verbatim from the text]"""

            # Generate response based on provider
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=model_id,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                raw_qa = response.content[0].text
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                raw_qa = response.choices[0].message.content
            elif self.provider == "google":
                model = genai.GenerativeModel(model_id)
                response = model.generate_content(prompt)
                raw_qa = response.text
            elif self.provider == "ollama":
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={"model": model_id, "prompt": prompt}
                )
                raw_qa = response.json().get("response", "")

            # Parse the response into Q&A pairs
            qa_pairs = []
            lines = raw_qa.strip().split('\n')
            current_question = None
            current_answer = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("In the context of"):
                    if current_question and current_answer:
                        qa_pairs.append({
                            "question": current_question,
                            "answer": current_answer
                        })
                    current_question = line
                    current_answer = None
                elif current_question and not line.startswith("In the context of"):
                    current_answer = line
                    qa_pairs.append({
                        "question": current_question,
                        "answer": current_answer
                    })
                    current_question = None
                    current_answer = None

            # Add the last pair if exists
            if current_question and current_answer:
                qa_pairs.append({
                    "question": current_question,
                    "answer": current_answer
                })

            return qa_pairs

        except Exception as e:
            logger.error(f"Error generating QA pairs: {str(e)}")
            return []

class DatasetProcessor:
    def __init__(self, ai_manager, output_file='qa_dataset.json'):
        self.ai_manager = ai_manager
        self.output_file = output_file
        self.qa_pairs = []

    def extract_text_from_pdf(self, pdf_file) -> List[str]:
        text_chunks = []
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                # Improved chunking to avoid breaking mid-sentence
                paragraphs = text.split('\n\n')
                for para in paragraphs:
                    if len(para) > 1000:
                        chunks = [para[i:i+1000] for i in range(0, len(para), 1000)]
                        text_chunks.extend(chunks)
                    else:
                        text_chunks.append(para)
            return text_chunks
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return []

    def extract_text_from_docx(self, docx_file) -> List[str]:
        text_chunks = []
        try:
            doc = docx.Document(docx_file)
            for para in doc.paragraphs:
                if para.text.strip():
                    if len(para.text) > 1000:
                        chunks = [para.text[i:i+1000] for i in range(0, len(para.text), 1000)]
                        text_chunks.extend(chunks)
                    else:
                        text_chunks.append(para.text)
            return text_chunks
        except Exception as e:
            st.error(f"Error extracting text from DOCX: {str(e)}")
            return []

    def process_document(self, file, context: str, selected_model: str):
        if not file:
            st.error("Please upload a file")
            return

        try:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            file_extension = Path(file.name).suffix.lower()
            if file_extension == '.pdf':
                text_chunks = self.extract_text_from_pdf(file)
            elif file_extension in ['.doc', '.docx']:
                text_chunks = self.extract_text_from_docx(file)
            else:
                st.error("Unsupported file format")
                return

            if not text_chunks:
                st.error("No text could be extracted from the document")
                return

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, chunk in enumerate(text_chunks):
                try:
                    qa_pairs = self.ai_manager.generate_qa_pairs(chunk, context, selected_model)
                    self.qa_pairs.extend(qa_pairs)

                    if i == 0 and qa_pairs:
                        st.write("### Sample Q&A Pairs")
                        for j, qa in enumerate(qa_pairs[:3], 1):
                            st.markdown(f"""
                            **Q{j}:** {qa['question']}
                            
                            **A{j}:** {qa['answer']}
                            
                            ---
                            """)

                    progress = (i + 1) / len(text_chunks)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing chunk {i + 1} of {len(text_chunks)}")

                except Exception as e:
                    st.error(f"Error processing chunk {i + 1}: {str(e)}")

            output_path = output_dir / self.output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "source_file": file.name,
                        "context": context,
                        "model": selected_model,
                        "total_qa_pairs": len(self.qa_pairs),
                        "total_chunks": len(text_chunks),
                        "average_pairs_per_chunk": len(self.qa_pairs)/len(text_chunks),
                        "generation_timestamp": str(datetime.datetime.now())
                    },
                    "qa_pairs": self.qa_pairs
                }, f, indent=2, ensure_ascii=False)

            status_text.text("Processing complete!")
            st.success(f"Q&A dataset saved to {output_path}")
            
            st.markdown("### Dataset Statistics")
            st.markdown(f"""
            - **Total Q&A pairs generated:** {len(self.qa_pairs)}
            - **Total text chunks processed:** {len(text_chunks)}
            - **Average Q&A pairs per chunk:** {len(self.qa_pairs)/len(text_chunks):.1f}
            """)

            if self.qa_pairs:
                st.markdown("### Random Sample Q&A Pairs")
                random_samples = random.sample(self.qa_pairs, min(3, len(self.qa_pairs)))
                for i, qa in enumerate(random_samples, 1):
                    st.markdown(f"""
                    **Q{i}:** {qa['question']}
                    
                    **A{i}:** {qa['answer']}
                    
                    ---
                    """)

        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            logger.error(traceback.format_exc())

def main():
    st.set_page_config(page_title="Q&A Dataset Generator", layout="wide")
    st.title("Q&A Dataset Generator")

    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {}

    with st.sidebar:
        st.header("Settings")
        
        ai_provider = st.selectbox(
            "Select AI Provider",
            ["OpenAI", "Anthropic", "Google", "Ollama"],
            key="ai_provider"
        )

        api_key = st.text_input(
            f"{ai_provider} API Key",
            type="password",
            key=f"{ai_provider.lower()}_api_key_input"
        )

        if api_key:
            st.session_state[f"{ai_provider.lower()}_api_key"] = api_key

    try:
        ai_manager = AIManager(ai_provider)
        available_models = ai_manager.get_available_models()
        
        if available_models:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_model = st.selectbox(
                    "Select AI Model",
                    options=list(available_models.keys()),
                    key="selected_model"
                )

                uploaded_file = st.file_uploader(
                    "Upload document",
                    type=["pdf", "doc", "docx"]
                )

            with col2:
                context = st.text_area(
                    "Enter context for Q&A generation",
                    help="This context will be used to frame the questions appropriately",
                    height=100
                )

                output_file = st.text_input(
                    "Output JSON filename",
                    value="qa_dataset.json"
                )

            if st.button("Generate Q&A Dataset", type="primary"):
                if uploaded_file is not None and context:
                    processor = DatasetProcessor(ai_manager, output_file)
                    processor.process_document(uploaded_file, context, selected_model)
                else:
                    st.error("Please upload a file and provide context")
        else:
            st.error(f"No models available for {ai_provider}. Please check your API key and try again.")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()
