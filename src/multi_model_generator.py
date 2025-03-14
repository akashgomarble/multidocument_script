import google.generativeai as genai
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from typing import Dict, List, Union
import docx
import PyPDF2
import argparse

# Load environment variables
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Define available models
AVAILABLE_MODELS = {
    'gemini-pro': genai.GenerativeModel('gemini-pro'),
    'gemini-pro-vision': genai.GenerativeModel('gemini-pro-vision'),
    # Add other models as needed
}

def count_tokens(model, text: str) -> int:
    """Count the number of tokens in a string using model's tokenizer."""
    try:
        result = model.count_tokens(text)
        return result.total_tokens
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        return 0

class DocumentProcessor:
    def __init__(self):
        self.doc_categories = {
            'marketing': Path('docs/marketing'),
            'hooks': Path('docs/hooks'),
            'scripts': Path('docs/scripts'),
            'products': Path('docs/products')
        }
        
    def read_text_file(self, file_path: Path) -> str:
        """Read content from a text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    def read_docx_file(self, file_path: Path) -> str:
        """Read content from a DOCX file."""
        doc = docx.Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        
    def read_pdf_file(self, file_path: Path) -> str:
        """Read content from a PDF file."""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return '\n'.join([page.extract_text() for page in pdf_reader.pages])
            
    def read_csv_file(self, file_path: Path) -> str:
        """Read content from a CSV file and return a formatted string."""
        df = pd.read_csv(file_path)
        return f"CSV Data Summary:\nColumns: {', '.join(df.columns)}\n\nFirst few rows:\n{df.head().to_string()}\n\nKey Statistics:\n{df.describe().to_string()}"
            
    def read_document(self, file_path: Path) -> str:
        """Read document content based on file extension."""
        if file_path.suffix.lower() == '.txt':
            return self.read_text_file(file_path)
        elif file_path.suffix.lower() == '.docx':
            return self.read_docx_file(file_path)
        elif file_path.suffix.lower() == '.pdf':
            return self.read_pdf_file(file_path)
        elif file_path.suffix.lower() == '.csv':
            return self.read_csv_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
    def get_documents_by_category(self, model) -> Dict[str, List[Dict[str, str]]]:
        """Get all documents grouped by category."""
        documents = {}
        total_tokens = 0
        
        for category, path in self.doc_categories.items():
            documents[category] = []
            if path.exists():
                for file_path in path.glob('*.*'):
                    if file_path.suffix.lower() in ['.txt', '.docx', '.pdf', '.csv']:
                        try:
                            content = self.read_document(file_path)
                            num_tokens = count_tokens(model, content)
                            total_tokens += num_tokens
                            
                            documents[category].append({
                                'name': file_path.name,
                                'content': content,
                                'type': file_path.suffix.lower()[1:],
                                'tokens': num_tokens
                            })
                            print(f"Successfully processed: {file_path.name} ({num_tokens} tokens)")
                        except Exception as e:
                            print(f"Error reading {file_path}: {str(e)}")
        
        print(f"\nTotal tokens across all documents: {total_tokens}")
        return documents

class ScriptGenerator:
    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.max_tokens = 30000  # Default token limit
        
    def generate_script(self, documents: Dict[str, List[Dict[str, str]]]) -> str:
        """Generate a script using the specified model."""
        prompt = self._create_prompt(documents)
        prompt_tokens = count_tokens(self.model, prompt)
        print(f"\nPrompt size for {self.model_name}: {prompt_tokens} tokens")
        
        if prompt_tokens > self.max_tokens:
            print(f"Warning: Total tokens ({prompt_tokens}) exceeds model's limit ({self.max_tokens})")
            print("Attempting to truncate content while preserving essential information...")
            prompt = self._truncate_prompt(documents)
            new_tokens = count_tokens(self.model, prompt)
            print(f"Truncated prompt size: {new_tokens} tokens")
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating script with {self.model_name}: {str(e)}")
            return None
        
    def _create_prompt(self, documents: Dict[str, List[Dict[str, str]]]) -> str:
        """Create a detailed prompt for the model."""
        prompt = f"""Analyze the following documents and create a winning script with timestamps and reasoning.
Model being used: {self.model_name}

Documents provided:
"""
        for category, docs in documents.items():
            if docs:
                prompt += f"\n{category.upper()} DOCUMENTS:\n"
                for doc in docs:
                    prompt += f"\nFile: {doc['name']} (Type: {doc['type']})\nContent:\n{doc['content']}\n"
        
        prompt += self._get_prompt_template()
        return prompt
        
    def _truncate_prompt(self, documents: Dict[str, List[Dict[str, str]]]) -> str:
        """Create a truncated prompt that fits within token limits."""
        prompt = f"""Analyze the following key excerpts from documents and create a winning script.
Model being used: {self.model_name}
Note: Content has been truncated to fit within limits while preserving key information.

Documents provided:
"""
        current_tokens = count_tokens(self.model, prompt)
        
        num_categories = len([cat for cat in documents.keys() if documents[cat]])
        tokens_per_category = (self.max_tokens - current_tokens) // (num_categories + 1)
        
        for category, docs in documents.items():
            if not docs:
                continue
                
            category_prompt = f"\n{category.upper()} DOCUMENTS:\n"
            current_tokens += count_tokens(self.model, category_prompt)
            prompt += category_prompt
            
            tokens_per_doc = tokens_per_category // len(docs)
            for doc in docs:
                doc_header = f"\nFile: {doc['name']} (Type: {doc['type']})\nContent:\n"
                current_tokens += count_tokens(self.model, doc_header)
                prompt += doc_header
                
                content = doc['content']
                while count_tokens(self.model, content) > tokens_per_doc:
                    if doc['type'] == 'csv':
                        lines = content.split('\n')
                        content = '\n'.join(lines[:10]) + "\n[Content truncated...]"
                    else:
                        words = content.split()
                        content = ' '.join(words[:len(words)//2]) + "\n[Content truncated...]"
                
                prompt += content + "\n"
        
        prompt += self._get_prompt_template()
        return prompt
    
    def _get_prompt_template(self) -> str:
        """Get the standard prompt template."""
        return """\nPlease create a comprehensive script that:
1. Incorporates the best elements from each document category
2. Includes timestamps for each section
3. Explains the reasoning behind each chosen element
4. References the source documents used (including file names)
5. Maintains a cohesive and engaging flow
6. Synthesizes information from multiple documents within each category
7. Highlights data insights from CSV files when relevant

Format the response as:
[Timestamp] Section Title
Content
Sources: [Document Names]
Reasoning: [Explanation including why specific documents were chosen]
Data Insights: [If applicable, include relevant data points from CSV files]
"""

def process_with_models(models: List[str], output_dir: str = "results"):
    """Process documents with multiple models and save results."""
    processor = DocumentProcessor()
    base_output_dir = Path(output_dir)
    
    for model_name in models:
        if model_name not in AVAILABLE_MODELS:
            print(f"Warning: Model {model_name} not available. Skipping...")
            continue
            
        model = AVAILABLE_MODELS[model_name]
        print(f"\nProcessing with model: {model_name}")
        
        # Create model-specific output directory
        model_output_dir = base_output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process documents
        documents = processor.get_documents_by_category(model)
        
        if not any(docs for docs in documents.values()):
            print("No documents found in any category. Skipping...")
            continue
        
        # Generate script
        generator = ScriptGenerator(model, model_name)
        script = generator.generate_script(documents)
        
        if script:
            # Save the script
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = model_output_dir / f"generated_script_{timestamp}.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(script)
            
            # Save metadata
            metadata = {
                'model': model_name,
                'timestamp': timestamp,
                'document_counts': {cat: len(docs) for cat, docs in documents.items()},
                'token_counts': {
                    'total_input': sum(doc['tokens'] for docs in documents.values() for doc in docs),
                    'output': count_tokens(model, script)
                }
            }
            
            metadata_path = model_output_dir / f"metadata_{timestamp}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\nScript generated successfully with {model_name}!")
            print(f"Output saved to: {output_path}")
            print(f"Metadata saved to: {metadata_path}")
        else:
            print(f"Failed to generate script with {model_name}")

def main():
    parser = argparse.ArgumentParser(description='Generate scripts using multiple models')
    parser.add_argument('--models', nargs='+', default=['gemini-pro'],
                      help='List of models to use for generation')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Base output directory for results')
    args = parser.parse_args()
    
    process_with_models(args.models, args.output_dir)

if __name__ == "__main__":
    main() 