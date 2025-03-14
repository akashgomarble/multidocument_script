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
model = genai.GenerativeModel('gemini-pro')

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
            
    def get_documents_by_category(self) -> Dict[str, List[Dict[str, str]]]:
        """Get all documents grouped by category."""
        documents = {}
        for category, path in self.doc_categories.items():
            documents[category] = []
            if path.exists():
                for file_path in path.glob('*.*'):
                    if file_path.suffix.lower() in ['.txt', '.docx', '.pdf', '.csv']:
                        try:
                            content = self.read_document(file_path)
                            documents[category].append({
                                'name': file_path.name,
                                'content': content,
                                'type': file_path.suffix.lower()[1:]  # Remove the dot from extension
                            })
                            print(f"Successfully processed: {file_path.name}")
                        except Exception as e:
                            print(f"Error reading {file_path}: {str(e)}")
        return documents

class ScriptGenerator:
    def __init__(self, model):
        self.model = model
        
    def generate_script(self, documents: Dict[str, List[Dict[str, str]]]) -> str:
        """Generate a script using Gemini AI."""
        prompt = self._create_prompt(documents)
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating script: {str(e)}")
            return None
        
    def _create_prompt(self, documents: Dict[str, List[Dict[str, str]]]) -> str:
        """Create a detailed prompt for Gemini AI."""
        prompt = """Analyze the following documents and create a winning script with timestamps and reasoning.
For each section, explain which source document influenced the content and why it works well.

Documents provided:
"""
        for category, docs in documents.items():
            if docs:  # Only add category if it has documents
                prompt += f"\n{category.upper()} DOCUMENTS:\n"
                for doc in docs:
                    prompt += f"\nFile: {doc['name']} (Type: {doc['type']})\nContent:\n{doc['content']}\n"
                    
        prompt += """\nPlease create a comprehensive script that:
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
        return prompt

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate a script from multiple documents')
    parser.add_argument('--output', type=str, default=None, help='Output file path for the generated script')
    args = parser.parse_args()

    print("Starting Document Analysis & Script Generation...")
    
    # Initialize processor and generator
    processor = DocumentProcessor()
    generator = ScriptGenerator(model)
    
    # Process documents
    print("\nProcessing documents from all categories...")
    documents = processor.get_documents_by_category()
    
    # Check if we have documents to process
    total_docs = sum(len(docs) for docs in documents.values())
    if total_docs == 0:
        print("No documents found in any category. Please add documents to the respective folders.")
        return
    
    print(f"\nFound {total_docs} documents across all categories.")
    
    # Generate script
    print("\nGenerating script...")
    script = generator.generate_script(documents)
    
    if script:
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path("docs/scripts") / f"generated_script_{timestamp}.txt"
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the script
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script)
        
        print(f"\nScript generated successfully!")
        print(f"Output saved to: {output_path}")
        
        # Display the script in the console
        print("\nGenerated Script:")
        print("-" * 80)
        print(script)
        print("-" * 80)
    else:
        print("Failed to generate script. Please check the error messages above.")

if __name__ == "__main__":
    main()