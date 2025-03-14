import streamlit as st
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
import tempfile
from google import genai


# Load environment variables
load_dotenv()

# Configure Gemini AI


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
                        except Exception as e:
                            st.error(f"Error reading {file_path}: {str(e)}")
        return documents

class ScriptGenerator:
    def __init__(self, model):
        self.model = model
        
    def generate_script(self, documents: Dict[str, List[Dict[str, str]]]) -> str:
        """Generate a script using Gemini AI."""
        prompt = self._create_prompt(documents)
        client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        return response.text
        
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
                    
        prompt += """

I need **10 performance marketing creative briefs** for Eskiin, with the following details:

#### **Key Components for Each Brief:**
1. **Elevator Pitch of Script Idea** – A concise summary of the ad's core message.
2. **Intended Length** – Duration of the ad (30-90 seconds, max 3 mins).
3. **Talent POV** – Who is speaking? (e.g., Founder, Employee, Female Influencer/Customer, Male Influencer/Customer).
4. **Production Style** – Type of content (e.g., UGC, Talking Head + Voice Over, B-Roll + Voice Over, B-Roll Only, etc.).
5. **Editing Style** – Format of the edit (e.g., Educational, Social Proof, Listicle, Organic TH, etc.).
6. **Problem/Solution Marketing Framework** – How the ad presents a problem and its solution.
7. **Brand Talking Points** – Key messages that must be included.
8. **Intended Awareness Phase** – Which part of the customer journey the ad targets.
9. **Customer Persona** – The target audience for this ad.
10. **Winning Ad Transcripts** – A structured breakdown of the ad.

#### **Output Structure for Each Brief:**
1. **Main Script in Table Format**  
   - **Audio** (Voiceover or dialogue)  
   - **Visual** (On-screen elements, B-roll, or scene descriptions)  

2. **Hook Variations:**  
   - **Three Unique Hook Visuals**  
   - **Three Unique Hook Headlines**  

3. **Debrief Analysis:**  
   - **Script Name** – Follows the correct naming convention (Template in Google Drive).  
   - **Hook Breakdown** – Three hooks and corresponding visuals (formatted appropriately).  
   - **Awareness Phase Breakdown** – Explanation of how the ad aligns with a specific phase and messaging strategies.  
   - **Ad Length** – Duration of the ad.  
   - **Ad Type** – Talking Head (TH), Voice Over (VO), or Static.  
   - **Editing Style** – (Educational, Listicle, Social Proof, Organic, VSL, etc.).  
   - **Call-To-Action (CTA)** – The CTA used in the ad.  
   - **Key Messaging** – Core brand talking points covered in the ad.  

#### **Additional Requirements:**
- **Each video should not exceed 3 minutes.**  
- **Most ads will be 30-90 seconds long.**  
- **Each ad must include:**  
  - **Three unique hook headlines.**  
  - **Three unique scroll-stopping visuals.** 
"""
        return prompt

def save_uploaded_file(uploaded_file, save_dir: Path) -> Path:
    """Save an uploaded file and return its path."""
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / uploaded_file.name
    
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getvalue())
    return save_path

def main():
    st.title("Multi-Document Analysis & Script Generator")
    st.write("Upload multiple documents in each category to generate a comprehensive script!")
    
    processor = DocumentProcessor()
    generator = ScriptGenerator(model)
    
    # File uploader for each category with multiple file support
    for category in processor.doc_categories.keys():
        st.header(f"{category.title()} Documents")
        uploaded_files = st.file_uploader(f"Upload {category} documents", 
                                        type=['txt', 'docx', 'pdf', 'csv'],
                                        accept_multiple_files=True,
                                        key=category)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                save_path = save_uploaded_file(uploaded_file, processor.doc_categories[category])
                st.success(f"Saved {uploaded_file.name} to {category} folder")
    
    # Add document list viewer
    if st.checkbox("Show uploaded documents"):
        for category, path in processor.doc_categories.items():
            if path.exists():
                files = list(path.glob('*.*'))
                if files:
                    st.subheader(f"{category.title()} Documents:")
                    for file_path in files:
                        st.text(f"- {file_path.name}")
    
    if st.button("Generate Script"):
        with st.spinner("Analyzing documents and generating script..."):
            try:
                # Get all documents
                documents = processor.get_documents_by_category()
                
                # Check if we have documents to process
                total_docs = sum(len(docs) for docs in documents.values())
                if total_docs == 0:
                    st.warning("Please upload some documents first!")
                    return
                
                # Generate the script
                script = generator.generate_script(documents)
                
                # Display the generated script
                st.header("Generated Script")
                st.markdown(script)
                
                # Save the generated script
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path("docs/scripts") / f"generated_script_{timestamp}.txt"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(script)
                
                # Add download button for the generated script
                with open(output_path, 'r') as f:
                    st.download_button(
                        label="Download Generated Script",
                        data=f.read(),
                        file_name=f"generated_script_{timestamp}.txt",
                        mime="text/plain"
                    )
                
                st.success(f"Script saved to {output_path}")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 