import streamlit as st
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
import uuid

# Load environment variables
load_dotenv()

# Define available models and their configurations
AVAILABLE_MODELS = {
    'gemini-pro': {
        'name': 'Gemini Pro',
        'model_id': 'gemini-2.0-pro-exp-02-05',
        'max_tokens': 1000000,
        'description': 'Best for general text generation and analysis'
    },
    'gemini-flash-thinking': {
        'name': 'Gemini Flash Thinking',
        'model_id': 'gemini-2.0-flash-thinking-exp-01-21',
        'max_tokens': 1000000,
        'description': 'Optimized for text and image analysis'
    },
    'gemini-2.0-flash': {
        'name': 'Gemini 2.0 Flash',
        'model_id': 'gemini-2.0-flash',
        'max_tokens': 200000
    }
}

# Session state initialization for feedback history
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = {}

# Initialize feedback form state
if 'feedback_form' not in st.session_state:
    st.session_state.feedback_form = {
        'content_quality': 3,
        'relevance': 3,
        'structure': 3,
        'hooks_quality': 3,
        'specific_feedback': ''
    }

# Initialize current script state if not exists
if 'current_script' not in st.session_state:
    st.session_state.current_script = None

# Track if we need to show improved script
if 'show_improved_script' not in st.session_state:
    st.session_state.show_improved_script = False
    st.session_state.improved_script = None
    st.session_state.improved_script_path = None

class DocumentProcessor:
    def __init__(self):
        self.doc_categories = {
            'Product Description': Path('docs/product_description'),
            'Best Practices': Path('docs/best_practices'),
            'Sample Scripts': Path('docs/sample_scripts')
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
    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
        self.default_prompt_template = """
I need **10 performance marketing creative briefs** for Eskiin, with the following details:

#### **Document Categories Used:**
- **Product Description** - Contains in-depth details about the product, use cases, and customer personas
- **Best Practices** - Contains marketing and creative best practices for script creation
- **Sample Scripts** - Contains winning scripts that should be used as inspiration

#### **Writing Style Requirements:**
- Write in a natural, conversational human voice - avoid mechanical or AI-sounding language
- Use varied sentence structures and natural transitions between ideas
- Include casual language, contractions, and occasional colloquialisms where appropriate
- Ensure perfect grammar, spelling, and punctuation
- Create emotionally engaging content that resonates with the target audience
- Avoid repetitive phrases, robotic structures, or overly formal language

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

*Critical Note - Please give the reasoning behind the source documents and why they are used and which documents are used. Please give document wise reasoning and citations. Give reference in a proper format document name vrs reason and part of the document used.*  

*Important: Use the Product Description documents to understand the product details, target audience, and use cases. Reference Best Practices documents for creative and marketing guidelines. Draw inspiration from Sample Scripts to create new, original scripts that follow proven patterns of success.*

*Critical: The final scripts MUST read as if written by a skilled human copywriter. They should flow naturally, use varied sentence structures, and avoid any AI-like patterns or mechanical language. Double-check for perfect grammar and spelling. The scripts should feel authentic, conversational, and emotionally engaging - as if a creative professional wrote them specifically for this brand.Please give complete output don't skip anything.*
"""
        
    def generate_script(self, documents: Dict[str, List[Dict[str, str]]], custom_prompt_template: str = None) -> str:
        """Generate a script using the selected model."""
        prompt = self._create_prompt(documents, custom_prompt_template)
        
        try:
            response = self.client.models.generate_content(
                model=self.model_config['model_id'],
                contents=prompt
            )
            return response.text
        except Exception as e:
            st.error(f"Error generating with {self.model_config['name']}: {str(e)}")
            return None
        
    def _create_prompt(self, documents: Dict[str, List[Dict[str, str]]], custom_prompt_template: str = None) -> str:
        """Create a detailed prompt for the model."""
        prompt = f"""Analyze the following documents and create a winning script with timestamps and reasoning.
Using model: {self.model_config['name']}
For each section, explain which source document influenced the content and why it works well.

Documents provided:
"""
        for category, docs in documents.items():
            if docs:  # Only add category if it has documents
                prompt += f"\n{category.upper()} DOCUMENTS:\n"
                for doc in docs:
                    prompt += f"\nFile: {doc['name']} (Type: {doc['type']})\nContent:\n{doc['content']}\n"
                    
        # Use custom prompt template if provided, otherwise use default
        prompt += custom_prompt_template if custom_prompt_template is not None else self.default_prompt_template
        return prompt
    
    def improve_script(self, original_script: str, feedback_data: dict, documents: Dict[str, List[Dict[str, str]]], custom_prompt: str = None) -> str:
        """Improve a script based on client feedback."""
        prompt = f"""You previously generated this script:

{original_script}

The client has provided the following structured feedback:

Content Quality Rating: {feedback_data['content_quality']}/5
Relevance Rating: {feedback_data['relevance']}/5
Structure Rating: {feedback_data['structure']}/5
Hooks Quality Rating: {feedback_data['hooks_quality']}/5

Specific Feedback: {feedback_data['specific_feedback']}

Please improve the script based on this feedback. Consider the following documents that were used to create the original script:
"""
        
        for category, docs in documents.items():
            if docs:  # Only add category if it has documents
                prompt += f"\n{category.upper()} DOCUMENTS:\n"
                for doc in docs:
                    prompt += f"\nFile: {doc['name']} (Type: {doc['type']})\n"
        
        # Include the original prompt template if available
        if custom_prompt:
            prompt += f"\n\nThe original prompt template used was:\n\n{custom_prompt}\n\n"
        
        prompt += """
Generate an improved version of the script that addresses all the feedback points.
Maintain the same structure but enhance the content according to the feedback.

Writing Style Requirements:
- Write in a natural, conversational human voice - avoid mechanical or AI-sounding language
- Use varied sentence structures and natural transitions between ideas
- Include casual language, contractions, and occasional colloquialisms where appropriate
- Ensure perfect grammar, spelling, and punctuation
- Create emotionally engaging content that resonates with the target audience
- Avoid repetitive phrases, robotic structures, or overly formal language

In your response, include:
1. The improved script
2. A summary of changes made
3. How each feedback point was addressed
4. Suggestions for further improvements
"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_config['model_id'],
                contents=prompt
            )
            return response.text
        except Exception as e:
            st.error(f"Error improving script with {self.model_config['name']}: {str(e)}")
            return None

def save_uploaded_file(uploaded_file, save_dir: Path) -> Path:
    """Save an uploaded file and return its path."""
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / uploaded_file.name
    
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getvalue())
    return save_path

def process_feedback(generator, selected_model, custom_prompt=None):
    """Process feedback and generate improved script."""
    with st.spinner("Generating new script with updated feedback..."):
        # Get documents
        processor = DocumentProcessor()
        documents = processor.get_documents_by_category()
        
        # Determine which script to improve
        script_to_improve = st.session_state.current_script
        
        # Make a copy of the feedback data
        feedback_data = st.session_state.feedback_form.copy()
        
        # Generate improved script
        improved_script = generator.improve_script(
            script_to_improve,
            feedback_data,
            documents,
            custom_prompt
        )
        
        if improved_script:
            # Update session state
            if 'current_session_id' not in st.session_state:
                # Create new session ID if none exists
                session_id = str(uuid.uuid4())
                st.session_state.current_session_id = session_id
                st.session_state.feedback_history[session_id] = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'original_script': st.session_state.current_script,
                    'iterations': []
                }
            
            # Add new iteration
            st.session_state.feedback_history[st.session_state.current_session_id]['iterations'].append({
                'feedback_data': feedback_data,
                'improved_script': improved_script,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Update current script to the improved version
            st.session_state.current_script = improved_script
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("results") / selected_model / "improvements"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"improved_script_{timestamp}.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(improved_script)
            
            # Set flag to show improved script
            st.session_state.show_improved_script = True
            st.session_state.improved_script = improved_script
            st.session_state.improved_script_path = output_path
            
            # Reset feedback form for next iteration
            st.session_state.feedback_form = {
                'content_quality': 3,
                'relevance': 3,
                'structure': 3,
                'hooks_quality': 3,
                'specific_feedback': ''
            }
            
            return True
    
    return False

def main():
    st.title("Multi-Document Analysis & Script Generator")
    st.write("Upload multiple documents in each category to generate a comprehensive script!")
    
    # Model selection
    st.sidebar.header("Model Selection")
    selected_models = st.sidebar.multiselect(
        "Choose one or more models",
        options=list(AVAILABLE_MODELS.keys()),
        default=[list(AVAILABLE_MODELS.keys())[0]],  # Default to first model
        format_func=lambda x: AVAILABLE_MODELS[x]['name']
    )
    
    # Ensure at least one model is selected
  
    if not selected_models:
        st.sidebar.warning("Please select at least one model")
        selected_models = [list(AVAILABLE_MODELS.keys())[0]]  # Default to first model if none selected
    
    # Display selected models information
    st.sidebar.subheader("Selected Models:")
    for model_id in selected_models:
        st.sidebar.markdown(f"**{AVAILABLE_MODELS[model_id]['name']}** - Max Tokens: {AVAILABLE_MODELS[model_id]['max_tokens']}")
    
    # Use the first selected model as the primary model for document processing
    primary_model = selected_models[0]
    processor = DocumentProcessor()
    generator = ScriptGenerator(AVAILABLE_MODELS[primary_model])
    
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
    
    # Prompt template editor
    st.header("Customize Prompt Template")
    show_prompt_editor = st.checkbox("Edit Prompt Template", value=False)
    custom_prompt = None
    
    if show_prompt_editor:
        st.markdown("""
        ### Prompt Template Guidelines:
        - Use markdown formatting for better readability
        - Include clear instructions for the model
        - Specify desired output format
        - You can reference uploaded documents in your prompt
        """)
        
        custom_prompt = st.text_area(
            "Edit Prompt Template",
            value=generator.default_prompt_template,
            height=400,
            help="Customize the prompt template for this generation only. Changes won't be saved permanently."
        )
        
        if st.checkbox("Preview Final Prompt", value=False):
            st.markdown("### Final Prompt Preview:")
            documents = processor.get_documents_by_category()
            final_prompt = generator._create_prompt(documents, custom_prompt)
            st.markdown(custom_prompt)
    
    # Check if we need to show improved script from previous feedback
    if st.session_state.show_improved_script and st.session_state.improved_script:
        st.header("Improved Script")
        st.markdown(st.session_state.improved_script)
        
        # Add download button
        st.download_button(
            label="Download Improved Script",
            data=st.session_state.improved_script,
            file_name=f"improved_script.txt",
            mime="text/plain"
        )
        
        st.success(f"Script improved and saved to {st.session_state.improved_script_path}")
        st.info("You can provide additional feedback below to generate another improved version.")
        
        # Reset the flag
        st.session_state.show_improved_script = False
    
    # Generate initial script button
    if st.button(f"Generate Scripts with Selected Models"):
        # Get all documents
        documents = processor.get_documents_by_category()
        
        # Check if we have documents to process
        total_docs = sum(len(docs) for docs in documents.values())
        if total_docs == 0:
            st.warning("Please upload some documents first!")
            return
        
        # Create tabs for each model
        model_tabs = st.tabs([AVAILABLE_MODELS[model_id]['name'] for model_id in selected_models])
        
        # Generate scripts for each selected model
        for i, model_id in enumerate(selected_models):
            with model_tabs[i]:
                with st.spinner(f"Analyzing documents and generating script using {AVAILABLE_MODELS[model_id]['name']}..."):
                    try:
                        # Create generator for this model
                        model_generator = ScriptGenerator(AVAILABLE_MODELS[model_id])
                        
                        # Generate the script with custom prompt if provided
                        script = model_generator.generate_script(documents, custom_prompt if show_prompt_editor else None)
                        
                        if script:
                            # Display the generated script
                            st.header(f"Generated Script ({AVAILABLE_MODELS[model_id]['name']})")
                            st.markdown(script)
                            
                            # Store in session state for feedback system
                            if model_id == primary_model:
                                st.session_state.current_script = script
                                if 'current_session_id' in st.session_state:
                                    del st.session_state.current_session_id  # Start fresh session for new script
                            
                            # Save the generated script
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_dir = Path("results") / model_id
                            output_dir.mkdir(parents=True, exist_ok=True)
                            output_path = output_dir / f"generated_script_{timestamp}.txt"
                            
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(script)
                            
                            # Add download button for the generated script
                            st.download_button(
                                label=f"Download Generated Script",
                                data=script,
                                file_name=f"generated_script_{model_id}_{timestamp}.txt",
                                mime="text/plain"
                            )
                            
                            st.success(f"Script saved to {output_path}")
                            
                            # Save metadata
                            metadata = {
                                'model': model_id,
                                'model_name': AVAILABLE_MODELS[model_id]['name'],
                                'timestamp': timestamp,
                                'document_counts': {cat: len(docs) for cat, docs in documents.items()},
                                'used_custom_prompt': show_prompt_editor
                            }
                            
                            metadata_path = output_dir / f"metadata_{timestamp}.json"
                            with open(metadata_path, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=2)
                            
                            # Only show feedback system for the primary model
                            if model_id == primary_model:
                                st.header("Script Feedback & Improvement System")
                                st.write("Provide feedback to improve the generated script. You can submit multiple rounds of feedback.")
                                
                                # Display feedback history if available
                                if st.session_state.feedback_history and 'current_session_id' in st.session_state:
                                    session_id = st.session_state.current_session_id
                                    if session_id in st.session_state.feedback_history:
                                        session_data = st.session_state.feedback_history[session_id]
                                        
                                        # Show feedback iterations for current session
                                        if 'iterations' in session_data and session_data['iterations']:
                                            st.subheader("Previous Feedback Rounds")
                                            for i, iteration in enumerate(session_data['iterations']):
                                                with st.expander(f"Feedback Round {i+1} - {iteration['timestamp']}"):
                                                    # Display structured feedback
                                                    feedback_data = iteration['feedback_data']
                                                    st.markdown(f"**Specific Feedback:** {feedback_data['specific_feedback']}")
                                                    
                                                    st.markdown("**Improved Script:**")
                                                    st.markdown(iteration['improved_script'])
                                                    
                                                    # Add button to download this version
                                                    st.download_button(
                                                        label=f"Download Version {i+1}",
                                                        data=iteration['improved_script'],
                                                        file_name=f"improved_script_v{i+1}.txt",
                                                        mime="text/plain",
                                                        key=f"download_{session_id}_{i}"
                                                    )
                                
                                # Feedback form
                                st.subheader("Provide Your Feedback")
                                
                                st.session_state.feedback_form['specific_feedback'] = st.text_area(
                                    "What would you like to improve in this script?",
                                    value=st.session_state.feedback_form['specific_feedback'],
                                    height=150,
                                    help="Provide detailed feedback on what to improve, add, or change in the script.",
                                    key=f"feedback_text_{model_id}"
                                )
                                
                                feedback_submitted = st.button("Submit Feedback & Generate Improved Script", key=f"feedback_button_{model_id}")
                                if feedback_submitted:
                                    process_feedback(generator, primary_model, custom_prompt if show_prompt_editor else None)
                                    st.rerun()
                    
                    except Exception as e:
                        st.error(f"An error occurred with {AVAILABLE_MODELS[model_id]['name']}: {str(e)}")
    
    # Show feedback form if we have a current script but not from the generate button flow
    elif st.session_state.current_script is not None:
        st.header("Provide Feedback")
        st.session_state.feedback_form['specific_feedback'] = st.text_area(
            "What would you like to improve in this script?",
            value=st.session_state.feedback_form['specific_feedback'],
            height=150,
            help="Provide detailed feedback on what to improve, add, or change in the script."
        )
        
        feedback_submitted = st.button("Submit Feedback & Generate Improved Script")
        if feedback_submitted:
            process_feedback(generator, primary_model, custom_prompt if show_prompt_editor else None)
            st.rerun()

if __name__ == "__main__":
    main()
