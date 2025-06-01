import os
from flask import Flask, render_template, request, jsonify
import PyPDF2
import pandas as pd
import openai
from dotenv import load_dotenv
import re
from werkzeug.exceptions import RequestEntityTooLarge
import json
import hashlib
from pathlib import Path
import gc  # For garbage collection

load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = "https://api.openai.com/v1"
openai.timeout = 30  # 30 seconds timeout for API calls

# Cache configuration
CACHE_DIR = Path('cache')
CACHE_DIR.mkdir(exist_ok=True)
SUMMARIES_CACHE_FILE = CACHE_DIR / 'summaries_cache.json'

# Upload folder configuration
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max file size
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_pdf_fingerprint(pdf_path):
    """Generate a fingerprint for the first page of the PDF."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            if len(pdf_reader.pages) > 0:
                first_page = pdf_reader.pages[0].extract_text()
                # Create a hash of the first page content
                return hashlib.md5(first_page.encode()).hexdigest()
    except Exception as e:
        print(f"Warning: Could not generate PDF fingerprint: {str(e)}")
    return None

def load_cache():
    """Load the summaries cache from file."""
    if SUMMARIES_CACHE_FILE.exists():
        try:
            with open(SUMMARIES_CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache: {str(e)}")
    return {}

def save_cache(cache_data):
    """Save the summaries cache to file."""
    try:
        with open(SUMMARIES_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        print(f"Warning: Could not save cache: {str(e)}")

# Initialize cache
summaries_cache = load_cache()

def clean_text(text):
    """Clean up text by fixing common PDF formatting issues."""
    # Fix common spacing issues
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Add space between lower and uppercase letters
    text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces
    text = re.sub(r'([a-z])-\s+([a-z])', r'\1\2', text)  # Fix hyphenation
    # Fix specific PDF formatting issues
    text = text.replace('a ttacks', 'attacks')  # Common PDF error
    text = text.replace('ADUL T', 'ADULT')  # Fix known typo
    return text.strip()

def extract_wrong_answer_rates(pdf_path):
    # Read the PDF using PyPDF2
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    # Define patterns to match question numbers and percentages
    # This pattern looks for:
    # 1. 1-3 digits (question number)
    # 2. Followed by optional whitespace
    # 3. Followed by either:
    #    - A percentage (1-3 digits followed by %)
    #    - The word "percent" or "%"
    patterns = [
        r"(\d{1,3})\s*(\d{1,3})%",  # Matches "371 60%"
        r"(\d{1,3})\s+(\d{1,3})\s*percent",  # Matches "371 60 percent"
        r"Question\s+(\d{1,3})[^\d]*?(\d{1,3})%"  # Matches "Question 371...60%"
    ]
    
    high_error_questions = {'60-79': [], '80+': []}
    found_questions = set()  # To avoid duplicates
    
    # Process each pattern
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                question_num = int(match.group(1))
                percentage = int(match.group(2))
                
                # Skip if we've already processed this question
                if question_num in found_questions:
                    continue
                
                if 60 <= percentage < 80:
                    high_error_questions['60-79'].append(question_num)
                    found_questions.add(question_num)
                elif percentage >= 80:
                    high_error_questions['80+'].append(question_num)
                    found_questions.add(question_num)
            except (ValueError, IndexError) as e:
                print(f"Error processing match {match.group()}: {str(e)}")
                continue
    
    # Sort the question numbers for better presentation
    high_error_questions['60-79'].sort()
    high_error_questions['80+'].sort()
    
    # Print debug information
    print(f"\nFound {len(high_error_questions['60-79'])} questions with 60-79% wrong answers:")
    print(high_error_questions['60-79'])
    print(f"\nFound {len(high_error_questions['80+'])} questions with 80%+ wrong answers:")
    print(high_error_questions['80+'])
    print(f"\nTotal questions found: {len(found_questions)}")
    
    return high_error_questions

def extract_general_category(pdf_reader, page_num):
    """Extract the general category from the top of the page."""
    try:
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        
        # Look for category in first few lines
        lines = text.split('\n')[:3]  # Check first 3 lines
        for line in lines:
            # Common patterns for general categories
            if any(pattern in line.upper() for pattern in [
                'DISORDERS', 'EPILEPSY', 'DISEASE', 'INFECTION', 'INJURY',
                'NEUROMUSCULAR', 'CEREBROVASCULAR', 'HEADACHE', 'NEURO-ONCOLOGY'
            ]):
                # Clean up the category text
                category = line.strip()
                
                # Remove any numbers (including page numbers) and their surrounding parentheses
                category = re.sub(r'\s*\(\d+.*?\)\s*', '', category)
                category = re.sub(r'^\d+\.?\s*', '', category)
                category = re.sub(r'\s*\d+\s*$', '', category)
                
                # Remove any remaining parenthetical content
                category = re.sub(r'\s*\(.*?\)\s*', '', category)
                
                # Standardize some common categories
                category_mapping = {
                    'BEHAVIORAL/NEUROCOGNITIVE DISORDERS': 'Behavioral/Neurocognitive Disorders',
                    'EPILEPSY AND EPISODIC DISORDERS': 'Epilepsy and Episodic Disorders',
                    'CEREBROVASCULAR DISEASE': 'Cerebrovascular Disease',
                    'NEUROMUSCULAR DISORDERS': 'Neuromuscular Disorders',
                    'NEURO-ONCOLOGY': 'Neuro-Oncology',
                    'NEUROIMAGING': 'Neuroimaging',
                    'NEUROIMMUNOLOGY': 'Neuroimmunology',
                    'HEADACHE': 'Headache Disorders',
                    'MOVEMENT DISORDERS': 'Movement Disorders',
                    'NEURODEGENERATIVE DISORDERS': 'Neurodegenerative Disorders',
                    'CRITICAL CARE NEUROLOGY': 'Critical Care Neurology',
                    'PEDIATRIC NEUROLOGY': 'Pediatric Neurology'
                }
                
                # Standardize the category name
                clean_category = category.strip()
                upper_category = clean_category.upper()
                
                # Try to match with standardized categories
                for standard_upper, standard_proper in category_mapping.items():
                    if standard_upper in upper_category:
                        return standard_proper
                
                # If no match found, return the cleaned category with proper capitalization
                return clean_category.title()
                
    except Exception as e:
        print(f"Error extracting general category from page {page_num}: {str(e)}")
    return "Uncategorized"

def extract_question_info(pdf_path, question_numbers):
    question_info = {}
    
    try:
        # Read the PDF to get page categories and full text
        page_categories = {}
        full_text = ""
        
        # Process PDF in smaller chunks to manage memory
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            # Process pages in chunks of 10
            for chunk_start in range(0, total_pages, 10):
                chunk_end = min(chunk_start + 10, total_pages)
                chunk_text = ""
                
                for page_num in range(chunk_start, chunk_end):
                    try:
                        page = pdf_reader.pages[page_num]
                        chunk_text += f"[PAGE {page_num + 1}]\n{page.extract_text()}\n"
                        page_categories[page_num] = extract_general_category(pdf_reader, page_num)
                    except Exception as e:
                        print(f"Warning: Error extracting text from page {page_num + 1}: {str(e)}")
                        page_categories[page_num] = "Uncategorized"
                
                full_text += chunk_text
                # Force garbage collection after each chunk
                gc.collect()
        
        # Process questions in even smaller batches
        batch_size = 5  # Reduced from 10 to 5
        for i in range(0, len(question_numbers), batch_size):
            batch = question_numbers[i:i + batch_size]
            
            for q_num in batch:
                try:
                    # Process question and generate summary
                    patterns = [
                        f"Question #{q_num}\\s+[A-Z]",
                        f"\\n{q_num}\\s+[A-Z]",
                        f"^{q_num}\\s+[A-Z]",
                        f"[^0-9]{q_num}\\s+[A-Z]"
                    ]
                    
                    # Find the first matching pattern
                    start_idx = -1
                    matched_text = ""
                    current_page = 0
                    
                    for pattern in patterns:
                        matches = list(re.finditer(pattern, full_text))
                        if matches:
                            for match in matches:
                                pos = match.start()
                                # Find which page this question is on
                                page_markers = list(re.finditer(r'\[PAGE (\d+)\]', full_text[:pos]))
                                if page_markers:
                                    current_page = int(page_markers[-1].group(1)) - 1
                                
                                next_q = re.search(r'Question #\d{1,3}|\\n\d{1,3}\\s+[A-Z]', full_text[pos+1:pos+500])
                                if not next_q or (next_q and int(re.search(r'\d+', next_q.group()).group()) > q_num):
                                    start_idx = pos
                                    matched_text = match.group()
                                    break
                            if start_idx != -1:
                                break
                    
                    if start_idx == -1:
                        print(f"Warning: Question {q_num} not found with primary patterns, trying fallback patterns")
                        fallback_patterns = [
                            f"{q_num}\\s+",
                            f"Question\\s+{q_num}",
                            f"\\n{q_num}\\s+"
                        ]
                        for pattern in fallback_patterns:
                            matches = list(re.finditer(pattern, full_text))
                            if matches:
                                start_idx = matches[0].start()
                                matched_text = matches[0].group()
                                # Find which page this question is on
                                page_markers = list(re.finditer(r'\[PAGE (\d+)\]', full_text[:start_idx]))
                                if page_markers:
                                    current_page = int(page_markers[-1].group(1)) - 1
                                break
                    
                    if start_idx == -1:
                        print(f"Warning: Could not find content for Question {q_num}")
                        question_info[q_num] = f"Question {q_num} information not found"
                        continue
                    
                    # Rest of your existing question processing code...
                    next_q_pattern = r'Question #\d{1,3}|\n\d{1,3}\s+[A-Z]|\[PAGE \d+\]'
                    next_q_match = re.search(next_q_pattern, full_text[start_idx + len(matched_text):])
                    
                    if next_q_match:
                        end_idx = start_idx + len(matched_text) + next_q_match.start()
                    else:
                        end_idx = len(full_text)
                    
                    question_text = full_text[start_idx:end_idx].strip()
                    question_text = re.sub(r'\[PAGE \d+\]\s*', ' ', question_text)
                    lines = [line.strip() for line in question_text.split('\n') if line.strip()]
                    
                    if not lines:
                        print(f"Warning: No content found for Question {q_num}")
                        question_info[q_num] = f"No content found for Question {q_num}"
                        continue
                    
                    first_line = lines[0]
                    category_text = re.sub(f'^(?:Question\\s+#{q_num}|{q_num})\\s*', '', first_line).strip()
                    parts = [p.strip() for p in re.split(r'\s{2,}', category_text) if p.strip()]
                    
                    category_parts = []
                    subcategory = None
                    
                    for part in parts:
                        cleaned_part = clean_text(part)
                        if cleaned_part.isupper() or any(phrase in cleaned_part for phrase in ["CORE KNOWLEDGE"]):
                            category_parts.append(cleaned_part)
                        elif not subcategory and cleaned_part[0].isupper():
                            subcategory = cleaned_part
                    
                    category = " ".join(category_parts) if category_parts else "Category Not Found"
                    
                    if not subcategory and len(lines) > 1:
                        second_line = clean_text(lines[1])
                        if not second_line.isupper() and second_line[0].isupper():
                            subcategory = second_line
                    
                    content_start = 2 if subcategory in lines[1:2] else 1
                    content = " ".join(lines[content_start:])
                    content = clean_text(content)
                    
                    if not content:
                        print(f"Warning: No content extracted for Question {q_num}")
                        content = "Content not found"
                    
                    reference = None
                    ref_match = re.search(r'(?:Reference|References):\s*([^\n]+)', content, re.IGNORECASE)
                    if ref_match:
                        reference = clean_text(ref_match.group(1))
                        content = clean_text(content[:ref_match.start()].strip())
                    
                    print(f"\nProcessing Question {q_num}:")
                    print(f"Category: {category}")
                    print(f"Subcategory: {subcategory}")
                    print(f"Content length: {len(content)} characters")
                    
                    prompt = (
                        f"You are analyzing a medical examination question. Based on the following information, provide a concise summary:\n\n"
                        f"Question Number: {q_num}\n"
                        f"Category: {category}\n"
                        f"Subcategory: {subcategory}\n"
                        f"Content: {content}\n\n"
                        f"Please provide a brief, specific summary that covers:\n"
                        f"1. The exact medical knowledge or concept being tested\n"
                        f"2. Why this specific topic is important for medical residents\n"
                        f"Keep the summary focused and under 100 words."
                    )
                    
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.5,
                            max_tokens=150
                        )
                        
                        summary = response.choices[0].message.content
                    except Exception as e:
                        print(f"Warning: Error generating summary for Question {q_num}: {str(e)}")
                        summary = "Error generating summary"
                    
                    # Get the general category for this question from the page it's on
                    general_category = page_categories.get(current_page, "Uncategorized")
                    
                    result = {
                        'category': category,
                        'subcategory': subcategory,
                        'general_category': general_category,  # Add the general category
                        'content': content,
                        'reference': reference,
                        'summary': summary
                    }
                    
                    question_info[q_num] = result
                    
                except Exception as e:
                    print(f"Error processing question {q_num}: {str(e)}")
                    error_result = {
                        'category': 'Error',
                        'subcategory': None,
                        'general_category': 'Error',
                        'content': f"Error processing question: {str(e)}",
                        'reference': None,
                        'summary': 'Error processing question'
                    }
                    question_info[q_num] = error_result
        
        return question_info
    except Exception as e:
        print(f"Error in extract_question_info: {str(e)}")
        raise

def generate_teaching_points(question_info_60_79, question_info_80_plus):
    """Generate key teaching points based on question summaries."""
    # Combine all summaries
    all_summaries = []
    for questions in [question_info_60_79, question_info_80_plus]:
        for info in questions.values():
            if isinstance(info, dict) and 'summary' in info:
                all_summaries.append(info['summary'])
    
    if not all_summaries:
        return []

    prompt = f"""As a chief resident, analyze these question summaries from commonly missed RITE exam questions and provide key teaching points. Focus on:
1. Common themes and patterns
2. Critical knowledge gaps
3. High-yield topics for resident education
4. Practical teaching strategies

Summaries to analyze:
{' '.join(all_summaries)}

Provide a concise, bullet-pointed list of 5-7 key teaching points that would be most valuable for chief residents."""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        teaching_points = response.choices[0].message.content.strip()
        return teaching_points
    except Exception as e:
        print(f"Error generating teaching points: {str(e)}")
        return "Error generating teaching points. Please try again."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'wrong_rates_pdf' not in request.files or 'manual_pdf' not in request.files:
            return jsonify({'error': 'Both PDF files are required'}), 400
        
        wrong_rates_file = request.files['wrong_rates_pdf']
        manual_file = request.files['manual_pdf']
        
        if not wrong_rates_file.filename or not manual_file.filename:
            return jsonify({'error': 'Both PDF files must be selected'}), 400
        
        wrong_rates_path = os.path.join(app.config['UPLOAD_FOLDER'], 'wrong_rates.pdf')
        manual_path = os.path.join(app.config['UPLOAD_FOLDER'], 'manual.pdf')
        
        try:
            wrong_rates_file.save(wrong_rates_path)
            manual_file.save(manual_path)
        except Exception as e:
            return jsonify({'error': f'Error saving files: {str(e)}'}), 500
        
        try:
            high_error_questions = extract_wrong_answer_rates(wrong_rates_path)
            if not isinstance(high_error_questions, dict) or '60-79' not in high_error_questions or '80+' not in high_error_questions:
                return jsonify({'error': 'Invalid format in wrong answer rates extraction'}), 500
            
            all_questions = high_error_questions['60-79'] + high_error_questions['80+']
            if not all_questions:
                return jsonify({'error': 'No questions found with high error rates'}), 400
            
            question_info = extract_question_info(manual_path, all_questions)
            if not isinstance(question_info, dict):
                return jsonify({'error': 'Invalid question information format'}), 500
            
            questions_60_79 = {q: question_info[q] for q in high_error_questions['60-79']}
            questions_80_plus = {q: question_info[q] for q in high_error_questions['80+']}
            
            teaching_points = generate_teaching_points(questions_60_79, questions_80_plus)
            
            # Initialize statistics
            stats = {
                'subcategory_stats': {'60-79': {}, '80+': {}},
                'general_category_stats': {'60-79': {}, '80+': {}},
                'population_stats': {'60-79': {}, '80+': {}}
            }
            
            # Process categories for both ranges
            for range_key in ['60-79', '80+']:
                questions = high_error_questions[range_key]
                for q_num in questions:
                    if q_num in question_info and isinstance(question_info[q_num], dict):
                        info = question_info[q_num]
                        
                        # Subcategory statistics
                        if info.get('subcategory'):
                            stats['subcategory_stats'][range_key][info['subcategory']] = \
                                stats['subcategory_stats'][range_key].get(info['subcategory'], 0) + 1
                        
                        # General category statistics
                        if info.get('general_category'):
                            stats['general_category_stats'][range_key][info['general_category']] = \
                                stats['general_category_stats'][range_key].get(info['general_category'], 0) + 1
                        
                        # Population statistics
                        category = info.get('category', '')
                        pop_type = 'Adult' if 'ADULT' in category else 'Pediatric' if 'PEDIATRIC' in category else 'Not Specified'
                        stats['population_stats'][range_key][pop_type] = \
                            stats['population_stats'][range_key].get(pop_type, 0) + 1
            
            result = {
                'stats': {
                    '60-79': len(high_error_questions['60-79']),
                    '80+': len(high_error_questions['80+']),
                    'categories': stats['subcategory_stats'],
                    'general_categories': stats['general_category_stats'],
                    'population': stats['population_stats']
                },
                'teaching_points': teaching_points if teaching_points else "No teaching points generated",
                'questions_60_79': questions_60_79,
                'questions_80_plus': questions_80_plus
            }
            
            # Validate the result before sending
            json.dumps(result)  # This will raise an error if the result is not JSON-serializable
            
            return jsonify(result)
            
        except Exception as e:
            print(f"Error processing PDFs: {str(e)}")
            return jsonify({'error': f'Error processing PDFs: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
        
    finally:
        # Clean up uploaded files
        for path in [wrong_rates_path, manual_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"Error removing temporary file {path}: {str(e)}")

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        'error': 'File too large',
        'message': 'The uploaded file exceeds the maximum allowed size of 64MB. Please try with a smaller file.'
    }), 413

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # In production, the port will be set by the environment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 