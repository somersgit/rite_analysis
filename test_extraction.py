import os
import PyPDF2
import re

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

def extract_question_info(pdf_path, question_numbers):
    question_info = {}
    
    # Read the manual PDF
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        full_text = ""
        for page in pdf_reader.pages:
            # Add page breaks to help identify content boundaries
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n[PAGE_BREAK]\n"
    
    # For each question number, find relevant information and generate summary
    for q_num in question_numbers:
        try:
            # More flexible pattern matching for question numbers
            patterns = [
                f"Question #{q_num}",
                f"{q_num}  ",  # Note the double space after number
                f"\n{q_num}  ",  # Number at start of line
                f"{q_num} "  # Single space after number
            ]
            
            # Find the first matching pattern
            start_idx = -1
            matched_pattern = ""
            for pattern in patterns:
                try:
                    start_idx = full_text.index(pattern)
                    matched_pattern = pattern
                    break
                except ValueError:
                    continue
            
            if start_idx == -1:
                question_info[q_num] = "Question information not found"
                continue
            
            # Look for the next question or section break
            # Find the end of the current section
            end_markers = [
                "\nQuestion #",
                "\n[PAGE_BREAK]\n",
                "\nReference:",
                "\nReferences:",
                "\\n\\d{3}\\s+"  # Pattern for next question number (properly escaped)
            ]
            
            # Find the closest end marker
            end_idx = len(full_text)
            for marker in end_markers:
                if marker.startswith("\\n\\d"):
                    # For number patterns, use regex to find next question
                    matches = list(re.finditer(marker, full_text[start_idx:]))
                    if matches:
                        pos = start_idx + matches[0].start()
                        if pos > start_idx and pos < end_idx:
                            end_idx = pos
                else:
                    pos = full_text.find(marker, start_idx + len(matched_pattern))
                    if pos != -1 and pos < end_idx:
                        end_idx = pos
            
            # Extract question text and clean it up
            question_text = full_text[start_idx:end_idx].strip()
            
            # Extract category and subcategory
            lines = [line.strip() for line in question_text.split('\n') if line.strip()]
            
            # The first line should contain the question number and category
            first_line = lines[0] if lines else ""
            # Remove the question number from the start
            category_text = first_line[len(str(q_num)):].strip()
            
            # Split into parts based on whitespace
            parts = [p.strip() for p in category_text.split('  ') if p.strip()]
            
            # Combine all-caps parts for category
            category_parts = []
            subcategory = None
            
            for part in parts:
                # Clean up the part text
                cleaned_part = clean_text(part)
                # Check if it's a category (all caps) or contains known category phrases
                if cleaned_part.isupper() or any(phrase in cleaned_part for phrase in ["CORE KNOWLEDGE"]):
                    category_parts.append(cleaned_part)
                elif not subcategory and cleaned_part[0].isupper():  # First Title Case part is subcategory
                    subcategory = cleaned_part
            
            category = " ".join(category_parts)
            
            # If subcategory wasn't found in first line parts, look in next line
            if not subcategory and len(lines) > 1:
                second_line = clean_text(lines[1])
                if not second_line.isupper() and second_line[0].isupper():
                    subcategory = second_line
            
            # Get the content (everything after category/subcategory)
            content_start = 2 if subcategory in lines[1:2] else 1
            content = " ".join(lines[content_start:])
            
            # Clean up the text
            # Remove page breaks and clean up whitespace
            content = content.replace("[PAGE_BREAK]", " ")
            content = clean_text(content)
            
            # Extract reference if present
            reference = None
            ref_match = re.search(r'(?:Reference|References):\s*(.+?)(?=\[PAGE_BREAK\]|$)', content, re.IGNORECASE)
            if ref_match:
                reference = clean_text(ref_match.group(1))
                content = clean_text(content[:ref_match.start()].strip())
            
            # Store the extracted information
            question_info[q_num] = {
                'category': category,
                'subcategory': subcategory,
                'content': content,
                'reference': reference
            }
            
        except Exception as e:
            question_info[q_num] = f"Error processing question: {str(e)}"
    
    return question_info

# Test the function
if __name__ == "__main__":
    pdf_path = "Original_Rite_Manual_2025 (dragged).pdf"
    question_numbers = [386, 387]
    
    print("Testing question extraction...")
    results = extract_question_info(pdf_path, question_numbers)
    
    for q_num, info in results.items():
        print(f"\nResults for Question {q_num}:")
        print("=" * 80)
        if isinstance(info, dict):
            print(f"Category: {info['category']}")
            print(f"Subcategory: {info['subcategory']}")
            print("\nContent:")
            print("-" * 40)
            print(info['content'])
            if info['reference']:
                print("\nReference:")
                print("-" * 40)
                print(info['reference'])
        else:
            print(info)
        print("=" * 80) 