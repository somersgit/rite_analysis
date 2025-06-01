# RITE Analysis Tool

A web application that analyzes RITE exam results and provides insights about questions with high wrong answer rates, along with their explanations from the manual.

## Features

- Upload and analyze RITE wrong answer rates PDF
- Upload and analyze RITE manual PDF
- Identify questions with 60-79% and 80%+ wrong answer rates
- Provide AI-generated summaries of question content and categories
- Modern, responsive web interface

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Java Runtime Environment (JRE) for tabula-py

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd riteAnalysis
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Upload the required PDF files:
   - Wrong Answer Rates PDF
   - RITE Manual PDF

4. Click "Analyze PDFs" to process the files

5. View the results:
   - Summary statistics
   - Questions with 60-79% wrong answers
   - Questions with 80%+ wrong answers

## Notes

- The application uses OpenAI's GPT-3.5 API to generate summaries of questions
- PDF processing may take a few moments depending on file size
- Uploaded files are processed temporarily and deleted after analysis
- Make sure your PDFs are readable and properly formatted

## Troubleshooting

If you encounter any issues:

1. Ensure all prerequisites are installed
2. Check that your OpenAI API key is valid
3. Verify that your PDF files are not corrupted
4. Make sure you have sufficient disk space for temporary file processing

## License

This project is licensed under the MIT License - see the LICENSE file for details. 