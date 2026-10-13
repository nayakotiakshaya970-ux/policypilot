# PolicyPilot — Document Q&A with Source Citations

## Problem
HR and policy answers are buried in long PDFs. Employees waste time
searching, and generic chatbots guess, which is risky for topics like
leave or reimbursements.

## Solution
A RAG (retrieval-augmented generation) assistant that answers only from
uploaded documents, cites the file and page for every answer, and says
so when the answer isn't there.

## Features
- Multiple PDF upload
- Page-level citations for every answer
- Refuses to answer instead of guessing
- Follow-up questions with chat memory
- Built-in accuracy test: upload questions with expected answers
- Token usage tracking and a per-session question limit

## How it works
1. Each PDF page is split into overlapping chunks
2. Chunks are stored in ChromaDB with file name and page number
3. A question retrieves the most relevant chunks
4. The LLM answers using only those chunks, with citations

## Results
- Accuracy on my [N]-question test set: [YOUR ACCURACY]%
- Test documents: [WHAT PDF YOU USED]
- Limitation: the check is keyword-based, so correct answers phrased
  differently can be marked wrong

## How to run
pip install -r requirements.txt
Add your API key to a .env file
streamlit run app.py

## Tech stack
Python, ChromaDB, pypdf, Streamlit, [Gemini API / Claude API]

## Live demos
[https://policypilot-akshaya.streamlit.app]

## Future improvements
- LLM-based grading for the accuracy test
- OCR support for scanned PDFs
- Support for Word and Excel files