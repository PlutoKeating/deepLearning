"""Extract text from all course PDFs and save to scratch/extracted/"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '.venv', 'lib', 'python3.14', 'site-packages'))
# Fallback approach: run with venv python
from pypdf import PdfReader

COURSE_DIR = "/home/pluto/2026finalExam/deepLearning/course"
OUT_DIR = "/home/pluto/2026finalExam/deepLearning/scratch/extracted"
os.makedirs(OUT_DIR, exist_ok=True)

pdf_files = sorted([
    f for f in os.listdir(COURSE_DIR) 
    if f.endswith('.pdf') and not f.startswith('Course Design')
])

for pdf_file in pdf_files:
    pdf_path = os.path.join(COURSE_DIR, pdf_file)
    out_path = os.path.join(OUT_DIR, pdf_file.replace('.pdf', '.txt'))
    
    print(f"Extracting: {pdf_file} ...")
    try:
        reader = PdfReader(pdf_path)
        lines = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                lines.append(f"=== PAGE {i+1} ===\n{text}\n")
        full_text = "\n".join(lines)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"  -> {out_path} ({len(full_text)} chars, {len(reader.pages)} pages)")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\nDone!")
