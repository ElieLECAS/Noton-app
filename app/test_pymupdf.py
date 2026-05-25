import pymupdf4llm
import re
import os

pdf_path = "media/documents/255.pdf"
if not os.path.exists(pdf_path):
    pdf_files = [f for f in os.listdir("media/documents") if f.endswith(".pdf")]
    if pdf_files:
        pdf_path = os.path.join("media/documents", pdf_files[0])

print(f"Using PDF: {pdf_path}")
markdown = pymupdf4llm.to_markdown(
    pdf_path,
    page_chunks=False,
    write_images=False,
    show_progress=False
)

print("--- RAW PYMUPDF4LLM OUTPUT LENGTH ---")
print(len(markdown))
print("--- FIRST 1000 CHARACTERS ---")
print(markdown[:1000])

print("--- LAST 1000 CHARACTERS ---")
print(markdown[-1000:])

print("--- DETECTED PATTERNS ---")
lines = markdown.split("\n")
for i, line in enumerate(lines):
    if "page" in line.lower() or "---" in line:
        print(f"Line {i}: {repr(line)}")
