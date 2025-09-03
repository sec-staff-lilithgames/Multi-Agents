from docx import Document
import sys

def create_word_file(filename, content):
    doc = Document()
    doc.add_paragraph(content)
    doc.save(filename)

if __name__ == "__main__":
    create_word_file("output.docx", "rand-1756888301")