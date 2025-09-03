from docx import Document
doc = Document()
doc.add_paragraph('Hello, world')
doc.save('output.docx')
