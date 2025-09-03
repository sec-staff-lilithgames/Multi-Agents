from docx import Document
doc = Document()
doc.add_paragraph('helloworld')
doc.save('output.docx')
