from docx import Document
doc = Document()
doc.add_paragraph('hello-from-broker-test-$(date +%s)')
doc.save('output.docx')
