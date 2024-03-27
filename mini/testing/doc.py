from docx import Document

def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return ' '.join(full_text)

# replace 'path_to_file.docx' with the path to the .docx file you want to read
content = read_docx('test.docx')

print(content)