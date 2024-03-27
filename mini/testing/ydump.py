import joblib
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def clean_resume(resume_text):
    resume_text = re.sub('http\S+\s*', ' ', resume_text)  # remove URLs
    resume_text = re.sub('\bRT\b|\bcc\b', ' ', resume_text)  # remove RT and cc
    resume_text = re.sub('#\S+', '', resume_text)  # remove hashtags
    resume_text = re.sub('@\S+', '  ', resume_text)  # remove mentions
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)  # remove punctuations
    resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text) 
    resume_text = re.sub('\s+', ' ', resume_text)  # remove extra whitespace
    return resume_text

def read_docx(file_path):
    doc = Document(file_path)
    full_text = [para.text for para in doc.paragraphs]
    return ' '.join(full_text)

# replace 'path_to_file.docx' with the path to the .docx file you want to read
content = read_docx('new.docx')

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',)
word_vectorizer.fit([content])  # Pass content as a list

print(content)

new_cleaned_resume = clean_resume(content)
new_resume_features = word_vectorizer.transform([new_cleaned_resume])

loaded_data = joblib.load('svm_model_with_labels.joblib')
loaded_model = loaded_data['model']
original_labels = loaded_data['labels']

new_prediction = loaded_model.predict(new_resume_features)
predicted_category_index = new_prediction[0]
predicted_category = original_labels[predicted_category_index]
print("Predicted Category:", predicted_category)

print('Hello new')
