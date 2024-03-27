import joblib
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('\bRT\b|\bcc\b', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return ' '.join(full_text)

# replace 'path_to_file.docx' with the path to the .docx file you want to read
content = read_docx('new.docx')

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=150)
word_vectorizer.fit([content])  # Pass content as a list

print(content)

new_cleaned_resume = cleanResume(content)
new_resume_features = word_vectorizer.transform([new_cleaned_resume])

# Assuming you have the original labels used during training
# original_labels = ["Data Science", "HR", "Advocate","Arts","Web Designing","Mechanical Engineer","Sales","Health and fitness","Civil Engineer","Java Developer","Business Analyst","SAP Developer","Automation Testing","Electrical Engineering","Operations Manager","Python Developer","DevOps Engineer","Network Security Engineer","PMO","Database","Hadoop","Hadoop","DotNet Developer","Blockchain","Testing"]

# Fit LabelEncoder with original labels
loaded_data = joblib.load('svm_model_with_labels.joblib')
loaded_model = loaded_data['model']
original_labels = loaded_data['labels']

le = LabelEncoder()
le.fit(original_labels)

# loaded_model = joblib.load('svm_model.joblib')
new_prediction = loaded_model.predict(new_resume_features)
predicted_category = le.inverse_transform(new_prediction)
print("Predicted Category:", predicted_category[0])

print('jello new')
