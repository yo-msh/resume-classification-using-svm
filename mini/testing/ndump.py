import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from docx import Document
import re

def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return ' '.join(full_text)

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('\bRT\b|\bcc\b', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

# Read the document and clean the entire content
content = read_docx('test.docx')
cleaned_content = cleanResume(content)

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=150)

# Fit the vectorizer on the cleaned content
word_vectorizer.fit([cleaned_content])

# Clean the entire document text
new_cleaned_resume = cleanResume(content)

# Transform the cleaned text using the pre-fitted vectorizer
new_resume_features = word_vectorizer.transform([new_cleaned_resume])

# Load the model and make predictions
loaded_model = joblib.load('svm_model.joblib')

# If needed, access the trained labels from the SVM model
# train_labels = loaded_model.classes_
# le = LabelEncoder()
# le.fit(train_labels)

# Inverse transform the predicted labels
new_prediction = loaded_model.predict(new_resume_features)
# predicted_category = le.inverse_transform(new_prediction)  # Uncomment if needed
print("Predicted Category:", new_prediction[0])
print("hello")
