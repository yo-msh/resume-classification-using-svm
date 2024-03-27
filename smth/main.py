from flask import Flask, render_template, request, redirect
import joblib
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re

app = Flask(__name__)

# Load the trained SVM model and label encoder
loaded_data = joblib.load('svm_model_with_labels.joblib')
loaded_model = loaded_data['model']
original_labels = loaded_data['labels']

# Fit LabelEncoder with original labels
le = LabelEncoder()
le.fit(original_labels)

# Initialize TfidfVectorizer for word representation
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=150)

# Placeholder for fitted content
fitted_content = None

# Function to clean the resume text
def clean_resume(resumeText):
    # ... (Same cleanResume function as before)
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('\bRT\b|\bcc\b', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

# Function to read content from a Word document
def read_docx(file_path):
    # ... (Same read_docx function as before)
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return ' '.join(full_text)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    global fitted_content  # Declare as global to store fitted content

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            content = read_docx(file)

            # Fit vectorizer if not fitted
            if fitted_content is None:
                word_vectorizer.fit([content])
                fitted_content = content

            new_cleaned_resume = clean_resume(content)
            new_resume_features = word_vectorizer.transform([new_cleaned_resume])
            new_prediction = loaded_model.predict(new_resume_features)
            predicted_category = le.inverse_transform(new_prediction)
            return render_template('result.html', category=predicted_category[0])

if __name__ == '__main__':
    app.run(debug=True)
