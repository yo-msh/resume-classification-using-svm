import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import joblib
import re
import nltk
from nltk.corpus import stopwords
import string
import seaborn as sns
import matplotlib.pyplot as plt


# Load and preprocess data
resumeDataSet = pd.read_csv('UpdatedResumeDataSet.csv', encoding='utf-8')
resumeDataSet['cleaned_resume'] = ''

# Function to clean resume text
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)
    resumeText = re.sub('\bRT\b|\bcc\b', ' ', resumeText)
    resumeText = re.sub('#\S+', '', resumeText)
    resumeText = re.sub('@\S+', '  ', resumeText)
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)
    return resumeText

resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))

# Download stopwords from nltk
nltk.download('stopwords')

# Download punkt from nltk
nltk.download('punkt')

oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
totalWords = []
Sentences = resumeDataSet['Resume'].values

cleanedSentences = ""

for i in range(0, 160):
    cleanedText = cleanResume(Sentences[i])
    cleanedSentences += cleanedText
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation:
            totalWords.append(word)

wordfreqdist = nltk.FreqDist(totalWords)
mostcommon = wordfreqdist.most_common(50)
print(mostcommon)

# Encode the 'Category' column using LabelEncoder
var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resumeDataSet[i] = le.fit_transform(resumeDataSet[i])
    
# Get the original labels before model training
original_labels = le.classes_

# Text Vectorization and Model Training
requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=150)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)

model = OneVsRestClassifier(svm.SVC(kernel='linear', C=1))
model.fit(X_train, y_train)

# Save the model and original labels
joblib.dump({'model': model, 'labels': original_labels}, 'svm_model_with_labels.joblib')
print("Model with original labels dumped!")

# Model Evaluation
prediction1 = model.predict(X_test)
print('Accuracy for train linear SVM is {:.2f}'.format(model.score(X_train, y_train)))
print('Accuracy for test linear SVM is ', metrics.accuracy_score(prediction1, y_test))

conf_matrix = metrics.confusion_matrix(y_test, prediction1)
# print("\nConfusion Matrix:\n", conf_matrix)
# Visualize Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=original_labels, yticklabels=original_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nClassification report for classifier %s:\n%s\n" % (model, metrics.classification_report(y_test, prediction1)))
