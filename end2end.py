# Ignoring Warnings!
import warnings
warnings.filterwarnings('ignore')

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

import sys
sys.stdout = Unbuffered(sys.stdout)

###         ###
###         ###
### IMPORTS ###
###         ###
###         ###

# Import random
import random

# Install and Import yake
# !pip install yake
import yake         

# Install and Import pandas
# !pip install pandas
import pandas as pd

# Install and Import scikit-learn
# !pip install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.cluster import SpectralClustering

from pyodide.http import open_url

# Install and Import pymongo
# !pip install pymongo

#from pymongo.mongo_client import MongoClient
#from pymongo.server_api import ServerApi

"""
# Install and Import spaCy, spaCy Dataset
!pip install spacy
!python -m spacy download en_core_web_lg

import spacy

# Load spaCy's NLP functionality.
nlp = spacy.load("en_core_web_lg")
"""

###                      ###
###                      ###
### FUNCTION DEFINITIONS ###
###                      ###
###                      ###

#-------------------------------------------------------------------------------
# Function: pruned()
#
# Params:   keywords (list)
# Purpose:  Prunes a list of input keywords.
#
def pruned(keywords):
  known_words_of_exodus = ['including', 'include', 'includes', 'understand', 'understanding', 'knowledge', 'skill', 'preferred', 'degree', 'requirements','abilities', 'experience', 'demonstrates', 'demonstrating', 'sales','customer', 'www', 'accommodation', 'recommendation', 'work','days', 'team', 'level', 'manage', 'education', 'genetic', 'san','opportunity', 'genotype', 'ancestry', 'gov', 'duties','qualifications', 'relationships', 'provides', 'related', 'based','hour', 'hours', 'year', 'years', 'issues', 'problems', 'involving','present', 'basic', 'emerging', 'perform', 'performs', 'ability', 'abilities', 'difficult', 'sufficient', 'apply', 'applying', 'identify']

  keywords = [x.lower() for x in keywords]
  keywords = list(dict.fromkeys(keywords))

  for x in known_words_of_exodus:
    for idy, y in enumerate(keywords):
      if x in y.split(' '):
        keywords.pop(idy)

  return keywords
#
# END pruned()

#-------------------------------------------------------------------------------
# Function: sortuple()
#
# Params:   tup (list of tuples)
# Purpose:  Sort tuples on first element.
#
def sortuple(tup):

  # getting length of list of tuples
  lst = len(tup)
  for i in range(0, lst):
    for j in range(0, lst-i-1):
      if (tup[j][1] > tup[j + 1][1]):
        temp = tup[j]
        tup[j]= tup[j + 1]
        tup[j + 1]= temp

  return tup
#
# END sortuple()

#-------------------------------------------------------------------------------
# Function: yakeExtract()
#
# Params:   corpus (string)
# Purpose:  Use YAKE extractor in our own way to collect keywords from input corpus.
#
def yakeExtract(corpus):

  # Extract skills.
  keywords = yake_Extractor.extract_keywords(corpus)

  # Append keywords to a seperate list.
  l = list()
  for kw in keywords:
    l.append(kw[0])

  # Return the keywords.
  return l
#
# END yakeExtract()

#-------------------------------------------------------------------------------
# Function: list2string()
#
# Params:   list and delimiter
# Purpose:  Converts a given list to a single string seperated by a given delimiter.
#
def list2string(l, delim):
  s = ''
  for i in l:
    s = s + i + delim
  
  return s
# 
# END list2string()

#-------------------------------------------------------------------------------
# Function: corpusExtraction()
#
# Params:   corpus (string)
# Purpose:  Here, the results of multiple keyword extraction tools can be combined.
# 
def corpusExtraction(corpus):
  return pruned( yakeExtract(corpus) )
#
# END corpusExtraction

#-------------------------------------------------------------------------------
# Function: KSAStringMapping()
#
# Params:   string
# Purpose:  Takes an input skill as input and returns that skill's 
#           predicted classification.
#
def KSAStringMapping(s):
  l = []
  l.append(s)

  X = vectorizer.transform(l)
  prediction = model.predict(X)

  s = ''
  s += train_enc.inverse_transform(prediction)[0]

  return s
#
# END KSAStringMapping()

#-------------------------------------------------------------------------------
# Function: classifySkills()
#
# Params:   list of skills
# Purpose:  Takes an input set of skills as input and returns each skill's 
#           classification.
#
def classifySkills(skills):
  classification = []
  for skill in skills:
    classification.append(KSAStringMapping(skill))

  return classification
#
# END classifySkills()

#-------------------------------------------------------------------------------
# Function: createOutcomes()
#
# Params:   list of classifications and skills, for each input
# Purpose:  IN-PROGRESS -- COMBINES ALL CLASSIFICATIONS AND SKILLS INTO A SINGLE
#           ENCOMPASSING DATA STRUCT
#
def createOutcomes(classification, skills):  
  outcomes = {classification[i]: skills[i] for i in range(len(classification))}
  outcomes = list(set(classification))

  for i in range(0,len(outcomes)):
    for j in range(0,len(classification)):
      if outcomes[i] == classification[j]:
        outcomes[i] += ' ' + skills[j]

  return outcomes
#
# END createOutcomes()

#-------------------------------------------------------------------------------
# Function: classification_report_csv()
#
# Params:   scikit-learn report generated from training a log. reg. model
# Purpose:  Generates a clean report of the model's ability to predict future 
#           skills.
#
def classification_report_csv(report):
  report_data = []
  lines = report.split('\n')

  for line in lines[2:-5]:
    row = {}
    row_data = line.split('      ')

    while("" in row_data):
      row_data.remove("")

    #print(row_data)

    row['class'] = row_data[0]
    row['precision'] = float(row_data[1])
    row['recall'] = float(row_data[2])
    row['f1_score'] = float(row_data[3])
    row['support'] = float(row_data[4])
    report_data.append(row)
  dataframe = pd.DataFrame.from_dict(report_data)
  dataframe.to_csv('classification_report.csv', index = False)
#
# END classification_report_csv()

#-------------------------------------------------------------------------------
# Function: loadData()
#
# Params:   user-defined corpus type ('postings', 'assessments') and specific filename
# Purpose:  Loads the data from the specific file to a list of JSON objects.
#
def loadData(corpusType, filename):
  labeledCorpus = pd.read_csv(filename)

  try:
    labeledCorpus = labeledCorpus.drop(columns=['description'])
  except: pass

  listofJSONs = []

  for index, row in labeledCorpus.iterrows():
    if corpusType == 'assessments':
      obj = {
          "university" : row["university"],
          "class_title" : row["class_title"],
          "skills" : row["skills"],
          "class_code" : row["class_code"],
          "assessment_title" : row["assessment_title"],
          "assessment_type" : row["assessment_type"]
      }
    else:
      obj = {
          "entry_level" : row["entry_level"],
          "federal_industrial" : row["federal_industrial"],
          "skills" : row["skills"],
          "title" : row["title"]
      }

    listofJSONs.append(obj)

  return listofJSONs
#
# END loadData()

#-------------------------------------------------------------------------------
# Function: postJSONtoDBcollection()
#
# Params:   Name of both the JSON you want to push and the collection name.
# Purpose:  Pushes the JSON object to the database, effectively saving it.
#
def postJSONtoDBcollection(JSONname, collectionName):
  x = collectionName.insert_many( JSONname )
  print(x.inserted_ids)
#
# END postJSONtoDBcollection()

#-------------------------------------------------------------------------------
# Function: computeAlignmentFromNLP()
#
# Params:   Two sets of learning outcomes, generated and NLP'd.
# Purpose:  Computes the percent alignment between two SETS OF OUTCOMES.
#
def computeAlignmentFromNLP(set1, set2):
  avg = 0
  count = 0
  for i in set1:
    for j in set2:
      avg += i.similarity(j)
      count += 1

  avg /= count
  return avg
#
# END computeAlignmentFromNLP()

#-------------------------------------------------------------------------------
# Function: unique()
#
# Params:   list
# Purpose:  Returns a list of unique objects within the input list.
#
def unique(list1):
  # initialize a null list
  unique_list = []

  # traverse for all elements
  for x in list1:
      # check if exists in unique_list or not
      if x not in unique_list:
          unique_list.append(x)

  return unique_list
#
# END unique()

#-------------------------------------------------------------------------------
# Function: NLPthoseOutcomes()
#
# Params:   a set of outcomes
# Purpose:  Returns a list of unique objects within the input list.
#
def NLPthoseOutcomes(outcomeSet):
  ALL_OUTCOMES = []
  TOPIC_COUNT_LIST = []
  SUBTOPIC_COUNT_LIST = []

  for i in outcomeSet:
    for j in i[1]:
      ALL_OUTCOMES.append(j)

  nodupes = []

  for sublist in ALL_OUTCOMES:
    if sublist not in nodupes:
      nodupes.append(sublist)

  allDocs = []

  for i in sorted(nodupes):
    TOPIC_COUNT_LIST.append(i[0])
    SUBTOPIC_COUNT_LIST.append(i[1])
    allDocs.append( nlp( list2string(i, ',') ) )

  #print('TOPIC SUPPORT: ', len(unique(TOPIC_COUNT_LIST)))
  #print('SUBTOPIC SUPPORT: ', len(unique(SUBTOPIC_COUNT_LIST)))

  return allDocs
#
# END NLPthoseOutcomes()








###             ###
###             ###
### DRIVER CODE ###
###             ###
###             ###

# Initialize YAKE extractor method with parameters.
yake_Extractor = yake.KeywordExtractor(lan='en', n=3, dedupLim=0.95, dedupFunc='seqm', top=25, features=None)

# ------------------------------------------------------------------------------
# Collecting KSAT Data
print('Collecting KSAT Data...')

url = "https://raw.githubusercontent.com/tylerjparks/tylerjparks.github.io/main/KSAT%20Mappings%20for%20NLP%20Model%20-%20Knowledge%20Unit%20Mapping.csv"
knowledgeDF = pd.read_csv(open_url(url))

# drop empty
knowledgeDF.dropna(subset=['TIER1 JOB'], inplace=True)

# drop all 'Knowledge of '
ko = 'Knowledge of '
knowledgeDF['SKILL'] = knowledgeDF['SKILL'].map(lambda x: x.replace(ko, ''))

url = "https://raw.githubusercontent.com/tylerjparks/tylerjparks.github.io/main/KSAT%20Mappings%20for%20NLP%20Model%20-%20Skill%20Unit%20Mapping.csv"
skillDF = pd.read_csv(open_url(url))

# drop empty
skillDF.dropna(subset=['TIER1 JOB'], inplace=True)

# drop all 'Skill in '
si = 'Skill in '
skillDF['SKILL'] = skillDF['SKILL'].map(lambda x: x.replace(si, ''))

url = "https://raw.githubusercontent.com/tylerjparks/tylerjparks.github.io/main/KSAT%20Mappings%20for%20NLP%20Model%20-%20Ability%20Unit%20Mapping.csv"
abilityDF = pd.read_csv(open_url(url))

# drop empty
abilityDF.dropna(subset=['TIER1 JOB'], inplace=True)

# drop all 'Ability to '
at = 'Ability to '
abilityDF['SKILL'] = abilityDF['SKILL'].map(lambda x: x.replace(at, ''))

url = "https://raw.githubusercontent.com/tylerjparks/tylerjparks.github.io/main/KSAT%20Mappings%20for%20NLP%20Model%20-%20Task%20Unit%20Mapping.csv"
taskDF = pd.read_csv(open_url(url))

# drop empty
# nothing in 'SKILL' to drop
taskDF.dropna(subset=['TIER1 JOB'], inplace=True)

# append all DFs to single mapping
frames = [knowledgeDF, skillDF, abilityDF, taskDF]

mappingDF = pd.concat(frames)
# ------------------------------------------------------------------------------
# END collecting KSAT Data

print('Training Logistic Regression Model...')
TRAINTEST_SPLIT = 0.25

max_acc = 0
max_size = 0
max_preds = []
max_train_enc = []
max_TestingY = []
for i in range(0, 3):

  # shuffled approach
  shuffled = mappingDF.sample(frac=1)
  df_train, df_test = train_test_split(shuffled, test_size=(TRAINTEST_SPLIT + (i/150)), random_state=random.randint(0,1000000))

  # init. vectorizer
  vectorizer = CountVectorizer()

  # fit training data
  TrainingX = vectorizer.fit_transform(df_train['SKILL'])
  TrainingX

  # transform the testing data using the previous model
  TestingX = vectorizer.transform(df_test['SKILL'])
  TestingX

  train_enc = preprocessing.LabelEncoder()
  test_enc  = preprocessing.LabelEncoder()

  # assume the following is the list of unique classes in your data
  #############################################
  train_data_targets = df_train['TIER1 JOB']
  test_data_targets = df_test['TIER1 JOB']
  #############################################
  # fit your targets of the training data to the LabelEncoder instance
  train_enc.fit(train_data_targets)
  test_enc.fit(test_data_targets)

  # encode the targets as numerical labels
  encoded_train = train_enc.transform(train_data_targets)
  encoded_test = test_enc.transform(test_data_targets)

  # load the testing categories
  TrainingY = encoded_train
  TrainingY

  TestingY = encoded_test
  TestingY

  scikit_log_reg = LogisticRegression(
                                      verbose=0,
                                      solver='lbfgs', # sag # newton-cg # lbfgs
                                      random_state=random.randint(0,1000000),
                                      C=1,
                                      penalty='l2',
                                      max_iter=500,
                                      class_weight='balanced',
                                      #multi_class = 'ovr'
                                      multi_class = 'multinomial'
                                    )


  model = scikit_log_reg.fit(TrainingX, TrainingY)

  # get predictions from testing set
  preds = model.predict(TestingX)

  # generate accuracy
  KSAT_MODEL_ACCURACY = metrics.accuracy_score(TestingY, preds)

  if KSAT_MODEL_ACCURACY > max_acc:
    max_acc = KSAT_MODEL_ACCURACY
    max_preds = preds
    max_train_enc = train_enc
    max_TestingY = TestingY
    max_size = TRAINTEST_SPLIT + (i/150)

  #print('ATTEMPT ', i, '-Test Size-', (TRAINTEST_SPLIT + (i/150)), '-', KSAT_MODEL_ACCURACY)
  print('\tTraining in-progress: ', KSAT_MODEL_ACCURACY)

preds = max_preds
KSAT_MODEL_ACCURACY = max_acc
train_enc = max_train_enc
TestingY = max_TestingY


labels = list(train_enc.transform(train_enc.classes_))
reportlabels = list(train_enc.classes_)
report = classification_report(TestingY, preds, target_names = reportlabels)

EXTRA_labels = labels

"""
print()
print()
print('Accuracy :', KSAT_MODEL_ACCURACY)
print('Test Size:', max_size)
print('Labels:', labels)

print()
print()
print(report)
print()
print()
"""

classification_report_csv(report)

cm = metrics.confusion_matrix(TestingY, preds, labels=labels)

#print('Confusion Matrix')
#print(cm)

# Spectral Clustering
NUM_CLUSTERS = 1

for j in range(NUM_CLUSTERS, NUM_CLUSTERS+1):
  sc = SpectralClustering(n_clusters = j, affinity ='nearest_neighbors', verbose=0).fit_predict(cm)

  #print(EXTRA_labels)
  sorted_labels_tuple = sortuple(list(zip(EXTRA_labels, sc)))
  temp = sorted_labels_tuple
  #print(temp)
  sorted_labels = list(zip(*temp))[0]
  sorted_clusters = list(zip(*temp))[1]

  #for i in sorted_labels_tuple:
  #  print(i)

  cm = metrics.confusion_matrix(TestingY, preds, labels=sorted_labels)

  # calculate all labels percentage correctness
  percentages = []
  i = 0
  for rows in cm:
    rowsL = rows.tolist()

    correct = rowsL.pop(i)
    incorrect = sum(rowsL)

    percentage = round(correct/(incorrect+correct)*100, 2)
    percentages.append(percentage)

    i = i + 1

  # add percentages to labels
  temp = list(train_enc.classes_)

  classesPercent = []
  for classes in temp:
    classesPercent.append(classes + ' - ' + str(labels.pop(0)) + ' - ' + str(percentages.pop(0)) + '%')




print('Collecting User Input...')

#url = "https://raw.githubusercontent.com/tylerjparks/tylerjparks.github.io/main/Assignments%20for%20NLP%20Tool%20-%20assignments.csv"
url = "https://raw.githubusercontent.com/tylerjparks/tylerjparks.github.io/main/Labeled%20-%20federal200.csv"
df_userinput = pd.read_csv(open_url(url))

SKILLS_LIST = []
CLASSIFIED_LIST = []
OUTCOMES_LIST = []
#FEDERAL_INDUSTRIAL = []
#ENTRY_LEVEL = []

df_userinput = df_userinput.sample(frac=1)
df_userinput = df_userinput.head(1)

print('Processing Objects...')
for index, row in df_userinput.iterrows():
  print('The full-text description input:')
  print('************************************************************')
  print(row['description'])
  print('************************************************************')
  print()

  # SKILLS EXTRACTION
  skills = corpusExtraction(row['description'])
  print('The SKILLS collected from Job Posting:')
  print(skills)
  print()
  SKILLS_LIST.append(skills)

  # SKILLS CLASSIFICATION
  classified = classifySkills(skills)
  print('The CLASSIFICATIONS used for Job Posting:')
  print(classified)
  print()
  CLASSIFIED_LIST.append(classified)

  # OUTCOME CREATION
  outcomes = createOutcomes(classified, skills)
  print('The OUTCOMES generated from Job Posting:')
  print(outcomes)
  print()
  OUTCOMES_LIST.append(outcomes)

  #FEDERAL_INDUSTRIAL.append('federal')
  #ENTRY_LEVEL.append(1)

try: df_userinput.insert(0, 'skills', SKILLS_LIST)
except: pass
df_userinput.insert(0, 'classified', CLASSIFIED_LIST)
df_userinput.insert(0, 'outcomes', OUTCOMES_LIST)

#df_userinput.insert(0, 'federal_industrial', FEDERAL_INDUSTRIAL)
#df_userinput.insert(0, 'entry_level', ENTRY_LEVEL)

df_userinput.to_csv('userinput-extracted.csv', index=False)
print('Complete!')
"""
# Get user password
#password = getpass()
#password MIGHT equal simcity4
password = 'simcity4'

# Create MongoDB connection URL
uri = "mongodb+srv://test-user:"+password+"@m0cluster.4phwiir.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Clear password var
password = ''

assessmentsDB = client.assessments
postingsDB    = client.postings
assessmentCollection1 = assessmentsDB.assessment1
postingCollection1 = postingsDB.posting1

assessmentData = loadData('assessments','Labeled - assessments.csv')
#federalData = loadData('postings','Labeled - federal200.csv')
#industryData = loadData('postings','Labeled - indeed4500.csv')

print('Posting Assessments to MongoDB Database...')
postJSONtoDBcollection(assessmentData, assessmentCollection1)

print('Objects Posted to Database!')
"""
"""
NLP_ASSESSMENTS = NLPthoseOutcomes(AssignmentOutcomes)
NLP_INDUSTRY    = NLPthoseOutcomes(IndustryOutcomes)
NLP_CAE         = NLPthoseOutcomes(CAEOutcomes)
NLP_FEDERAL     = NLPthoseOutcomes(FederalOutcomes)

print('ASSESMENT vs CAE', computeAlignmentFromNLP(NLP_ASSESSMENTS, NLP_CAE))
print('ASSESMENT vs INDUSTRY', computeAlignmentFromNLP(NLP_ASSESSMENTS, NLP_INDUSTRY))
print('ASSESMENT vs FEDERAL', computeAlignmentFromNLP(NLP_ASSESSMENTS, NLP_FEDERAL))
"""