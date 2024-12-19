# Код разработали Кулаков Алексей, Георгий Козлов 
from pgmpy.models import BayesianNetwork
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

model = BayesianNetwork([
    ('Diagnosis', 'Medication'),
    ('Diagnosis', 'Symptom Severity'),
    ('Medication', 'Outcome'),
    ('Medication', 'Treatment Progress'),
    ('Adherence to Treatment', 'Treatment Progress'),
    ('Treatment Progress', 'Outcome'),
    ('Stress Level', 'Mood Score'),
    ('Sleep Quality', 'Mood Score'),
    ('Physical Activity', 'Mood Score'),
    ('Mood Score', 'Symptom Severity'),
    ('Symptom Severity', 'Outcome'),
    ('Age', 'Outcome'),
    ('Gender', 'Outcome')
])

df = pd.read_csv("mental_health_diagnosis_treatment_.csv")
df.drop(['Patient ID', 'Treatment Start Date', 'Treatment Duration (weeks)', 'AI-Detected Emotional State', 'Therapy Type'], axis=1, inplace=True)


print(dict(enumerate(df['Diagnosis'].astype('category').cat.categories)))
print(dict(enumerate(df['Gender'].astype('category').cat.categories)))
print(dict(enumerate(df['Outcome'].astype('category').cat.categories)))
print(dict(enumerate(df['Medication'].astype('category').cat.categories)))

df['Outcome'] = df['Outcome'].astype('category').cat.codes
df['Gender'] = df['Gender'].astype('category').cat.codes
df['Diagnosis'] = df['Diagnosis'].astype('category').cat.codes
df['Medication'] = df['Medication'].astype('category').cat.codes


numerical_columns = [ 'Symptom Severity', 'Mood Score', 'Sleep Quality', 'Physical Activity', 'Stress Level', 'Treatment Progress']
df['Age'] /= 100
df['Adherence to Treatment'] /= 100
for i in numerical_columns:
    df[i] /= 10
for i in numerical_columns:
    df[i] = df[i].apply(lambda x: 0 if x < 0.33 else (1 if x < 0.67 else 2))
df['Adherence to Treatment'] = df['Adherence to Treatment'].apply(lambda x: 0 if x < 0.33 else (1 if x < 0.67 else 2))
df['Age'] = df['Age'].apply(lambda x: 0 if x < 0.18 else (1 if x < 0.35 else (2 if x < 60 else 3)))

model.fit(df, estimator=MaximumLikelihoodEstimator)
inference = VariableElimination(model)
result = inference.query(variables=['Outcome'], evidence={'Gender': 1, 'Sleep Quality': 2, 'Stress Level': 1, 'Age' : 2, 'Mood Score': 1, "Diagnosis": 0})
print(result)
result = inference.query(variables=['Medication'], evidence={'Gender': 0, 'Sleep Quality': 1, 'Stress Level': 1, 'Age' : 1, 'Outcome': 1})
print(result)