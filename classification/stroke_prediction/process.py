import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from pickle import load

#load new data, replace path 
df = pd.read_csv("/Users/andreac.henechesierra/Desktop/data.csv")
# load the model, replace path 
model = load(open("/Users/andreac.henechesierra/Desktop/model.pkl", 'rb'))
# load the encoder, replace path 
encoder = load(open("/Users/andreac.henechesierra/Desktop/encoder.pkl", 'rb'))
# load the scaler, replace path 
scaler = load(open('"/Users/andreac.henechesierra/Desktop/scaler.pkl", 'rb'))


#new feature, group classes and drop useless features

def preprocess_df(df):
    """Function to preprocess df in order to prepare the features used
    by the model."""

    #group children and never worked classes in work_type variable
    df.work_type.replace('children','Never_worked',inplace=True)

    #Create new smoking status feature and drop the old one
    df['new_smoking_status'] = df.smoking_status
    df['new_smoking_status'].replace('formerly smoked','have_smoked',inplace=True)
    df['new_smoking_status'].replace('smokes','have_smoked',inplace=True)
    df = df.drop(['id','smoking_status'], axis=1)

    return df

#preprocess df
df = preprocess_df(df)

#categorical encoding
cat_var = df[['gender','ever_married','work_type','Residence_type','new_smoking_status']]

for j in cat_var:
    cat_var[j] = encoder.transform(cat_var[j])

#Numerical scaling  
numerical_features = df[['age','avg_glucose_level','bmi']]

scaled_features = scaler.transform(numerical_features)
scaled_df = pd.DataFrame(scaled_features, columns=numerical_features.columns)

#create postprocessed input df
input_features = pd.concat([df[['hypertension','heart_disease']],cat_var,scaled_df], axis=1)

#prediction
corte = 0.0738489451260411
predictions = model.predict_proba(input_features)
predictions = [1 if val > corte else 0 for val in predictions[:, 1]]

