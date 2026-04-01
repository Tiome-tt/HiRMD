## config

API_KEY = "your API Key"  # Replace with your API Key
BASE_URL = "your GPT service API address"  # Replace with your GPT service API address
MODEL = "gpt-3.5-turbo-1106"

datasets = "mimic-iii" # mimic-iii/mimic-iv/eicu
model = "GRU" # GRU/Transformer/LSTM

# Hyperparameters
if datasets == "mimic-iv" and model == "GRU":
    Hidden_Size = 64
    Num_Layers = 2
    Learning_Rate = 0.0003
    Num_Epochs = 20
    Batch_Size = 64
    Dropout = 0.3
elif datasets == "mimic-iv" and model == "Transformer":
    Hidden_Size = 64
    Num_Layers = 3
    Learning_Rate = 0.0005
    Num_Epochs = 30
    Batch_Size = 64
    Dropout = 0.3
elif datasets == "mimic-iv" and model == "LSTM":
    Hidden_Size = 64
    Num_Layers = 3
    Learning_Rate = 0.0005
    Num_Epochs = 25
    Batch_Size = 64
    Dropout = 0.1
elif datasets == "mimic-iii" and model == "GRU":
    Hidden_Size = 64
    Num_Layers = 2
    Learning_Rate = 0.0003
    Num_Epochs = 20
    Batch_Size = 64
    Dropout = 0.3
    # # mini
#     Hidden_Size = 32
#     Num_Layers = 1
#     Learning_Rate = 0.0005
#     Num_Epochs = 30
#     Batch_Size = 32
#     Dropout = 0.2
elif datasets == "mimic-iii" and model == "Transformer":
    Hidden_Size = 64
    Num_Layers = 2
    Learning_Rate = 0.00005
    Num_Epochs = 30
    Batch_Size = 64
    Dropout = 0.3
elif datasets == "mimic-iii" and model == "LSTM":
    Hidden_Size = 64
    Num_Layers = 2
    Learning_Rate = 0.0005
    Num_Epochs = 20
    Batch_Size = 64
    Dropout = 0.3
elif datasets == "eicu" and model == "GRU":
    Hidden_Size = 64
    Num_Layers = 2
    Learning_Rate = 0.001
    Num_Epochs = 20
    Batch_Size = 64
    Dropout = 0.3
#     # mini
#     Hidden_Size = 32
#     Num_Layers = 2
#     Learning_Rate = 0.0002
#     Num_Epochs = 30
#     Batch_Size = 32
#     Dropout = 0.3
elif datasets == "eicu" and model == "Transformer":
    Hidden_Size = 64
    Num_Layers = 3
    Learning_Rate = 0.0008
    Num_Epochs = 30
    Batch_Size = 64
    Dropout = 0.1
elif datasets == "eicu" and model == "LSTM":
    Hidden_Size = 64
    Num_Layers = 2
    Learning_Rate = 0.0006
    Num_Epochs = 30
    Batch_Size = 64
    Dropout = 0.1


if datasets == "mimic-iii":
    Input_Features = [
        'Sex', 'Age', 'Height', 'Weight', 'PH', 'Hemoglobin', 'Temperature', 'Glucose',
        'Hematocrit', 'PlateletMin', 'PlateletMax', 'WBCMin', 'WBCMax', 'HeartRateMean', 'SBPMean', 'DBPMean', 'MBPMean',
        'RespiratoryRateMean', 'SpO2Mean', 'Height_missing', 'Weight_missing', 'PH_missing', 'Hemoglobin_missing',
        'Temperature_missing', 'Glucose_missing', 'Hematocrit_missing', 'PlateletMin_missing', 'PlateletMax_missing',
        'WBCMin_missing', 'WBCMax_missing', 'HeartRateMean_missing', 'SBPMean_missing', 'DBPMean_missing',
        'MBPMean_missing', 'RespiratoryRateMean_missing', 'SpO2Mean_missing'
    ]
elif datasets == "mimic-iv":
    Input_Features = [
        'Sex', 'Age', 'Height', 'Weight', 'PH', 'Hemoglobin', 'Temperature', 'Glucose',
        'White Blood Cell Count', 'Lymphocytes', 'Hematocrit', 'Platelet', 'Red Blood Cell Count',
        'Heart Rate', 'Respiratory Rate', 'Systolic Blood Pressure', 'Diastolic Blood Pressure',
        'Mean Blood Pressure', 'Height_missing', 'Weight_missing', 'PH_missing', 'Hemoglobin_missing',
        'Temperature_missing', 'Glucose_missing', 'White Blood Cell Count_missing',
        'Lymphocytes_missing', 'Hematocrit_missing', 'Platelet_missing', 'Red Blood Cell Count_missing',
        'Systolic Blood Pressure_missing', 'Diastolic Blood Pressure_missing',
        'Mean Blood Pressure_missing', 'Respiratory Rate_missing', 'Heart Rate_missing'
    ]
elif datasets == "eicu":
    Input_Features = [
        'Sex', 'Age', 'AdmissionHeight', 'AdmissionWeight', 'DischargeWeight', 'PH', 'HemoglobinMin',
        'HemoglobinMax','Temperature', 'GlucoseMin', 'GlucoseMax', 'HematocritMin', 'HematocritMax',
        'PlateletMin', 'PlateletMax', 'WBCMin', 'HeartRate','RespiratoryRate', 'AdmissionHeight_missing',
        'AdmissionWeight_missing', 'DischargeWeight_missing', 'UnitVisitNumber_missing','GlucoseMin_missing',
        'GlucoseMax_missing', 'HematocritMin_missing','HematocritMax_missing', 'HemoglobinMin_missing',
        'HemoglobinMax_missing', 'PH_missing','Temperature_missing', 'PlateletMin_missing','PlateletMax_missing',
        'WBCMin_missing', 'HeartRate_missing', 'RespiratoryRate_missing'
    ]