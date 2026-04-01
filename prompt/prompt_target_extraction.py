# SYSTEM PROMPT
SYSTEMPROMPT = "You are an experienced doctor in Intensive Care Unit (ICU) treatment with expertise in identifying high-risk diseases and their primary treatments."

# USER PROMPT
USERPROMPT = '''
I will provide you with unique medical information about a patient, including a list of their diseases, medications, and procedures. 

Your task is to process the medical information and identify the following:
1. High-risk diseases: These are diseases that are life-threatening or have a high risk of causing severe complications if left untreated or poorly controlled. High-risk diseases typically affect critical systems such as cardiovascular, respiratory, renal, or nervous systems. Examples include severe infections, organ failure, acute strokes, etc.
2. Primary medications: These are medications directly used to treat or manage the high-risk diseases, addressing their root cause or controlling life-threatening complications. Exclude medications used solely for symptom relief or minor supportive care.
3. Primary procedures: These are essential procedures used for treating, diagnosing, or managing the high-risk diseases, such as surgeries, mechanical ventilation, or organ replacement therapies.

Patient's Medical Information:
Diseases: {disease_list}.
Medications: {medication_list}.
Procedures: {procedure_list}.

Let’s think step by step:
1. Based on the list of diseases, identify the high-risk diseases according to the following characteristics:
   - Diseases that are life-threatening if left untreated or poorly managed.
   - Diseases with a high risk of severe complications or death even if treated.
   - Diseases impacting critical systems such as cardiovascular, respiratory, renal, or nervous systems.

2. Using the identified high-risk diseases, determine the primary medications and procedures by analyzing their direct role in treating or managing these diseases. Exclude medications or procedures used solely for symptom relief or minor supportive care.

3. Ensure that all diseases, medications, and procedures in the output are unique and do not repeat.

4. Reply in the following format and ensure uniqueness of the output:
Disease: (List of high-risk diseases separated by semicolons).
Medication: (List of primary medications separated by semicolons).
Procedure: (List of primary procedures separated by semicolons).

Example of correct response format:
Disease: Acute kidney failure, unspecified; Chronic diastolic heart failure.
Medication: Amoxicillin-Clavulanic Acid; Aspirin; Atenolol; Atorvastatin.
Procedure: Aorto-coronary bypass of two coronary arteries; Coronary arteriography using two catheters.
   
Only use the provided input data to generate your response in the specified format. Do not add explanations, interpretations, or irrelevant content. If no high-risk diseases, medications, or procedures are identified, reply with 'None'.

RESPONSE:
'''