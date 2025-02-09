# SYSTEM PROMPT
SYSTEMPROMPT = "You are an experienced ICU doctor, skilled in evaluating patient conditions based on medical history and visits."

# USER PROMPT
USERPROMPT = '''
You will receive a patient's medical visit history, including diseases, medications, and procedures. Your task is to assess whether any high-risk diseases were not adequately controlled, which could potentially lead to patient death.

Your task is to evaluate the classification criteria based on the patient's visits, and reply with Fourteen binary 0/1 characters, separated by commas.

patient's visits:
{visits}

Classification criteria:
{CLASSIFICATION_CRITERIA}

Let's think step by step:
1. Evaluate whether each classification criterion is met based on the patient's visits. If it is met, return 1; otherwise, return 0.
2. Respond with 14 binary 0/1 characters separated by commas, where each character corresponds to a classification criterion. Do not output spaces. Do not output anything else!

RESPONSE:
'''

CLASSIFICATION_CRITERIA = {
    '''
    1.Whether the patient's respiratory disease has not been controlled or worsened.
    2.Whether the patient's digestive disease has not been controlled or worsened.
    3.Whether the patient's cardiovascular disease has not been controlled or worsened.
    4.Whether the patient's urinary disease has not been controlled or worsened.
    5.Whether the patient's reproductive disease has not been controlled or worsened.
    6.Whether the patient's nervous system disease has not been controlled or worsened.
    7.Whether the patient's endocrine or metabolic disease has not been controlled or worsened.
    8.Whether the patient's hematologic or immune disease has not been controlled or worsened.
    9.Whether the patient's tumors or cancers have not been controlled or worsened.
    10.Whether the patient's systemic infections or acute syndromes have not been controlled or worsened.
    11.Whether any other diseases, not listed above, have not been controlled or worsened.
    12.Whether the patient's condition has deteriorated rapidly or reached a critical level based on the last five visits (or all available visits if fewer than five).
    13.Whether the patient has any critically dangerous conditions (e.g., shock, cardiac arrest, acute failure, septicemia, organ necrosis).
    14.Whether the patient's overall condition has improved based on all visits.
    '''
}
