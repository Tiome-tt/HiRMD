-- result_table Table

CREATE TABLE result_table AS
WITH admissions_data AS (
    SELECT 
        subject_id, 
        hadm_id, 
        admittime, 
        dischtime
    FROM admissions
),
patients_data AS (
    SELECT
        subject_id,
        CASE 
            WHEN gender = 'M' THEN 1 
            ELSE 0 
        END AS gender,
        CASE 
            WHEN dod_hosp IS NOT NULL THEN 1 
            ELSE 0 
        END AS dod_hosp,
        dob
    FROM patients
),
age_calculation AS (
    SELECT
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        p.gender,
        p.dod_hosp,  
        p.dob,
        CASE
            WHEN p.dob IS NOT NULL AND a.admittime IS NOT NULL THEN
                EXTRACT(YEAR FROM AGE(a.admittime, p.dob))  
            ELSE
                NULL  
        END AS first_admit_age
    FROM admissions_data a
    JOIN patients_data p ON a.subject_id = p.subject_id
    WHERE p.dob IS NOT NULL
),
age_group AS (
    SELECT 
        subject_id, 
        dob, 
        gender, 
        admittime, 
        dischtime, 
        first_admit_age,
        hadm_id,  
        dod_hosp  
    FROM age_calculation
),

icustays_data AS (
    SELECT 
        icustay_id, 
        subject_id, 
        hadm_id, 
        intime, 
        outtime
    FROM icustays
),
heightweight_data AS (
    SELECT 
        icustay_id, 
        weight_first, 
        height_first
    FROM heightweight
),
blood_gas_avg AS (
    SELECT
        icustay_id,
        AVG(glucose) AS avg_glucose,
        AVG(hematocrit) AS avg_hematocrit,
        AVG(hemoglobin) AS avg_hemoglobin,
        AVG(ph) AS avg_ph,
        AVG(temperature) AS avg_temperature
    FROM blood_gas_first_day
    GROUP BY icustay_id
),
labs_first_day_data AS (
    SELECT
        icustay_id,
        platelet_min,
        platelet_max,
        wbc_min,
        wbc_max
    FROM labs_first_day
    GROUP BY icustay_id, platelet_min, platelet_max, wbc_min, wbc_max
),
vitals_first_day_data AS (
    SELECT
        icustay_id,
        heartrate_mean,
        sysbp_mean,
        diasbp_mean,
        meanbp_mean,
        resprate_mean,
        spo2_mean
    FROM vitals_first_day
    GROUP BY icustay_id, heartrate_mean, sysbp_mean, diasbp_mean, meanbp_mean, resprate_mean, spo2_mean
),
diagnoses_agg AS (
    SELECT
        hadm_id,
        STRING_AGG(icd9_code, ',') AS Disease_Codes
    FROM diagnoses_icd
    GROUP BY hadm_id
),
diagnoses_long_titles AS (
    SELECT
        d.hadm_id,
        STRING_AGG(di.long_title, ';') AS Diseases
    FROM diagnoses_icd d
    JOIN d_icd_diagnoses di ON d.icd9_code = di.icd9_code
    GROUP BY d.hadm_id
),
prescriptions_agg AS (
    SELECT
        icustay_id,
        STRING_AGG(drug, ';') AS Medications
    FROM prescriptions
    GROUP BY icustay_id
),
procedures_agg AS (
    SELECT
        hadm_id,
        STRING_AGG(icd9_code, ',') AS Procedure_Codes
    FROM procedures_icd
    GROUP BY hadm_id
),
procedures_long_titles AS (
    SELECT
        p.hadm_id,
        STRING_AGG(dp.long_title, ';') AS Procedure_Names
    FROM procedures_icd p
    JOIN d_icd_procedures dp ON p.icd9_code = dp.icd9_code
    GROUP BY p.hadm_id
)
SELECT
    ac.subject_id AS PatientID,
    ac.hadm_id AS HadmID,
    i.icustay_id AS ICUstayID,
    ac.admittime AS Admittime,
    ac.dischtime AS Dischtime,
    i.intime AS ICUIntime,
    i.outtime AS ICUOuttime,
    ac.gender AS Sex,
    ac.first_admit_age AS Age, 
    hw.weight_first AS weight,
    hw.height_first AS height,
    ac.dod_hosp AS Outcome,
    bg.avg_glucose AS glucose,
    bg.avg_hematocrit AS hematocrit,
    bg.avg_hemoglobin AS hemoglobin,
    bg.avg_ph AS PH,
    bg.avg_temperature AS Temperature,
    lfd.platelet_min AS PlateletMin,
    lfd.platelet_max AS PlateletMax,
    lfd.wbc_min AS WBCMin,
    lfd.wbc_max AS WBCMax,
    vfd.heartrate_mean AS HeartRateMean,
    vfd.sysbp_mean AS SBPMean,
    vfd.diasbp_mean AS DBPMean,
    vfd.meanbp_mean AS MBPMean,
    vfd.resprate_mean AS RespiratoryRateMean,
    vfd.spo2_mean AS SPO2Mean,
    da.Disease_Codes,
    dl.Diseases,
    pa.Medications,
    pra.Procedure_Codes,
    pl.Procedure_Names
FROM age_group ac
LEFT JOIN icustays_data i ON ac.subject_id = i.subject_id AND ac.hadm_id = i.hadm_id
LEFT JOIN heightweight_data hw ON i.icustay_id = hw.icustay_id
LEFT JOIN blood_gas_avg bg ON i.icustay_id = bg.icustay_id
LEFT JOIN labs_first_day_data lfd ON i.icustay_id = lfd.icustay_id
LEFT JOIN vitals_first_day_data vfd ON i.icustay_id = vfd.icustay_id
LEFT JOIN diagnoses_agg da ON ac.hadm_id = da.hadm_id
LEFT JOIN diagnoses_long_titles dl ON ac.hadm_id = dl.hadm_id
LEFT JOIN prescriptions_agg pa ON i.icustay_id = pa.icustay_id
LEFT JOIN procedures_agg pra ON ac.hadm_id = pra.hadm_id
LEFT JOIN procedures_long_titles pl ON ac.hadm_id = pl.hadm_id
WHERE ac.first_admit_age BETWEEN 18 AND 100  
ORDER BY PatientID, HadmID, ICUstayID;

-- icu_rating Table

CREATE TABLE public.icu_rating AS
SELECT
	r.patientid,
	r.hadmid,
    r.icustayid,
    a.apsiii,
    a.apsiii_prob,
    a.hr_score AS apsiii_hr_score,
    a.meanbp_score AS apsiii_meanbp_score,
    a.temp_score AS apsiii_temp_score,
    a.resprate_score AS apsiii_resprate_score,
    a.pao2_aado2_score AS apsiii_pao2_aado2_score,
    a.hematocrit_score AS apsiii_hematocrit_score,
    a.wbc_score AS apsiii_wbc_score,
    a.creatinine_score AS apsiii_creatinine_score,
    a.uo_score AS apsiii_uo_score,
    a.bun_score AS apsiii_bun_score,
    a.sodium_score AS apsiii_sodium_score,
    a.albumin_score AS apsiii_albumin_score,
    a.bilirubin_score AS apsiii_bilirubin_score,
    a.glucose_score AS apsiii_glucose_score,
    a.acidbase_score AS apsiii_acidbase_score,
    a.gcs_score AS apsiii_gcs_score,
    s.sapsii,
    s.sapsii_prob,
    s.age_score AS sapsii_age_score,
    s.hr_score AS sapsii_hr_score,
    s.sysbp_score AS sapsii_sysbp_score,
    s.temp_score AS sapsii_temp_score,
    s.pao2fio2_score AS sapsii_pao2fio2_score,
    s.uo_score AS sapsii_uo_score,
    s.bun_score AS sapsii_bun_score,
    s.wbc_score AS sapsii_wbc_score,
    s.potassium_score AS sapsii_potassium_score,
    s.sodium_score AS sapsii_sodium_score,
    s.bicarbonate_score AS sapsii_bicarbonate_score,
    s.bilirubin_score AS sapsii_bilirubin_score,
    s.gcs_score AS sapsii_gcs_score,
    s.comorbidity_score AS sapsii_comorbidity_score,
    s.admissiontype_score AS sapsii_admissiontype_score
FROM public.result_table r
LEFT JOIN public.apsiii a ON r.icustayid = a.icustay_id
LEFT JOIN public.sapsii s ON r.icustayid = s.icustay_id;

-- mimiciii_format Table

CREATE TABLE mimiciii_format AS
SELECT
    r.patientid,
    r.hadmid,
    r.icustayid,
    r.admittime,
    r.dischtime,
    r.icuintime,
    r.icuouttime,
    r.sex,
    r.age,
    r.weight,
    r.height,
    r.outcome,
    r.glucose,
    r.hematocrit,
    r.hemoglobin,
    r.ph,
    r.temperature,
    r.plateletmin,
    r.plateletmax,
    r.wbcmin,
    r.wbcmax,
    r.heartratemean,
    r.sbpmean,
    r.dbpmean,
    r.mbpmean,
    r.respiratoryratemean,
    r.spo2mean,
    r.disease_codes,
    r.diseases,
    r.medications,
    r.procedure_codes,
    r.procedure_names,
    
    ir.apsiii,
    ir.apsiii_prob,
    ir.apsiii_hr_score,
    ir.apsiii_meanbp_score,
    ir.apsiii_temp_score,
    ir.apsiii_resprate_score,
    ir.apsiii_pao2_aado2_score,
    ir.apsiii_hematocrit_score,
    ir.apsiii_wbc_score,
    ir.apsiii_creatinine_score,
    ir.apsiii_uo_score,
    ir.apsiii_bun_score,
    ir.apsiii_sodium_score,
    ir.apsiii_albumin_score,
    ir.apsiii_bilirubin_score,
    ir.apsiii_glucose_score,
    ir.apsiii_acidbase_score,
    ir.apsiii_gcs_score,
    ir.sapsii,
    ir.sapsii_prob,
    ir.sapsii_age_score,
    ir.sapsii_hr_score,
    ir.sapsii_sysbp_score,
    ir.sapsii_temp_score,
    ir.sapsii_pao2fio2_score,
    ir.sapsii_uo_score,
    ir.sapsii_bun_score,
    ir.sapsii_wbc_score,
    ir.sapsii_potassium_score,
    ir.sapsii_sodium_score,
    ir.sapsii_bicarbonate_score,
    ir.sapsii_bilirubin_score,
    ir.sapsii_gcs_score,
    ir.sapsii_comorbidity_score,
    ir.sapsii_admissiontype_score
FROM
    public.result_table r
LEFT JOIN
    public.icu_rating ir
ON
    r.patientid = ir.patientid
    AND r.hadmid = ir.hadmid
    AND r.icustayid = ir.icustayid;
