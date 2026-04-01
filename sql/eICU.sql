-- eICU_result Table

CREATE TABLE eICU_results AS
WITH patient_data AS (
    SELECT 
        uniquepid,
        patientunitstayid,
        patienthealthsystemstayid,
        CASE WHEN gender = 'Male' THEN 1 ELSE 0 END AS gender,
        age,
        admissionheight,
        admissionweight,
        dischargeweight,
        CASE WHEN hospitaldischargestatus = 'Alive' THEN 0 ELSE 1 END AS hospitaldischargestatus
    FROM 
        eicu.patient
),
patient_grouped AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (
            PARTITION BY uniquepid 
            ORDER BY hospitaldischargestatus ASC
        ) AS rn
    FROM 
        patient_data
)

, icu_data AS (
    SELECT 
        icd.patientunitstayid,
        icd.unitvisitnumber,
        icd.icu_los_hours
    FROM 
        public.icustay_detail icd
)

, lab_data AS (
    SELECT 
        lf.patientunitstayid,
        lf.hemoglobin_min,
        lf.hemoglobin_max,
        lf.hematocrit_min,
        lf.hematocrit_max,
        lf.platelet_min,
        lf.platelet_max,
        lf.wbc_min,
        lf.wbc_max,
        lf.glucose_min,
        lf.glucose_max
    FROM 
        public.labsfirstday lf
)

, bg_data AS (
    SELECT 
        pb.patientunitstayid,
        AVG(pb.ph) AS avg_ph
    FROM 
        public.pivoted_bg pb
    GROUP BY 
        pb.patientunitstayid
)

, vita_data AS (
    SELECT 
        pv.patientunitstayid,
        AVG(pv.heartrate) AS avg_heartrate,
        AVG(pv.respiratoryrate) AS avg_respiratoryrate,
        AVG(pv.temperature) AS avg_temperature
    FROM 
        public.pivoted_vital pv
    GROUP BY 
        pv.patientunitstayid
)

, diagnosis_data AS (
    SELECT 
        d.patientunitstayid,
        STRING_AGG(DISTINCT d.icd9code, '; ') AS icd9code_list,
        STRING_AGG(DISTINCT d.diagnosisstring, '; ') AS diagnosis_list
    FROM 
        eicu.diagnosis d
    GROUP BY 
        d.patientunitstayid
)

, medication_data AS (
    SELECT 
        m.patientunitstayid,
        STRING_AGG(DISTINCT m.drugname, '; ') AS drugname_list
    FROM 
        eicu.medication m
    GROUP BY 
        m.patientunitstayid
)

, treatment_data AS (
    SELECT 
        t.patientunitstayid,
        STRING_AGG(DISTINCT t.treatmentstring, '; ') AS treatment_list
    FROM 
        eicu.treatment t
    GROUP BY 
        t.patientunitstayid
)

, score_data AS (
    SELECT 
        ps.patientunitstayid,
        AVG(ps.gcs) AS avg_gcs,
        AVG(ps.gcs_motor) AS avg_gcs_motor,
        AVG(ps.gcs_verbal) AS avg_gcs_verbal,
        AVG(ps.gcs_eyes) AS avg_gcs_eyes,
        AVG(ps.gcs_unable) AS avg_gcs_unable,
        AVG(ps.gcs_intub) AS avg_gcs_intub,
        AVG(ps.fall_risk) AS avg_fall_risk,
        AVG(ps.delirium_score) AS avg_delirium_score,
        AVG(ps.sedation_score) AS avg_sedation_score,
        AVG(ps.sedation_goal) AS avg_sedation_goal,
        AVG(ps.pain_score) AS avg_pain_score,
        AVG(ps.pain_goal) AS avg_pain_goal
    FROM 
        public.pivoted_score ps
    GROUP BY 
        ps.patientunitstayid
)


SELECT 
    pg.uniquepid as patientid,
    pg.patientunitstayid as stayid,
    pg.patienthealthsystemstayid as systemstayid,
    pg.gender as sex,
    pg.age,
    pg.admissionheight,
    pg.admissionweight,
    pg.dischargeweight,
    icu.unitvisitnumber,
    labs.hemoglobin_min,
    labs.hemoglobin_max,
    labs.hematocrit_min,
    labs.hematocrit_max,
    labs.platelet_min,
    labs.platelet_max,
    labs.wbc_min,
    labs.wbc_max,
    labs.glucose_min,
    labs.glucose_max,
    bg.avg_ph as ph,
    vita.avg_heartrate as heartrate,
    vita.avg_respiratoryrate as respiratoryrate,
    vita.avg_temperature as temperature,
    diag.icd9code_list as disease_codes,
    diag.diagnosis_list as diseases,
    med.drugname_list as medications,
    treat.treatment_list as procedure_names,
	pg.hospitaldischargestatus as outcome,
	icu.icu_los_hours,
    scores.avg_gcs as gcs,
    scores.avg_gcs_motor as gcs_motor,
    scores.avg_gcs_verbal as gcs_verbal,
    scores.avg_gcs_eyes as gcs_eyes,
    scores.avg_gcs_unable as gcs_unable,
    scores.avg_gcs_intub as gcs_intub,
    scores.avg_fall_risk as fall_risk,
    scores.avg_delirium_score as delirium_score,
    scores.avg_sedation_score as sedation_score,
    scores.avg_sedation_goal as sedation_goal,
    scores.avg_pain_score as pain_score,
    scores.avg_pain_goal as pain_goal
FROM 
    patient_grouped pg
LEFT JOIN 
    icu_data icu ON pg.patientunitstayid = icu.patientunitstayid
LEFT JOIN 
    lab_data labs ON pg.patientunitstayid = labs.patientunitstayid
LEFT JOIN 
    bg_data bg ON pg.patientunitstayid = bg.patientunitstayid
LEFT JOIN 
    vita_data vita ON pg.patientunitstayid = vita.patientunitstayid
LEFT JOIN 
    diagnosis_data diag ON pg.patientunitstayid = diag.patientunitstayid
LEFT JOIN 
    medication_data med ON pg.patientunitstayid = med.patientunitstayid
LEFT JOIN 
    treatment_data treat ON pg.patientunitstayid = treat.patientunitstayid
LEFT JOIN 
    score_data scores ON pg.patientunitstayid = scores.patientunitstayid
WHERE 
    pg.rn = 1 
ORDER BY 
    pg.uniquepid, 
    pg.hospitaldischargestatus DESC;