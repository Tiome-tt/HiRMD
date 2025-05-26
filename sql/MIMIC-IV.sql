-- normal_feature Table

SET work_mem = '512MB';
DROP TABLE IF EXISTS basic_data;
DROP TABLE IF EXISTS diagnoses_data;
DROP TABLE IF EXISTS prescriptions_data;
DROP TABLE IF EXISTS procedures_data;


CREATE TEMP TABLE basic_data AS
SELECT
    pa.subject_id,
    ad.hadm_id,
    ad.admittime,
    ad.dischtime,
	 ROW_NUMBER () OVER(PARTITION BY ad.subject_id
    ORDER BY ad.admittime) AS admittime_rank,
    ROUND(EXTRACT(EPOCH FROM (ad.dischtime - ad.admittime)) / 86400, 2) AS hosp_los,
    pa.gender,
    ROUND(pa.anchor_age + EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year, 2) AS age,
    CASE
        WHEN ad.deathtime IS NOT NULL THEN 1
        ELSE 0
    END AS death_status
FROM
    mimiciv_hosp.admissions ad
INNER JOIN mimiciv_hosp.patients pa ON ad.subject_id = pa.subject_id;

CREATE TEMP TABLE diagnoses_data AS
SELECT
    ad.hadm_id,
    STRING_AGG(DISTINCT dc.icd_code, '; ') AS disease_codes,
    STRING_AGG(DISTINCT icd.long_title, '; ') AS surface_name
FROM
    mimiciv_hosp.admissions ad
LEFT JOIN mimiciv_hosp.diagnoses_icd dc ON ad.hadm_id = dc.hadm_id
LEFT JOIN mimiciv_hosp.d_icd_diagnoses icd ON dc.icd_code = icd.icd_code
GROUP BY ad.hadm_id;

CREATE TEMP TABLE prescriptions_data AS
SELECT
    ad.hadm_id,
    STRING_AGG(DISTINCT ps.pharmacy_id::text, '; ') AS pharmacy_id,
    STRING_AGG(DISTINCT ps.drug, '; ') AS drug_name
FROM
    mimiciv_hosp.admissions ad
LEFT JOIN mimiciv_hosp.prescriptions ps ON ad.hadm_id = ps.hadm_id
GROUP BY ad.hadm_id;

CREATE TEMP TABLE procedures_data AS
SELECT
    ad.hadm_id,
    STRING_AGG(DISTINCT pc.icd_code, '; ') AS procedure_codes,
    STRING_AGG(DISTINCT picd.long_title, '; ') AS procedures_description
FROM
    mimiciv_hosp.admissions ad
LEFT JOIN mimiciv_hosp.procedures_icd pc ON ad.hadm_id = pc.hadm_id
LEFT JOIN mimiciv_hosp.d_icd_procedures picd ON pc.icd_code = picd.icd_code
GROUP BY ad.hadm_id;

CREATE TABLE normal_features AS
SELECT
    b.subject_id,
    b.hadm_id,
    b.admittime,
    b.dischtime,
	b.admittime_rank,
    b.hosp_los,
    b.gender,
    b.age,
    b.death_status,
    d.disease_codes,
    d.surface_name,
    p.pharmacy_id,
    p.drug_name,
    pr.procedure_codes,
    pr.procedures_description
FROM
    basic_data b
LEFT JOIN diagnoses_data d ON b.hadm_id = d.hadm_id
LEFT JOIN prescriptions_data p ON b.hadm_id = p.hadm_id
LEFT JOIN procedures_data pr ON b.hadm_id = pr.hadm_id
ORDER BY b.subject_id ASC, b.hadm_id ASC;

-- lab_features Table

CREATE TABLE lab_features AS
SELECT 
    idf.subject_id,
    idf.hadm_id,
    ROUND(CAST(AVG(bg.ph) AS numeric), 2) AS ph,
    ROUND(CAST(AVG(bg.hemoglobin) AS numeric), 2) AS hemoglobin,
    ROUND(CAST(AVG(bg.temperature) AS numeric), 2) AS temperature,
    ROUND(CAST(AVG(bg.glucose) AS numeric), 2) AS glucose,
	ROUND(CAST(AVG(bd.wbc) AS numeric), 2) AS wbc,
	ROUND(CAST(AVG(bd.lymphocytes) AS numeric), 2) AS lymphocytes,
	ROUND(CAST(AVG(cbc.hematocrit) AS numeric), 2) AS hematocrit,
	ROUND(CAST(AVG(cbc.platelet) AS numeric), 2) AS platelet,
	ROUND(CAST(AVG(cbc.rbc) AS numeric), 2) AS rbc

FROM 
    public.id_features idf
LEFT JOIN mimiciv_derived.bg bg ON idf.hadm_id = bg.hadm_id
LEFT JOIN mimiciv_derived.blood_differential bd ON idf.hadm_id = bd.hadm_id
LEFT JOIN mimiciv_derived.complete_blood_count cbc ON idf.hadm_id = cbc.hadm_id

GROUP BY 
    idf.subject_id, idf.hadm_id
ORDER BY 
    idf.subject_id, idf.hadm_id;

-- icu_feature Table

SET work_mem = '128MB';
CREATE TABLE icu_features
SELECT 
    idf.subject_id AS subject_id,
    idf.hadm_id AS hadm_id,
    idf.stay_id AS stay_id,
	icud.icu_intime AS icu_intime,
	icud.icu_outtime AS icu_outtime,
	icud.los_icu AS icu_los,
    ROUND(CAST(AVG(gs.gcs) AS numeric), 2) AS gcs,
	ht.height AS height,
	ROUND(CAST(AVG(wd.weight) AS numeric), 2) AS weight,
	ROUND(CAST(AVG(vs.heart_rate) AS numeric), 2) AS heart_rate,
	ROUND(CAST(AVG(vs.sbp) AS numeric), 2) AS sbp,
	ROUND(CAST(AVG(vs.dbp) AS numeric), 2) AS dbp,
	ROUND(CAST(AVG(vs.mbp) AS numeric), 2) AS mbp,
	ROUND(CAST(AVG(vs.resp_rate) AS numeric), 2) AS resp_rate,
	CASE
        WHEN icud.dod IS NOT NULL THEN 1
        ELSE 0
    END AS death
FROM 
    public.id_features idf
LEFT JOIN mimiciv_derived.gcs gs ON idf.stay_id = gs.stay_id
LEFT JOIN mimiciv_derived.height ht ON idf.stay_id = ht.stay_id
LEFT JOIN mimiciv_derived.icustay_detail icud ON idf.stay_id = icud.stay_id
LEFT JOIN mimiciv_derived.vitalsign vs ON idf.stay_id = vs.stay_id
LEFT JOIN mimiciv_derived.weight_durations wd ON idf.stay_id = wd.stay_id
GROUP BY 
    idf.subject_id, idf.hadm_id, idf.stay_id, ht.height, icud.icu_intime, icud.icu_outtime, icud.los_icu, icud.dod
ORDER BY 
    idf.subject_id;

-- id_feature Table

CREATE TABLE id_features AS
SELECT nf.subject_id, nf.hadm_id, icu.stay_id
FROM public.normal_features nf
LEFT JOIN mimiciv_derived.icustay_detail icu ON nf.hadm_id = icu.hadm_id

-- result_table Table

CREATE TABLE result_table AS
SELECT 
	icuf.subject_id, 
	icuf.hadm_id, 
	icuf.stay_id, 
	nf.gender, 
	nf.age, 
	nf.admittime,
	nf.dischtime,
	icuf.icu_intime, 
	icuf.icu_outtime, 
	icuf.height,
	icuf.weight, 
	lf.ph, 
	lf.hemoglobin, 
	lf.temperature,
	lf.glucose, 
	lf.wbc, 
	lf.lymphocytes, 
	lf.hematocrit, 
	lf.platelet, 
	lf.rbc,
	icuf.gcs,
	icuf.heart_rate, 
	icuf.sbp, 
	icuf.dbp, 
	icuf.mbp, 
	icuf.resp_rate, 
	nf.disease_codes, 
	nf.surface_name, 
	nf.pharmacy_id, 
	nf.drug_name,
	nf.procedure_codes, 
	nf.procedures_description,
	nf.hosp_los, 
	icuf.icu_los, 
	nf.death_status, 
	icuf.death,
	ir.apsiii,
	ir.apsiii_prob,
	ir.aps_hr_score, 
	ir.aps_mbp_score, 
	ir.aps_temp_score, 
	ir.aps_resp_rate_score, 
	ir.aps_pao2_aado2_score, 
	ir.aps_hematocrit_score, 
	ir.aps_wbc_score, 
	ir.aps_creatinine_score, 
	ir.aps_uo_score, 
	ir.aps_bun_score, 
	ir.aps_sodium_score, 
	ir.aps_albumin_score, 
	ir.aps_bilirubin_score, 
	ir.aps_glucose_score, 
	ir.aps_acidbase_score, 
	ir.aps_gcs_score, 
	ir.sapsii,
	ir.sapsii_prob,
	ir.saps_age_score, 
	ir.saps_hr_score, 
	ir.saps_sysbp_score, 
	ir.saps_temp_score, 
	ir.saps_pao2fio2_score, 
	ir.saps_uo_score, 
	ir.saps_bun_score, 
	ir.saps_wbc_score, 
	ir.saps_potassium_score, 
	ir.saps_sodium_score, 
	ir.saps_bicarbonate_score, 
	ir.saps_bilirubin_score, 
	ir.saps_gcs_score, 
	ir.saps_comorbidity_score, 
	ir.saps_admissiontype_score
FROM public.icu_features icuf
LEFT JOIN public.normal_features nf ON icuf.hadm_id = nf.hadm_id
LEFT JOIN public.lab_features lf ON icuf.hadm_id = lf.hadm_id
LEFT JOIN public.icu_rating ir ON icuf.hadm_id = ir.hadm_id AND icuf.stay_id = ir.stay_id
ORDER BY icuf.subject_id, icuf.hadm_id, icuf.stay_id;

-- icu_rating Table

DROP TABLE IF EXISTS public.icu_rating;

CREATE TABLE public.icu_rating AS
SELECT 
    idf.subject_id,
    idf.hadm_id,
    idf.stay_id,

    aps.apsiii,
    aps.apsiii_prob,
    aps.hr_score AS aps_hr_score,
    aps.mbp_score AS aps_mbp_score,
    aps.temp_score AS aps_temp_score,
    aps.resp_rate_score AS aps_resp_rate_score,
    aps.pao2_aado2_score AS aps_pao2_aado2_score,
    aps.hematocrit_score AS aps_hematocrit_score,
    aps.wbc_score AS aps_wbc_score,
    aps.creatinine_score AS aps_creatinine_score,
    aps.uo_score AS aps_uo_score,
    aps.bun_score AS aps_bun_score,
    aps.sodium_score AS aps_sodium_score,
    aps.albumin_score AS aps_albumin_score,
    aps.bilirubin_score AS aps_bilirubin_score,
    aps.glucose_score AS aps_glucose_score,
    aps.acidbase_score AS aps_acidbase_score,
    aps.gcs_score AS aps_gcs_score,

    saps.sapsii,
    saps.sapsii_prob,
    saps.age_score AS saps_age_score,
    saps.hr_score AS saps_hr_score,
    saps.sysbp_score AS saps_sysbp_score,
    saps.temp_score AS saps_temp_score,
    saps.pao2fio2_score AS saps_pao2fio2_score,
    saps.uo_score AS saps_uo_score,
    saps.bun_score AS saps_bun_score,
    saps.wbc_score AS saps_wbc_score,
    saps.potassium_score AS saps_potassium_score,
    saps.sodium_score AS saps_sodium_score,
    saps.bicarbonate_score AS saps_bicarbonate_score,
    saps.bilirubin_score AS saps_bilirubin_score,
    saps.gcs_score AS saps_gcs_score,
    saps.comorbidity_score AS saps_comorbidity_score,
    saps.admissiontype_score AS saps_admissiontype_score

FROM 
    public.id_features AS idf

LEFT JOIN 
    mimiciv_derived.apsiii AS aps
ON 
    idf.subject_id = aps.subject_id 
    AND idf.hadm_id = aps.hadm_id 
    AND idf.stay_id = aps.stay_id

LEFT JOIN 
    mimiciv_derived.sapsii AS saps
ON 
    idf.subject_id = saps.subject_id 
    AND idf.hadm_id = saps.hadm_id 
    AND idf.stay_id = saps.stay_id;
