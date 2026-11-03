"""Make data/synthetic_admissions.csv: made-up hospital admissions for the dstoolkit demos.

Every value is generated here from a fixed seed; no real patient is behind any row.
The data is built to contain the problems dstoolkit looks for:
  - several admissions per patient (patient_id repeats), and a hidden per-patient risk
  - dates stored as text, with length of stay as a real signal
  - diagnosis codes with a long tail of rare values
  - lab results (hba1c) missing more often for low-risk patients: the test wasn't ordered
  - two columns recorded after the outcome: readmission_cost (blank unless readmitted)
    and followup_outcome (from a call 30 days later)

Run from the repository root:  python data/make_admissions.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
n_patients = 1500
visits = rng.poisson(1.7, n_patients) + 1                       # 1 or more admissions each
patient = np.repeat(np.arange(n_patients), visits)
n = len(patient)

frailty = rng.normal(0, 1, n_patients)                           # hidden per-patient risk
birth_year = rng.integers(1935, 2000, n_patients)
sex = rng.choice(["F", "M"], n_patients)
admitted = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
age = admitted.year - birth_year[patient]
admission_type = rng.choice(["Emergency", "Urgent", "Elective"], n, p=[0.55, 0.25, 0.20])
stay = np.clip(np.round(rng.gamma(2.0, 2.2, n) + (admission_type == "Emergency") * 1.5), 1, 40).astype(int)
discharged = admitted + pd.to_timedelta(stay, unit="D")
prior = rng.poisson(np.exp(0.3 * frailty[patient]))

common = [f"I{i:02d}" for i in range(10)]                        # a few common codes...
tail = [f"R{i:03d}" for i in range(180)]                          # ...and many rare ones
dx = np.where(rng.random(n) < 0.75, rng.choice(common, n), rng.choice(tail, n))
dx_risk = {c: rng.normal(0, 0.4) for c in common}

hba1c = np.round(rng.normal(6.2, 1.1, n) + 0.4 * frailty[patient], 1)
logit = (-2.6 + 0.9 * frailty[patient] + 0.035 * (age - 60) + 0.12 * (stay - 5) + 0.25 * prior
         + 0.5 * (admission_type == "Emergency") + 0.3 * (hba1c > 7.5) + np.array([dx_risk.get(c, 0.3) for c in dx]))
readmitted = rng.random(n) < 1 / (1 + np.exp(-logit))

ordered = rng.random(n) < np.clip(0.35 + 0.25 * frailty[patient] + 0.2 * (admission_type == "Emergency"), 0.05, 0.95)
followup = np.where(readmitted, rng.choice(["Readmitted", "No answer", "Doing well"], n, p=[0.5, 0.35, 0.15]),
                    rng.choice(["No answer", "Doing well"], n, p=[0.55, 0.45]))

df = pd.DataFrame({
    "admission_id": [f"A{i:06d}" for i in range(n)],
    "patient_id": [f"P{p:05d}" for p in patient],
    "admitted": admitted.strftime("%Y-%m-%d"),
    "discharged": discharged.strftime("%Y-%m-%d"),
    "age": age.astype(float),
    "sex": sex[patient],
    "admission_type": admission_type,
    "diagnosis_code": dx,
    "prior_admissions_12m": prior,
    "hba1c": np.where(ordered, hba1c, np.nan),
    "readmission_cost": np.where(readmitted, np.round(rng.lognormal(9, 0.6, n), 2), np.nan),
    "followup_outcome": followup,
    "readmitted_30d": np.where(readmitted, "yes", "no"),
})
df.loc[rng.random(n) < 0.03, "age"] = np.nan                     # a few ages not recorded
df = df.sort_values("admitted", kind="stable").reset_index(drop=True)
df.to_csv(Path(__file__).with_name("synthetic_admissions.csv"), index=False)
print(f"{len(df):,} admissions for {n_patients:,} patients; {df['readmitted_30d'].eq('yes').mean():.1%} readmitted")
