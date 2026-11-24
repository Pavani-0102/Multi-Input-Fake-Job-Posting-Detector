"""TEST-ONLY synthetic data in the EMSCAD column layout.

This module exists solely so unit/integration tests can exercise the pipeline without
the real dataset. It is never used by train_model.py or the app, and metrics from it
mean nothing about real-world performance.
"""
from __future__ import annotations

import random

import pandas as pd

_LEGIT_TITLES = ["Software Engineer", "Data Analyst", "Marketing Manager", "Registered Nurse", "Accountant", "Mechanical Engineer", "HR Generalist"]
_FRAUD_TITLES = ["Home Based Data Entry", "Earn Money Online", "Urgent Hiring Typist", "Work From Home Assistant", "Online Form Filling"]
_LEGIT_SENT = [
    "You will collaborate with cross functional teams to design and maintain scalable systems.",
    "We offer health insurance, retirement plans and a structured career development program.",
    "The ideal candidate holds a bachelor degree and has three years of relevant professional experience.",
    "Responsibilities include preparing quarterly reports, reviewing budgets and mentoring junior staff.",
    "Our company is an established employer with offices in several cities and a strong engineering culture.",
    "Candidates will attend a panel interview and a technical assessment before an offer is made.",
]
_FRAUD_SENT = [
    "No experience needed, earn 5000 per week working from home with guaranteed income.",
    "Pay a small registration fee to receive your training kit and start immediately.",
    "Send your bank account number and aadhaar copy to confirm your selection today.",
    "Contact our manager on WhatsApp or at hiring.team@gmail.com, urgent hiring, limited seats!",
    "Congratulations you are selected, no interview required, direct joining available.",
    "Simple copy paste typing jobs, daily payout, apply now and hurry!",
]


def make_synthetic_jobs(n: int = 600, fraud_ratio: float = 0.15, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        fraud = rng.random() < fraud_ratio
        sent_pool = _FRAUD_SENT if fraud else _LEGIT_SENT
        title_pool = _FRAUD_TITLES if fraud else _LEGIT_TITLES
        rows.append({
            "job_id": i,
            "title": rng.choice(title_pool),
            "location": rng.choice(["US, NY, New York", "GB, LND, London", "IN, KA, Bengaluru", ""]),
            "department": "",
            "salary_range": rng.choice(["", "20000-30000"]),
            "company_profile": "" if fraud and rng.random() < 0.7 else f"Company {rng.randint(1, 9999)} builds products for customers worldwide.",
            "description": " ".join(rng.sample(sent_pool, 3)) + f" Reference {i}.",
            "requirements": "" if fraud else " ".join(rng.sample(_LEGIT_SENT, 2)),
            "benefits": "",
            "telecommuting": 0,
            "has_company_logo": 1,
            "has_questions": 0,
            "employment_type": rng.choice(["Full-time", "Part-time", "Contract", ""]),
            "required_experience": "",
            "required_education": "",
            "industry": "",
            "function": "",
            "fraudulent": int(fraud),
        })
    return pd.DataFrame(rows)
