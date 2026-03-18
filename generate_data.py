"""
OE Recommendation System — Dummy Data Generator
================================================
Locked decisions:
  - 500 students, 30 professors, 5 branches
  - 8 OEs per branch per semester × 3 semesters = 120 OEs total
  - 40 OEs per semester pool, 32 eligible per student (own branch excluded)
  - Exactly 1 OE per student per semester (sem 5, 6, 7)
  - 30 seats per OE
  - Students cannot take OEs from their own branch
  - Students cannot repeat an OE across semesters

Tables produced:
  1. students.csv
  2. student_courses.csv
  3. student_oe.csv
  4. oe_info.csv          → oe_id, offering_branch, available_semester,
                              prof_id, is_new_oe, total_seats
  5. course_feedback.csv
  6. course_comments.csv

Run:
  pip install pandas numpy faker
  python generate_data.py
"""

import os
import random
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

os.makedirs("data/raw", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# LOCKED CONFIG
# ─────────────────────────────────────────────────────────────

NUM_STUDENTS   = 500
NUM_PROFESSORS = 30
BRANCHES       = ["CSE", "ECE", "ME", "CE", "EEE"]
BATCH_YEARS    = [2020, 2021, 2022, 2023]
OE_SEMESTERS   = [5, 6, 7]
TOTAL_SEATS    = 30

CORE_COURSES = {
    "CSE": ["CSE101", "CSE102", "CSE201", "CSE202", "CSE301", "CSE302"],
    "ECE": ["ECE101", "ECE102", "ECE201", "ECE202", "ECE301", "ECE302"],
    "ME" : ["ME101",  "ME102",  "ME201",  "ME202",  "ME301",  "ME302" ],
    "CE" : ["CE101",  "CE102",  "CE201",  "CE202",  "CE301",  "CE302" ],
    "EEE": ["EEE101", "EEE102", "EEE201", "EEE202", "EEE301", "EEE302"],
}

COURSE_SEM_MAP = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}

OE_CATALOGUE = {
    "CSE": {
        5: ["CSE_OE501_ML",            "CSE_OE502_DeepLearning",    "CSE_OE503_CloudComputing",
            "CSE_OE504_WebDev",        "CSE_OE505_Cybersecurity",   "CSE_OE506_BigData",
            "CSE_OE507_IoTProg",       "CSE_OE508_NLP"],
        6: ["CSE_OE601_ComputerVision","CSE_OE602_Blockchain",      "CSE_OE603_DevOps",
            "CSE_OE604_GameDev",       "CSE_OE605_AR_VR",           "CSE_OE606_Quantum",
            "CSE_OE607_EdgeAI",        "CSE_OE608_MLOps"],
        7: ["CSE_OE701_AutonomousSys", "CSE_OE702_DataPrivacy",     "CSE_OE703_FinTech",
            "CSE_OE704_HCI",           "CSE_OE705_SoftwareArch",    "CSE_OE706_AIEthics",
            "CSE_OE707_DigitalTwin",   "CSE_OE708_5GNetworks"],
    },
    "ECE": {
        5: ["ECE_OE501_VLSI",          "ECE_OE502_EmbeddedSys",     "ECE_OE503_IoTHardware",
            "ECE_OE504_SignalProc",    "ECE_OE505_Microcontrollers","ECE_OE506_PCBDesign",
            "ECE_OE507_Antenna",       "ECE_OE508_OpticalComm"],
        6: ["ECE_OE601_5G_Wireless",   "ECE_OE602_RFDesign",        "ECE_OE603_PowerElec",
            "ECE_OE604_Photonics",     "ECE_OE605_MEMS",            "ECE_OE606_RadarSys",
            "ECE_OE607_NanoElec",      "ECE_OE608_SensorNet"],
        7: ["ECE_OE701_AutomotiveElec","ECE_OE702_MedElec",         "ECE_OE703_SatelliteComm",
            "ECE_OE704_SpaceElec",     "ECE_OE705_FPGA",            "ECE_OE706_WirelessPower",
            "ECE_OE707_Terahertz",     "ECE_OE708_QuantumComm"],
    },
    "ME": {
        5: ["ME_OE501_Robotics",       "ME_OE502_CAD_CAM",          "ME_OE503_ThermalEng",
            "ME_OE504_FluidMech",      "ME_OE505_Composite",        "ME_OE506_3DPrinting",
            "ME_OE507_Tribology",      "ME_OE508_NanoMfg"],
        6: ["ME_OE601_AutomotiveEng",  "ME_OE602_Aerospace",        "ME_OE603_RenewableEnergy",
            "ME_OE604_Mechatronics",   "ME_OE605_FEA",              "ME_OE606_BioMech",
            "ME_OE607_Acoustics",      "ME_OE608_CryoEng"],
        7: ["ME_OE701_IndustrialAuto", "ME_OE702_SmartMfg",         "ME_OE703_HydrogenEnergy",
            "ME_OE704_SpaceProp",      "ME_OE705_TurbineDes",       "ME_OE706_Nanotech",
            "ME_OE707_SustainableMfg", "ME_OE708_AdvancedRobotics"],
    },
    "CE": {
        5: ["CE_OE501_GIS",            "CE_OE502_SmartCity",        "CE_OE503_StructAnalysis",
            "CE_OE504_Geotechnical",   "CE_OE505_TransportEng",     "CE_OE506_WaterResources",
            "CE_OE507_SoilMech",       "CE_OE508_BuildingInfo"],
        6: ["CE_OE601_EarthquakeEng",  "CE_OE602_RemoteSensing",    "CE_OE603_UrbanPlanning",
            "CE_OE604_SustainableConst","CE_OE605_CoastalEng",      "CE_OE606_EnvironImpact",
            "CE_OE607_BridgeDesign",   "CE_OE608_HydroEng"],
        7: ["CE_OE701_SmartInfra",     "CE_OE702_DisasterMgmt",     "CE_OE703_GreenBuilding",
            "CE_OE704_ConstructionMgmt","CE_OE705_TunnelEng",       "CE_OE706_SeismicDesign",
            "CE_OE707_WasteManagement","CE_OE708_ClimateAdapt"],
    },
    "EEE": {
        5: ["EEE_OE501_PowerSys",      "EEE_OE502_ElecDrives",      "EEE_OE503_RenewableEE",
            "EEE_OE504_SmartGrid",     "EEE_OE505_HighVoltage",     "EEE_OE506_PowerElec",
            "EEE_OE507_ElecMachines",  "EEE_OE508_EnergyStor"],
        6: ["EEE_OE601_HVDC",          "EEE_OE602_FuelCells",       "EEE_OE603_ElecVehicles",
            "EEE_OE604_Microgrids",    "EEE_OE605_PowerQuality",    "EEE_OE606_NuclearPower",
            "EEE_OE607_DistribGen",    "EEE_OE608_WirelessEE"],
        7: ["EEE_OE701_EnergyAudit",   "EEE_OE702_SolarFarm",       "EEE_OE703_WindEnergy",
            "EEE_OE704_BatteryTech",   "EEE_OE705_GridCyber",       "EEE_OE706_Superconductors",
            "EEE_OE707_TidalEnergy",   "EEE_OE708_SpacePower"],
    },
}

GRADE_OPTIONS  = ["O", "A+", "A", "B+", "B", "C", "F"]
GRADE_WEIGHTS  = [0.10, 0.15, 0.25, 0.20, 0.15, 0.10, 0.05]
GRADE_TO_SCORE = {
    "O": 1.0, "A+": 0.9, "A": 0.8,
    "B+": 0.7, "B": 0.6, "C": 0.5, "F": 0.0,
}

COMMENTS = {
    "positive": [
        "Very engaging lectures, the professor explains concepts clearly.",
        "One of the best courses I have taken. Highly interactive sessions.",
        "Assignments were well structured and helped reinforce learning.",
        "Professor is very approachable and always ready to help students.",
        "The course content was relevant and up to date with industry trends.",
        "Excellent teaching style, made difficult topics easy to understand.",
        "Very well organized course with clear learning outcomes.",
        "The professor's real-world examples made the subject very interesting.",
    ],
    "neutral": [
        "Average course. Content was okay but could have been more detailed.",
        "Lectures were fine. Nothing extraordinary but covered the syllabus.",
        "The course was decent. Assignments could be more challenging.",
        "Professor was okay. Explanations were sometimes unclear.",
        "Course organization was average. Slides were not always up to date.",
        "Some topics were skipped due to time constraints.",
    ],
    "negative": [
        "Very boring lectures. Professor reads directly from slides.",
        "Assignments were too difficult and not related to what was taught.",
        "The professor was often unavailable outside of class hours.",
        "Poor course structure. Topics were taught out of order.",
        "The content felt outdated and not relevant to current practices.",
        "Very disappointing course. Expected much more depth.",
    ],
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def random_grade():
    return np.random.choice(GRADE_OPTIONS, p=GRADE_WEIGHTS)

def random_rating(mean=3.5, std=0.8):
    return round(float(np.clip(np.random.normal(mean, std), 1.0, 5.0)), 2)


# ─────────────────────────────────────────────────────────────
# TABLE 1 — students
# ─────────────────────────────────────────────────────────────

def generate_students():
    rows = []
    for i in range(1, NUM_STUDENTS + 1):
        rows.append({
            "student_id": f"STU{i:04d}",
            "branch"    : random.choice(BRANCHES),
            "cgpa"      : round(random.uniform(5.5, 10.0), 2),
            "batch_year": random.choice(BATCH_YEARS),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# TABLE 2 — student_courses
# ─────────────────────────────────────────────────────────────

def generate_student_courses(students_df):
    rows = []
    for _, s in students_df.iterrows():
        for idx, course_id in enumerate(CORE_COURSES[s["branch"]]):
            rows.append({
                "student_id": s["student_id"],
                "course_id" : course_id,
                "grade"     : random_grade(),
                "semester"  : COURSE_SEM_MAP[idx],
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# TABLE 3 — student_oe
# Rules:
#   - exactly 1 OE per student per OE semester
#   - cannot pick own branch OE
#   - cannot repeat an OE from a previous semester
# ─────────────────────────────────────────────────────────────

def generate_student_oe(students_df, oe_info_df):
    rows = []
    for _, s in students_df.iterrows():
        taken_oes = set()

        for sem in OE_SEMESTERS:
            eligible = oe_info_df[
                (oe_info_df["available_semester"] == sem) &
                (oe_info_df["offering_branch"]    != s["branch"]) &
                (~oe_info_df["oe_id"].isin(taken_oes))
            ]["oe_id"].tolist()

            if not eligible:
                continue

            chosen = random.choice(eligible)
            taken_oes.add(chosen)

            rows.append({
                "student_id": s["student_id"],
                "oe_id"     : chosen,
                "grade"     : random_grade(),
                "semester"  : sem,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# TABLE 4 — oe_info
# ─────────────────────────────────────────────────────────────

def generate_oe_info():
    prof_ids = [f"PROF{i:03d}" for i in range(1, NUM_PROFESSORS + 1)]
    rows = []
    for branch, sems in OE_CATALOGUE.items():
        for sem, oes in sems.items():
            for oe_id in oes:
                rows.append({
                    "oe_id"             : oe_id,
                    "offering_branch"   : branch,
                    "available_semester": sem,
                    "prof_id"           : random.choice(prof_ids),
                    "is_new_oe"         : 0,
                    "total_seats"       : TOTAL_SEATS,
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# TABLE 5 — course_feedback
# ─────────────────────────────────────────────────────────────

def generate_course_feedback():
    prof_ids    = [f"PROF{i:03d}" for i in range(1, NUM_PROFESSORS + 1)]
    all_courses = [c for courses in CORE_COURSES.values() for c in courses]

    # Assign a fixed bias per professor so some are consistently better
    prof_bias_map = {p: random.uniform(-0.6, 0.6) for p in prof_ids}

    rows = []
    for course_id in all_courses:
        for sem in range(1, 7):
            prof_id = random.choice(prof_ids)
            bias    = prof_bias_map[prof_id]
            rows.append({
                "course_id"                : course_id,
                "prof_id"                  : prof_id,
                "semester"                 : sem,
                "avg_assignment_usefulness": random_rating(3.5 + bias),
                "avg_teaching_clarity"     : random_rating(3.6 + bias),
                "avg_course_organization"  : random_rating(3.4 + bias),
                "avg_interaction"          : random_rating(3.5 + bias),
                "avg_overall_rating"       : random_rating(3.5 + bias),
                "num_responses"            : random.randint(20, 80),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# TABLE 6 — course_comments
# ─────────────────────────────────────────────────────────────

def generate_course_comments(course_feedback_df):
    rows = []
    for _, row in course_feedback_df.iterrows():
        rating = row["avg_overall_rating"]
        if rating >= 4.0:
            weights = [0.70, 0.20, 0.10]
        elif rating >= 3.0:
            weights = [0.30, 0.50, 0.20]
        else:
            weights = [0.10, 0.30, 0.60]

        sentiments = np.random.choice(
            ["positive", "neutral", "negative"],
            size=random.randint(3, 8),
            p=weights,
        )
        for sentiment in sentiments:
            rows.append({
                "course_id": row["course_id"],
                "prof_id"  : row["prof_id"],
                "semester" : row["semester"],
                "comment"  : random.choice(COMMENTS[sentiment]),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# SANITY CHECKS
# ─────────────────────────────────────────────────────────────

def run_sanity_checks(students_df, student_oe_df, oe_info_df):
    print("\n--- Sanity Checks ---")

    merged = (
        student_oe_df
        .merge(oe_info_df[["oe_id", "offering_branch", "available_semester"]], on="oe_id")
        .merge(students_df[["student_id", "branch"]], on="student_id")
    )

    # 1. No student took their own branch OE
    v1 = merged[merged["branch"] == merged["offering_branch"]]
    print(f"  Branch exclusion violations  : {len(v1):>4}  (expected 0)")

    # 2. Each student took exactly 1 OE per OE semester
    counts = student_oe_df.groupby(["student_id", "semester"]).size()
    v2     = counts[counts > 1]
    print(f"  >1 OE per student per sem    : {len(v2):>4}  (expected 0)")

    # 3. No student repeated an OE
    v3 = student_oe_df[student_oe_df.duplicated(["student_id", "oe_id"])]
    print(f"  Repeated OE by same student  : {len(v3):>4}  (expected 0)")

    # 4. OE taken in correct semester
    v4 = merged[merged["semester"] != merged["available_semester"]]
    print(f"  OE taken in wrong semester   : {len(v4):>4}  (expected 0)")

    print("---------------------")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating tables...")

    students_df        = generate_students()
    oe_info_df         = generate_oe_info()
    student_courses_df = generate_student_courses(students_df)
    student_oe_df      = generate_student_oe(students_df, oe_info_df)
    course_feedback_df = generate_course_feedback()
    course_comments_df = generate_course_comments(course_feedback_df)

    students_df.to_csv("data/raw/students.csv",               index=False)
    student_courses_df.to_csv("data/raw/student_courses.csv", index=False)
    student_oe_df.to_csv("data/raw/student_oe.csv",           index=False)
    oe_info_df.to_csv("data/raw/oe_info.csv",                 index=False)
    course_feedback_df.to_csv("data/raw/course_feedback.csv", index=False)
    course_comments_df.to_csv("data/raw/course_comments.csv", index=False)

    print("\n✓ All tables saved to data/raw/\n")
    print(f"  students         : {len(students_df):>6} rows")
    print(f"  student_courses  : {len(student_courses_df):>6} rows  (6 courses × {NUM_STUDENTS} students)")
    print(f"  student_oe       : {len(student_oe_df):>6} rows  (1 OE × 3 sems × {NUM_STUDENTS} students)")
    print(f"  oe_info          : {len(oe_info_df):>6} rows  (40 OEs/sem × 3 sems)")
    print(f"  course_feedback  : {len(course_feedback_df):>6} rows")
    print(f"  course_comments  : {len(course_comments_df):>6} rows")

    run_sanity_checks(students_df, student_oe_df, oe_info_df)
