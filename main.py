from fastapi import FastAPI, HTTPException, Depends,Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
from jose import jwt, JWTError
from groq import Groq
from dotenv import load_dotenv
from pydantic import EmailStr
from typing import Literal
import email_validator
from fastapi.responses import JSONResponse


load_dotenv()  

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # set in environment
groq_client = Groq(api_key=GROQ_API_KEY)
# ==================================================
# App + Config
# ==================================================
app = FastAPI(title="Dropout Risk Prediction API")

JWT_SECRET = "CHANGE_ME_TO_A_RANDOM_SECRET"   # move to env in real deployments
JWT_ALG = "HS256"
JWT_EXPIRE_MIN = 60 * 12  # 12 hours

security = HTTPBearer()

# ==================================================
# Load Model Package
# ==================================================
MODEL_PATH = os.path.join("model", "model.pkl")
MENTOR_CSV = os.path.join("data", "mentor.csv")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found.")

with open(MODEL_PATH, "rb") as f:
    model_package = pickle.load(f)

model = model_package["model"]
feature_columns = model_package["feature_columns"]

# ==================================================
# Load Teachers CSV
# ==================================================
TEACHERS_CSV = os.path.join("data", "teachers.csv")
if not os.path.exists(TEACHERS_CSV):
    raise FileNotFoundError("teachers.csv not found in data/ folder")

teachers_df = pd.read_csv(TEACHERS_CSV)

# Normalize emails for lookup
teachers_df["email_norm"] = teachers_df["email"].astype(str).str.strip().str.lower()

# ==================================================
# Schemas
# ==================================================
class TeacherLogin(BaseModel):
    email: str
    password: str

class StudentInput(BaseModel):
    current_year: int = Field(..., ge=1, le=4)
    current_semester: int = Field(..., ge=1, le=8)
    attendance_rate: int = Field(..., ge=0, le=100)
    avg_internal: float = Field(..., ge=0, le=100)
    cgpa: float = Field(..., ge=0, le=10)
    backlog_count: int = Field(..., ge=0)
    fee_status: int = Field(..., ge=0, le=1)
    normalized_engagement: float = Field(..., ge=0, le=1)
    scholarship_status: int = Field(..., ge=0, le=1)

    class Config:
        extra = "forbid"

class StudentCreate(BaseModel):
    student_uid: int = Field(..., ge=1)
    college_code: str = Field(..., min_length=3, max_length=20)   # e.g. COL-011
    dept_code: str = Field(..., min_length=2, max_length=20)      # e.g. CSE, ECE
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr

    batch_year: int = Field(..., ge=2000, le=2100)

    current_year: int = Field(..., ge=1, le=4)
    current_semester: int = Field(..., ge=1, le=8)

    attendance_rate: int = Field(..., ge=0, le=100)
    avg_internal: float = Field(..., ge=0, le=100)
    cgpa: float = Field(..., ge=0, le=10)

    backlog_count: int = Field(..., ge=0, le=50)

    # Matches your CSV values exactly: paid/unpaid/delayed
    fee_status: Literal["paid", "unpaid", "delayed"]

    normalized_engagement: float = Field(..., ge=0, le=1)

    # Matches your CSV values exactly: yes/no
    scholarship_status: Literal["yes", "no"]

    class Config:
        extra = "forbid"   # reject unknown keys
class MentorAssignRequest(BaseModel):
    student_id: int = Field(..., ge=1)
    mentor_id: str = Field(..., min_length=3)   # teacher_uid like "T203219567"
    enforce_high_risk: bool = True              # if True => only HIGH students can be assigned
# ==================================================
# JWT Helpers (TEACHERS ONLY)
# ==================================================


def create_access_token(payload: dict) -> str:
    exp = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MIN)
    to_encode = {**payload, "exp": exp}
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)

def get_current_teacher(
    creds: HTTPAuthorizationCredentials = Depends(security),
):
    token = creds.credentials
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        teacher_uid = data.get("teacher_uid")
        email = data.get("email")
        if not teacher_uid or not email:
            raise HTTPException(status_code=401, detail="Invalid token payload")

        # verify teacher still exists
        row = teachers_df[teachers_df["teacher_uid"] == teacher_uid]
        if row.empty:
            raise HTTPException(status_code=401, detail="Teacher not found")

        return row.iloc[0].to_dict()

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid/expired token")
def generate_explanation(student_data: dict, prediction: int, probability: float):
    risk_map = {
        0: "LOW RISK",
        1: "MEDIUM RISK",
        2: "HIGH RISK"
    }

    prompt = f"""
You are an academic risk analysis expert.

A student has been classified as: {risk_map.get(prediction)}.

Prediction confidence: {probability}

Student Information:
{student_data}

Explain clearly and professionally why this student falls into this category.
Tell the user how he can improve.
Base your reasoning on academic performance, attendance, engagement, backlog, and financial indicators.

Give a structured explanation:
1. Key Strengths
2. Key Risk Factors
3. Overall Interpretation
4. Suggested Intervention Strategy

Be precise, analytical, and actionable.
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an expert academic risk analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=600
    )

    return response.choices[0].message.content


RISK_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

def prepare_features_for_model(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()

    # Convert numeric columns safely
    numeric_cols = [
        "current_year",
        "current_semester",
        "attendance_rate",
        "avg_internal",
        "cgpa",
        "backlog_count",
        "normalized_engagement",
    ]
    for c in numeric_cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    # Map fee_status (paid/unpaid/delayed) -> 1/0 like your StudentInput expects
    if "fee_status" in df2.columns:
        df2["fee_status"] = (
            df2["fee_status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"paid": 1, "unpaid": 0, "delayed": 0})
        )

    # Map scholarship_status (yes/no) -> 1/0
    if "scholarship_status" in df2.columns:
        df2["scholarship_status"] = (
            df2["scholarship_status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0})
        )

    # Ensure we have all features in the right order
    X = df2.reindex(columns=feature_columns)

    # Handle missing values (for hackathon/demo, fill 0)
    X = X.fillna(0)

    return X

# ==================================================
# Routes
# ==================================================
@app.get("/")
def root():
    return {"message": "Dropout Risk Prediction API Running"}

# ✅ Teacher Login (ONLY teachers authenticate)
@app.post("/auth/login")
def teacher_login(body: TeacherLogin):
    email_norm = body.email.strip().lower()
    row = teachers_df[teachers_df["email_norm"] == email_norm]

    if row.empty:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    teacher = row.iloc[0].to_dict()

    # NOTE: CSV currently has plaintext passwords
    if str(teacher["password"]) != body.password:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({
        "teacher_uid": teacher["teacher_uid"],
        "email": teacher["email"],
        "college_code": teacher["college_code"],
        "dept_code": teacher["dept_code"],
    })

    return {
        "access_token": token,
        "token_type": "bearer",
        "teacher": {
            "teacher_uid": teacher["teacher_uid"],
            "name": teacher["name"],
            "email": teacher["email"],
            "college_code": teacher["college_code"],
            "dept_code": teacher["dept_code"],
        }
    }

# ✅ Predict endpoint takes student data, BUT authenticates ONLY teachers
# ❌ No student login/authentication exists anywhere
@app.post("/predict")
def predict(data: StudentInput, teacher=Depends(get_current_teacher)):
    try:
        input_df = pd.DataFrame([data.model_dump()])
        input_df = input_df[feature_columns]

        pred = int(model.predict(input_df)[0])

        confidence = None
        risk_probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            confidence = float(proba[pred])
            if len(proba) > 1:
                risk_probability = float(proba[1])

        # 🔥 Generate LLM Explanation
        explanation = generate_explanation(
            student_data=data.model_dump(),
            prediction=pred,
            probability=confidence
        )

        return {
            "teacher_uid": teacher["teacher_uid"],
            "risk_level_prediction": pred,
            "confidence": confidence,
            "risk_probability": risk_probability,
            "explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

STUDENTS_CSV = os.path.join("data", "students.csv")

@app.post("/students")
def add_student(student: StudentCreate, teacher=Depends(get_current_teacher)):
    try:
        new_data = pd.DataFrame([student.model_dump()])

        if os.path.exists(STUDENTS_CSV):
            new_data.to_csv(STUDENTS_CSV, mode='a', header=False, index=False)
        else:
            new_data.to_csv(STUDENTS_CSV, index=False)

        return {"message": "Student added successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.get("/students")
def get_students(
    sort_by: str = Query("id", enum=["id", "gpa", "risk"]),
    order: str = Query("asc", enum=["asc", "desc"]),
    teacher=Depends(get_current_teacher)
):
    if not os.path.exists(STUDENTS_CSV):
        return JSONResponse(content=[])

    df = pd.read_csv(STUDENTS_CSV)

    if df.empty:
        return JSONResponse(content=[])

    # Prepare model features
    X = prepare_features_for_model(df)
    preds = model.predict(X)

    df["risk_pred"] = preds
    risk_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    df["risk_label"] = df["risk_pred"].map(risk_map)

    # Convert numeric safely
    df["student_uid"] = pd.to_numeric(df["student_uid"], errors="coerce")
    df["cgpa"] = pd.to_numeric(df["cgpa"], errors="coerce")

    ascending = True if order == "asc" else False

    # 🔥 Sorting logic based on UI selection
    if sort_by == "id":
        df = df.sort_values(by="student_uid", ascending=ascending)

    elif sort_by == "gpa":
        df = df.sort_values(by="cgpa", ascending=ascending)

    elif sort_by == "risk":
        # Custom ranking for risk
        risk_rank = {2: 3, 1: 2, 0: 1}
        df["risk_rank"] = df["risk_pred"].map(risk_rank)

        df = df.sort_values(by="risk_rank", ascending=ascending)
        df = df.drop(columns=["risk_rank"])

    return JSONResponse(content=df.to_dict(orient="records"))


@app.get("/dashboard")
def dashboard(teacher=Depends(get_current_teacher)):
    if not os.path.exists(STUDENTS_CSV):
        return JSONResponse(content={
            "total_students": 0,
            "risk_counts": {"LOW": 0, "MEDIUM": 0, "HIGH": 0},
            "semester_trend": [],
            "high_risk_semesterwise": []
        })

    df = pd.read_csv(STUDENTS_CSV)

    if df.empty:
        return JSONResponse(content={
            "total_students": 0,
            "risk_counts": {"LOW": 0, "MEDIUM": 0, "HIGH": 0},
            "semester_trend": [],
            "high_risk_semesterwise": []
        })

    # Batch predict on entire CSV
    X = prepare_features_for_model(df)
    preds = model.predict(X)

    df["risk_pred"] = preds
    df["risk_label"] = df["risk_pred"].map(RISK_LABELS)

    total_students = int(len(df))

    # Overall risk counts
    vc = df["risk_label"].value_counts().to_dict()
    risk_counts = {
        "LOW": int(vc.get("LOW", 0)),
        "MEDIUM": int(vc.get("MEDIUM", 0)),
        "HIGH": int(vc.get("HIGH", 0)),
    }

    # Semester trend (sem-wise low/med/high)
    if "current_semester" not in df.columns:
        raise HTTPException(status_code=400, detail="current_semester column not found in students.csv")

    trend_raw = (
        df.groupby(["current_semester", "risk_label"])
          .size()
          .reset_index(name="count")
    )

    semester_trend = []
    semesters = sorted([int(x) for x in df["current_semester"].dropna().unique()])
    for sem in semesters:
        row = {"semester": sem, "LOW": 0, "MEDIUM": 0, "HIGH": 0}
        sub = trend_raw[trend_raw["current_semester"] == sem]
        for _, r in sub.iterrows():
            row[str(r["risk_label"])] = int(r["count"])
        semester_trend.append(row)

    high_risk_semesterwise = [{"semester": x["semester"], "high_risk": x["HIGH"]} for x in semester_trend]

    return JSONResponse(content={
        "total_students": total_students,
        "risk_counts": risk_counts,
        "semester_trend": semester_trend,
        "high_risk_semesterwise": high_risk_semesterwise
    })
@app.post("/assign-mentor")
def assign_mentor(body: MentorAssignRequest, teacher=Depends(get_current_teacher)):
    # 1) Validate files
    if not os.path.exists(STUDENTS_CSV):
        raise HTTPException(status_code=400, detail="students.csv not found")

    df_students = pd.read_csv(STUDENTS_CSV)
    if df_students.empty:
        raise HTTPException(status_code=400, detail="No students found")

    # 2) Validate student exists
    df_students["student_uid"] = pd.to_numeric(df_students["student_uid"], errors="coerce")
    student_row = df_students[df_students["student_uid"] == body.student_id]
    if student_row.empty:
        raise HTTPException(status_code=404, detail="Student not found")

    # 3) Validate mentor exists
    mentor_row = teachers_df[teachers_df["teacher_uid"].astype(str) == str(body.mentor_id)]
    if mentor_row.empty:
        raise HTTPException(status_code=404, detail="Mentor (teacher_uid) not found")

    mentor = mentor_row.iloc[0]

    # 4) (Optional) enforce HIGH risk only
    if body.enforce_high_risk:
        X = prepare_features_for_model(df_students)
        preds = model.predict(X)
        df_students["risk_pred"] = preds
        df_students["risk_label"] = df_students["risk_pred"].map(RISK_LABELS)

        student_risk = df_students[df_students["student_uid"] == body.student_id].iloc[0]["risk_label"]
        if student_risk != "HIGH":
            raise HTTPException(
                status_code=400,
                detail=f"Student is not HIGH risk (current risk={student_risk}). Set enforce_high_risk=false to override."
            )

    # 5) Prepare assignment record
    assignment = {
        "mentor_id": str(mentor["teacher_uid"]),
        "mentor_name": str(mentor["name"]),
        "student_id": int(body.student_id),
        "assigned_at": datetime.utcnow().isoformat()
    }

    # 6) Write to mentor.csv (create if missing, update if student already assigned)
    if os.path.exists(MENTOR_CSV):
        df_mentor = pd.read_csv(MENTOR_CSV)

        # Ensure columns exist (safe for first-time file formats)
        for col in ["mentor_id", "mentor_name", "student_id", "assigned_at"]:
            if col not in df_mentor.columns:
                df_mentor[col] = None

        df_mentor["student_id"] = pd.to_numeric(df_mentor["student_id"], errors="coerce")

        # If student already has mentor => overwrite that row
        if (df_mentor["student_id"] == body.student_id).any():
            df_mentor.loc[df_mentor["student_id"] == body.student_id, ["mentor_id", "mentor_name", "assigned_at"]] = [
                assignment["mentor_id"], assignment["mentor_name"], assignment["assigned_at"]
            ]
        else:
            df_mentor = pd.concat([df_mentor, pd.DataFrame([assignment])], ignore_index=True)

        df_mentor.to_csv(MENTOR_CSV, index=False)

    else:
        pd.DataFrame([assignment]).to_csv(MENTOR_CSV, index=False)

    # 7) Return assignment
    return {
        "message": "Mentor assigned successfully",
        "assignment": assignment
    }