import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

model = joblib.load("student_model.pkl")
app = FastAPI(title="Student Performance Prediction")


class Student(BaseModel):
    study_hours: float = Field(ge=0, le=24)
    attendance: float = Field(ge=0, le=100)


@app.get("/")
def home():
    return {"message": "Student Prediction API is running"}


@app.post("/predict")
def predict(student: Student):
    result = int(model.predict([[
        student.study_hours,
        student.attendance
    ]])[0])

    return {
        "prediction": "Pass" if result == 1 else "Fail"
    }
