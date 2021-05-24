from pydantic import BaseModel


class HeartData(BaseModel):
    id: int = 0
    age: int = 18
    sex: int = 0
    cp: int = 0
    trestbps: int = 100
    chol: int = 245
    fbs: int = 0
    restecg: int = 0
    thalach: int = 100
    exang: int = 0
    oldpeak: float = 0
    slope: int = 0
    ca: int = 0
    thal: int = 0
