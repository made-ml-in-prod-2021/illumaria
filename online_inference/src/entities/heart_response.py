from pydantic import BaseModel


class HeartResponse(BaseModel):
    id: int
    target: int
