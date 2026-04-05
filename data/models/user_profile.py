from pydantic import BaseModel
from typing import List, Optional

class UserProfile(BaseModel):
    name: str
    age: int
    gender: str
    occupation: str
    annual_income: float
    is_farmer: bool = False
    is_below_poverty_line: bool = False
    has_preexisting_conditions: bool = False
    location: Optional[str] = "Rural India"
    additional_info: Optional[str] = None

class RecommendedPolicy(BaseModel):
    policy_name: str
    eligibility_status: str  # e.g., "Highly Recommended", "Eligible", "Not Eligible"
    reasoning: str
    benefits: List[str]

class RecommendationResponse(BaseModel):
    user_name: str
    recommendations: List[RecommendedPolicy]
    summary: str