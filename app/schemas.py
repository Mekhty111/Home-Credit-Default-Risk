from pydantic import BaseModel, Field
from typing import Optional

class ApplicantInput(BaseModel):
    """Input schema for a single loan applicant."""

    EXT_SOURCE_1            : Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2            : Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3            : Optional[float] = Field(None, ge=0, le=1)
    DAYS_BIRTH              : Optional[int]   = Field(None, description="Negative: days before application")
    DAYS_EMPLOYED           : Optional[int]   = Field(None)
    AMT_INCOME_TOTAL        : Optional[float] = Field(None, ge=0)
    AMT_CREDIT              : Optional[float] = Field(None, ge=0)
    AMT_ANNUITY             : Optional[float] = Field(None, ge=0)
    AMT_GOODS_PRICE         : Optional[float] = Field(None, ge=0)
    CODE_GENDER             : Optional[str]   = Field(None, pattern="^(M|F)$")
    NAME_CONTRACT_TYPE      : Optional[str]   = None
    NAME_EDUCATION_TYPE     : Optional[str]   = None
    NAME_INCOME_TYPE        : Optional[str]   = None
    NAME_FAMILY_STATUS      : Optional[str]   = None
    NAME_HOUSING_TYPE       : Optional[str]   = None
    REGION_RATING_CLIENT    : Optional[int]   = Field(None, ge=1, le=3)
    DAYS_LAST_PHONE_CHANGE  : Optional[float] = None
    FLAG_OWN_CAR            : Optional[str]   = Field(None, pattern="^(Y|N)$")
    FLAG_OWN_REALTY         : Optional[str]   = Field(None, pattern="^(Y|N)$")
    OCCUPATION_TYPE         : Optional[str]   = None
    ORGANIZATION_TYPE       : Optional[str]   = None

    class Config:
        json_schema_extra = {
            "example": {
                "EXT_SOURCE_1"         : 0.51,
                "EXT_SOURCE_2"         : 0.62,
                "EXT_SOURCE_3"         : 0.72,
                "DAYS_BIRTH"           : -12000,
                "DAYS_EMPLOYED"        : -2000,
                "AMT_INCOME_TOTAL"     : 180000,
                "AMT_CREDIT"           : 450000,
                "AMT_ANNUITY"          : 22500,
                "AMT_GOODS_PRICE"      : 400000,
                "CODE_GENDER"          : "M",
                "NAME_EDUCATION_TYPE"  : "Higher education",
                "NAME_INCOME_TYPE"     : "Working",
                "REGION_RATING_CLIENT" : 2,
                "FLAG_OWN_CAR"         : "N",
                "FLAG_OWN_REALTY"      : "Y"
            }
        }


class ScoringResult(BaseModel):
    """Output schema for scoring response."""
    lr_probability   : float
    lgbm_probability : float
    score            : int
    risk_band        : str
    action           : str
    interpretation   : str