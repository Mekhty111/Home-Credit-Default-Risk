from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Union


class ApplicantInput(BaseModel):
    """Input schema for a single loan applicant.

    The fields below are the ones exposed in the web form. Any other variable
    known to the binning process (e.g. bureau / installment aggregates such as
    `bureau_active_count`) can be passed as an extra field.
    """

    model_config = ConfigDict(
        extra='allow',
        json_schema_extra={
            "example": {
                "EXT_SOURCE_1"           : 0.51,
                "EXT_SOURCE_2"           : 0.62,
                "EXT_SOURCE_3"           : 0.72,
                "DAYS_BIRTH"             : -12000,
                "DAYS_EMPLOYED"          : -2000,
                "AMT_CREDIT"             : 450000,
                "AMT_ANNUITY"            : 22500,
                "AMT_GOODS_PRICE"        : 400000,
                "CODE_GENDER"            : "M",
                "NAME_EDUCATION_TYPE"    : "Higher education",
                "NAME_INCOME_TYPE"       : "Working",
                "bureau_active_count"    : 1,
                "inst_late_payment_rate" : 0.02,
            }
        },
    )

    # External credit bureau scores
    EXT_SOURCE_1                : Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2                : Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3                : Optional[float] = Field(None, ge=0, le=1)

    # Applicant
    DAYS_BIRTH                  : Optional[int]   = Field(None, le=0, description="Negative: days before application")
    DAYS_EMPLOYED               : Optional[int]   = Field(None, description="Negative: days before application; 365243 = not employed")
    CODE_GENDER                 : Optional[str]   = Field(None, pattern="^(M|F|XNA)$")
    NAME_EDUCATION_TYPE         : Optional[str]   = None
    NAME_INCOME_TYPE            : Optional[str]   = None
    OCCUPATION_TYPE             : Optional[str]   = None
    ORGANIZATION_TYPE           : Optional[str]   = None

    # Loan
    AMT_INCOME_TOTAL            : Optional[float] = Field(None, ge=0, description="Not a model feature; used for affordability ratios")
    AMT_CREDIT                  : Optional[float] = Field(None, ge=0)
    AMT_ANNUITY                 : Optional[float] = Field(None, ge=0)
    AMT_GOODS_PRICE             : Optional[float] = Field(None, ge=0)

    # Documents & region
    DAYS_ID_PUBLISH             : Optional[int]   = Field(None, le=0)
    DAYS_REGISTRATION           : Optional[float] = Field(None, le=0)
    DAYS_LAST_PHONE_CHANGE      : Optional[float] = Field(None, le=0)
    REGION_RATING_CLIENT_W_CITY : Optional[int]   = Field(None, ge=1, le=3)
    REGION_POPULATION_RELATIVE  : Optional[float] = Field(None, ge=0)
    REG_CITY_NOT_WORK_CITY      : Optional[int]   = Field(None, ge=0, le=1)
    REG_CITY_NOT_LIVE_CITY      : Optional[int]   = Field(None, ge=0, le=1)
    FLAG_EMP_PHONE              : Optional[int]   = Field(None, ge=0, le=1)
    FLAG_DOCUMENT_3             : Optional[int]   = Field(None, ge=0, le=1)
    OWN_CAR_AGE                 : Optional[float] = Field(None, ge=0)


class Contribution(BaseModel):
    """Contribution of one feature to the scorecard result."""
    feature  : str
    value    : Optional[Union[float, int, str]] = None
    provided : bool
    woe      : float
    impact   : float = Field(description="Points vs. a neutral applicant; positive lowers risk")


class ScoringResult(BaseModel):
    """Output schema for scoring response."""
    lr_probability   : float
    lgbm_probability : float
    score            : int
    risk_band        : str
    action           : str
    interpretation   : str
    contributions    : List[Contribution] = []
