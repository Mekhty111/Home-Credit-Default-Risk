// Sample applicants in the model's own feature format (POST /score payloads).
const y = (years) => -Math.round(years * 365.25);

export const PRESETS = {
  prime: {
    EXT_SOURCE_1: 0.78, EXT_SOURCE_2: 0.74, EXT_SOURCE_3: 0.8,
    DAYS_BIRTH: y(46), CODE_GENDER: 'F', NAME_EDUCATION_TYPE: 'Higher education', NAME_INCOME_TYPE: 'State servant',
    OCCUPATION_TYPE: 'Accountants', ORGANIZATION_TYPE: 'Government', DAYS_EMPLOYED: y(12),
    AMT_INCOME_TOTAL: 270000, AMT_CREDIT: 450000, AMT_ANNUITY: 21000, AMT_GOODS_PRICE: 405000,
    bureau_active_count: 1, bureau_closed_count: 6, bureau_debt_sum: 120000, bureau_max_overdue: 0,
    prev_refused_count: 0, prev_approval_rate: 1, inst_late_payment_rate: 0, inst_days_late_max: 0,
    DAYS_ID_PUBLISH: y(12), DAYS_REGISTRATION: y(15), DAYS_LAST_PHONE_CHANGE: y(4),
    REGION_RATING_CLIENT_W_CITY: 1, REGION_POPULATION_RELATIVE: 0.035,
    REG_CITY_NOT_WORK_CITY: 0, REG_CITY_NOT_LIVE_CITY: 0, FLAG_EMP_PHONE: 1, FLAG_DOCUMENT_3: 1, OWN_CAR_AGE: 5,
  },
  average: {
    EXT_SOURCE_1: 0.52, EXT_SOURCE_2: 0.56, EXT_SOURCE_3: 0.51,
    DAYS_BIRTH: y(35), CODE_GENDER: 'M', NAME_EDUCATION_TYPE: 'Secondary / secondary special', NAME_INCOME_TYPE: 'Working',
    OCCUPATION_TYPE: 'Laborers', ORGANIZATION_TYPE: 'Business Entity Type 3', DAYS_EMPLOYED: y(3),
    AMT_INCOME_TOTAL: 180000, AMT_CREDIT: 600000, AMT_ANNUITY: 29000, AMT_GOODS_PRICE: 540000,
    bureau_active_count: 2, bureau_closed_count: 3, bureau_debt_sum: 350000, bureau_max_overdue: 0,
    prev_refused_count: 1, prev_approval_rate: 0.7, inst_late_payment_rate: 0.05, inst_days_late_max: 7,
    DAYS_ID_PUBLISH: y(6), DAYS_REGISTRATION: y(10), DAYS_LAST_PHONE_CHANGE: y(2),
    REGION_RATING_CLIENT_W_CITY: 2, REGION_POPULATION_RELATIVE: 0.019,
    REG_CITY_NOT_WORK_CITY: 0, REG_CITY_NOT_LIVE_CITY: 0, FLAG_EMP_PHONE: 1, FLAG_DOCUMENT_3: 1,
  },
  subprime: {
    EXT_SOURCE_1: 0.18, EXT_SOURCE_2: 0.22, EXT_SOURCE_3: 0.15,
    DAYS_BIRTH: y(24), CODE_GENDER: 'M', NAME_EDUCATION_TYPE: 'Lower secondary', NAME_INCOME_TYPE: 'Working',
    OCCUPATION_TYPE: 'Low-skill Laborers', ORGANIZATION_TYPE: 'Self-employed', DAYS_EMPLOYED: y(0.5),
    AMT_INCOME_TOTAL: 112500, AMT_CREDIT: 900000, AMT_ANNUITY: 46000, AMT_GOODS_PRICE: 810000,
    bureau_active_count: 4, bureau_closed_count: 1, bureau_debt_sum: 1200000, bureau_max_overdue: 25000,
    prev_refused_count: 4, prev_approval_rate: 0.3, inst_late_payment_rate: 0.3, inst_days_late_max: 45,
    DAYS_ID_PUBLISH: y(1), DAYS_REGISTRATION: y(1), DAYS_LAST_PHONE_CHANGE: y(0.2),
    REGION_RATING_CLIENT_W_CITY: 3, REGION_POPULATION_RELATIVE: 0.007,
    REG_CITY_NOT_WORK_CITY: 1, REG_CITY_NOT_LIVE_CITY: 1, FLAG_EMP_PHONE: 1, FLAG_DOCUMENT_3: 1,
  },
};
