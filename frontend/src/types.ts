export type Store = {
  store_id: string;
  store_name: string;
};

export type StoreStatus = {
  store_id: string;
  current_date: string;
  last_uploaded_date: string | null;
  gap_days: number | null;
  ready_to_forecast: boolean;
  days_uploaded: number;
};

export type StoreProfilePatch = {
  customers_per_staff?: number;
  country?: string;
  holiday_subdivision?: string;
};

export type StoreProfileResponse = {
  store_id: string;
  profile: {
    customers_per_staff?: number;
    country?: string;
    holiday_subdivision?: string;
  };
};

export type ForecastDayOverride = {
  horizon: number;
  open?: 0 | 1;
  promo?: 0 | 1;
};

export type ForecastExplanationFactor = {
  feature: string;
  label: string;
  shap_value: number;
  abs_shap_value: number;
};

export type ForecastExplanation = {
  base_log1p: number;
  predicted_log1p: number;
  top_positive_factors: ForecastExplanationFactor[];
  top_negative_factors: ForecastExplanationFactor[];
};

export type UploadResponse = {
  store_id: string;
  rows_written?: number;
};

export type ForecastResponse = {
  store_id: string;
  horizon: number;
  issue_date: string;
  target_date: string;
  weekday?: string;
  planned_open?: 0 | 1;
  planned_promo?: 0 | 1;
  state_holiday?: string;
  school_holiday?: 0 | 1;
  public_holiday_name?: string | null;
  school_holiday_name?: string | null;
  prediction_customers: number;
  customers_per_staff?: number | null;
  suggested_staff?: number | null;
  explanation?: ForecastExplanation | null;
  source_model_h?: 1 | 7 | 14;
  source_issue_date?: string;
};

export type ForecastRangeRequest = {
  k: number;
  day_overrides?: ForecastDayOverride[];
  include_explanations?: boolean;
};

export type ForecastRangeResponse = {
  store_id: string;
  issue_date: string;
  k: number;
  cache_hit?: boolean;
  scenario_hash?: string;
  day_overrides?: ForecastDayOverride[];
  forecasts: ForecastResponse[];
};

export type StoreMeta = {
  Store?: number;
  StoreType?: string;
  Assortment?: string;
  CompetitionDistance?: number;
  Promo2?: number;
  PromoInterval?: string;
  CompetitionOpenSinceMonth?: number;
  CompetitionOpenSinceYear?: number;
  Promo2SinceWeek?: number;
  Promo2SinceYear?: number;
};

export type StoreMetaResponse = {
  store_id: string;
  store_meta: StoreMeta;
};

export type StoreMetaPatch = Partial<StoreMeta>;

export type CatchUpDayIn = {
  date: string;       // YYYY-MM-DD
  customers: number;
  open: 0 | 1;
  promo: 0 | 1;
};

export type CatchUpResponse = {
  store_id: string;
  rows_written: number;
};
