import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------- Load the saved models ----------
@st.cache_resource
def load_models():
    return joblib.load('sales_model.sav')

art = load_models()
lin_reg = art['lin_reg']
log_reg = art['log_reg']
scaler = art['scaler']
FEATURE_COLUMNS = art['feature_columns']
store_info = art['store_info']

# ---------- Same feature steps as in Colab ----------
MONTH_NAMES = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
NUM_FEATURES = ['Promo', 'SchoolHoliday', 'Promo2', 'IsPromo2Month', 'LogCompDist',
                'CompOpenMonths', 'Day', 'LogStoreAvgSales']
CAT_FEATURES = ['DayOfWeek', 'Month', 'StateHoliday', 'StoreType', 'Assortment']


def build_features(d):
    d = d.copy()
    d['Date'] = pd.to_datetime(d['Date'])
    d['Year'] = d['Date'].dt.year
    d['Month'] = d['Date'].dt.month
    d['Day'] = d['Date'].dt.day
    d['DayOfWeek'] = d['Date'].dt.dayofweek + 1
    d['WeekOfYear'] = d['Date'].dt.isocalendar().week.astype(int)
    d['LogCompDist'] = np.log1p(d['CompetitionDistance'])
    comp_months = (12 * (d['Year'] - d['CompetitionOpenSinceYear'])
                   + (d['Month'] - d['CompetitionOpenSinceMonth']))
    d['CompOpenMonths'] = comp_months.fillna(0).clip(lower=0)
    month_str = d['Month'].map(MONTH_NAMES)
    in_interval = [isinstance(p, str) and m in p.split(',')
                   for p, m in zip(d['PromoInterval'], month_str)]
    promo2_started = (12 * (d['Year'] - d['Promo2SinceYear'])
                      + (d['WeekOfYear'] - d['Promo2SinceWeek']) / 4).fillna(-1) >= 0
    d['IsPromo2Month'] = (np.array(in_interval) & promo2_started).astype(int)
    return d


def make_X(d, columns):
    X = d[NUM_FEATURES + CAT_FEATURES].copy()
    X[CAT_FEATURES] = X[CAT_FEATURES].astype(str)
    X = pd.get_dummies(X, columns=CAT_FEATURES, drop_first=True, dtype=int)
    return X.reindex(columns=columns, fill_value=0)


def predict_store_day(store_id, date, promo, state_holiday, school_holiday):
    d = store_info[store_info['Store'] == int(store_id)].copy()
    d['Date'] = pd.to_datetime(date)
    d['Promo'] = int(promo)
    d['StateHoliday'] = str(state_holiday)
    d['SchoolHoliday'] = int(school_holiday)
    d = build_features(d)
    d['LogStoreAvgSales'] = np.log(d['StoreAvgSales'])
    X = scaler.transform(make_X(d, FEATURE_COLUMNS))
    sales = float(np.expm1(lin_reg.predict(X))[0])
    prob = float(log_reg.predict_proba(X)[0, 1])
    return sales, prob, float(d['StoreMedianSales'].iloc[0])


# ---------- User interface ----------
st.set_page_config(page_title='ABC Ltd - Store Sales Predictor', page_icon='📈')
st.title('📈 Store Sales Predictor')
st.write('Predicts daily sales (Linear Regression) and the chance of an '
         'above-normal sales day (Logistic Regression) for a store.')

HOLIDAY_MAP = {'None': '0', 'Public holiday': 'a', 'Easter': 'b', 'Christmas': 'c'}

col1, col2 = st.columns(2)
with col1:
    store_id = st.number_input('Store ID', min_value=1,
                               max_value=int(store_info['Store'].max()), value=1, step=1)
    date = st.date_input('Date', value=pd.Timestamp('2015-08-03'))
    promo = st.radio('Promo running?', ['No', 'Yes'], horizontal=True)
with col2:
    holiday = st.selectbox('State holiday', list(HOLIDAY_MAP.keys()))
    school = st.radio('School holiday?', ['No', 'Yes'], horizontal=True)

if st.button('Predict', type='primary'):
    sales, prob, med = predict_store_day(store_id, date, promo == 'Yes',
                                         HOLIDAY_MAP[holiday], school == 'Yes')
    c1, c2 = st.columns(2)
    c1.metric('Predicted Sales', f'{sales:,.0f}',
              delta=f'{sales - med:,.0f} vs store normal')
    c2.metric('Chance of High-Sales Day', f'{prob:.1%}')
    if prob >= 0.5:
        st.success('🔥 HIGH-SALES DAY — plan extra staff and stock.')
    else:
        st.info('📉 Normal / low-sales day — standard staffing is enough.')
    st.caption(f"This store's normal (median) daily sales: {med:,.0f}")
