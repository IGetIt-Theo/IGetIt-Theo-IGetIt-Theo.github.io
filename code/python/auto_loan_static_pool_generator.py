import pandas as pd
import numpy as np
import random
from datetime import date

np.random.seed(42)
random.seed(42)

# ── DATA GENERATION ───────────────────────────────────────────────────────────

CREDIT_BANDS = {
    'Super Prime': {'range': (720, 850), 'base_co_rate': 0.008,  'weight': 0.25},
    'Prime':       {'range': (680, 719), 'base_co_rate': 0.020,  'weight': 0.30},
    'Near Prime':  {'range': (640, 679), 'base_co_rate': 0.055,  'weight': 0.28},
    'Subprime':    {'range': (580, 639), 'base_co_rate': 0.120,  'weight': 0.17},
}
VEHICLE_CO_MULT = {'New': 0.70, 'Used': 1.30}
VEHICLE_VALUES = {
    'New':  {2007: (26500, 5500), 2008: (27000, 5800), 2009: (25500, 5200), 2010: (26800, 5600)},
    'Used': {2007: (13500, 3500), 2008: (13000, 3200), 2009: (12200, 3000), 2010: (13800, 3400)},
}
LOANS_PER_MONTH = {'New': 180, 'Used': 220}
TERM_WEIGHTS = [36, 48, 48, 48, 48, 60]

HAZARD_BY_MONTH = {}
for m in range(1, 61):
    if m <= 3: h = 0.015
    elif m <= 6: h = 0.045
    elif m <= 12: h = 0.085
    elif m <= 18: h = 0.095
    elif m <= 24: h = 0.060
    elif m <= 36: h = 0.020
    else: h = 0.005
    HAZARD_BY_MONTH[m] = h
total_h = sum(HAZARD_BY_MONTH.values())
for m in HAZARD_BY_MONTH: HAZARD_BY_MONTH[m] /= total_h

loans = []
loan_id = 1
for year in range(2007, 2011):
    for month in range(1, 13):
        orig_date = date(year, month, 1)
        quarter = (month - 1) // 3 + 1
        tranche = f"{year}-Q{quarter}"
        for vtype, n_loans in LOANS_PER_MONTH.items():
            val_mu, val_sd = VEHICLE_VALUES[vtype][year]
            band_names = list(CREDIT_BANDS.keys())
            band_weights = [CREDIT_BANDS[b]['weight'] for b in band_names]
            for _ in range(n_loans):
                band = random.choices(band_names, weights=band_weights)[0]
                fico_lo, fico_hi = CREDIT_BANDS[band]['range']
                fico = random.randint(fico_lo, fico_hi)
                principal = max(5000, round(np.random.normal(val_mu, val_sd), -2))
                term = random.choice(TERM_WEIGHTS)
                base_rate = {'Super Prime': 0.058, 'Prime': 0.075, 'Near Prime': 0.095, 'Subprime': 0.135}[band]
                rate = round(base_rate + np.random.normal(0, 0.005), 4)
                rate = max(0.03, min(0.20, rate))
                r = rate / 12
                pmt = principal * (r * (1+r)**term) / ((1+r)**term - 1)
                co_prob = CREDIT_BANDS[band]['base_co_rate'] * VEHICLE_CO_MULT[vtype]
                if year == 2008 and month >= 9: co_prob *= 1.4
                if year == 2009: co_prob *= 1.6
                if year == 2010 and month <= 6: co_prob *= 1.2
                will_default = random.random() < co_prob
                charge_off_month = None
                charge_off_balance = None
                recovery = None
                deficiency_paid = False
                if will_default:
                    valid_months = [m for m in range(1, 61) if m <= term]
                    valid_probs = [HAZARD_BY_MONTH[m] for m in valid_months]
                    norm = sum(valid_probs)
                    valid_probs = [p/norm for p in valid_probs]
                    charge_off_month = random.choices(valid_months, weights=valid_probs)[0]
                    balance = principal
                    for _ in range(charge_off_month - 1):
                        interest = balance * r
                        balance = max(0, balance - (pmt - interest))
                    charge_off_balance = round(balance, 2)
                    # Auction/repo proceeds: 45-60% of charge-off balance for new,
                    # 38-55% for used. Both vary with auction price noise.
                    if vtype == 'New':
                        auction_pct = np.random.uniform(0.45, 0.60)
                    else:
                        auction_pct = np.random.uniform(0.38, 0.55)
                    # Recession years: wholesale auction prices fell ~10-15%
                    if year == 2009:
                        auction_pct *= 0.88
                    elif year == 2008 and month >= 9:
                        auction_pct *= 0.93
                    repo_val = round(charge_off_balance * auction_pct, 2)
                    deficiency = max(0, charge_off_balance - repo_val)
                    # ~20% of members pay some/all of deficiency to protect credit score;
                    # when they do, they pay a negotiated portion (50-100% of deficiency)
                    deficiency_paid = random.random() < 0.20 if deficiency > 0 else False
                    deficiency_recovered = round(deficiency * np.random.uniform(0.50, 1.0), 2) if deficiency_paid else 0
                    recovery = repo_val + deficiency_recovered
                    net_loss = charge_off_balance - recovery
                loans.append({
                    'loan_id': f"AL{loan_id:06d}",
                    'origination_date': orig_date,
                    'year': year,
                    'month': month,
                    'quarter': quarter,
                    'tranche': tranche,
                    'vehicle_type': vtype,
                    'credit_band': band,
                    'fico_score': fico,
                    'original_balance': principal,
                    'term_months': term,
                    'annual_rate': rate,
                    'monthly_payment': round(pmt, 2),
                    'charged_off': will_default,
                    'charge_off_month': charge_off_month if will_default else None,
                    'charge_off_balance': charge_off_balance if will_default else None,
                    'recovery': round(recovery, 2) if will_default else None,
                    'deficiency_paid': deficiency_paid if will_default else False,
                    'net_loss': round(charge_off_balance - recovery, 2) if will_default else 0,
                })
                loan_id += 1

df = pd.DataFrame(loans)

# ── OBSERVATION CUTOFF ───────────────────────────────────────────────────────
# Simulates a reporting snapshot date. Tranches originated close to the cutoff
# will have incomplete curves, mirroring real static pool analyses where later
# vintages are still maturing. The visualization relies on this — the eye
# extrapolates the incomplete curves against the fully-seasoned earlier tranches.
CUTOFF_DATE = date(2011, 6, 30)

# Static pool
pool_rows = []
tranches = sorted(df['tranche'].unique())
vtypes = ['New', 'Used']
bands = list(CREDIT_BANDS.keys())

for tranche in tranches:
    for vtype in vtypes:
        for band in bands:
            mask = (df['tranche'] == tranche) & (df['vehicle_type'] == vtype) & (df['credit_band'] == band)
            subset = df[mask]
            n_loans = len(subset)
            if n_loans == 0: continue
            total_orig_balance = subset['original_balance'].sum()
            defaulters = subset[subset['charged_off'] == True]

            # Max observable MOB for this tranche = months from first origination date to cutoff
            tranche_orig_date = subset['origination_date'].min()
            max_observable_mob = (
                (CUTOFF_DATE.year - tranche_orig_date.year) * 12
                + (CUTOFF_DATE.month - tranche_orig_date.month)
            )
            max_observable_mob = max(1, min(max_observable_mob, 60))

            for age_month in range(1, max_observable_mob + 1):
                cum_co_count = int((defaulters['charge_off_month'] <= age_month).sum())
                cum_co_balance = float(defaulters.loc[defaulters['charge_off_month'] <= age_month, 'charge_off_balance'].sum())
                cum_net_loss = float(defaulters.loc[defaulters['charge_off_month'] <= age_month, 'net_loss'].sum())
                pool_rows.append({
                    'tranche': tranche,
                    'vehicle_type': vtype,
                    'credit_band': band,
                    'loan_age_months': age_month,
                    'n_loans_in_tranche': n_loans,
                    'total_orig_balance': round(total_orig_balance, 0),
                    'cum_chargeoff_count': cum_co_count,
                    'cum_chargeoff_balance': round(cum_co_balance, 0),
                    'cum_net_loss': round(cum_net_loss, 0),
                    'cum_co_rate_count': round(cum_co_count / n_loans, 6) if n_loans > 0 else 0,
                    'cum_co_rate_balance': round(cum_co_balance / total_orig_balance, 6) if total_orig_balance > 0 else 0,
                })

df_pool = pd.DataFrame(pool_rows)

df.to_csv('auto_loan_detail.csv', index=False)
df_pool.to_csv('auto_loan_static_pool.csv', index=False)
