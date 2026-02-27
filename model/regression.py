import pandas as pd
import matplotlib
matplotlib.use('Agg')  # REQUIRED for Flask
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import os

def process_csv(file):

    # =========================
    # 1️⃣ LOAD DATA
    # =========================
    df = pd.read_csv(file)

    if 'Date' not in df.columns or 'Price' not in df.columns:
        raise Exception("CSV must have columns: Date and Price")

    initial_info = {
        "jumlah_baris": df.shape[0],
        "jumlah_kolom": df.shape[1],
        "kolom": list(df.columns)
    }

    initial_preview = df.head(10)

    # =========================
    # 2️⃣ DATA CLEANING
    # =========================
    df['Price'] = df['Price'].astype(str).str.replace(',', '', regex=False)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna(subset=['Price'])

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    df = df.sort_values('Date').reset_index(drop=True)

    after_clean_info = {
        "jumlah_baris": df.shape[0]
    }

    # =========================
    # 📊 DESCRIPTIVE STATISTICS
    # =========================
    desc_stats = df['Price'].describe().to_dict()
    statistics = {
        'count': int(desc_stats['count']),
        'mean': round(desc_stats['mean'], 2),
        'std': round(desc_stats['std'], 2),
        'min': round(desc_stats['min'], 2),
        '25%': round(desc_stats['25%'], 2),
        '50%': round(desc_stats['50%'], 2),
        '75%': round(desc_stats['75%'], 2),
        'max': round(desc_stats['max'], 2)
    }

    # =========================
    # 3️⃣ TIME VARIABLE (Days)
    # =========================
    start_date = df['Date'].min()
    df['Days'] = (df['Date'] - start_date).dt.days

    regression_df = df[['Date', 'Days', 'Price']]
    regression_preview = regression_df.head(10)

    # =========================
    # 4️⃣ STATIC FOLDER
    # =========================
    os.makedirs("static", exist_ok=True)

    # =========================
    # 5️⃣ DAILY VISUALIZATION (AUTO RANGE)
    # =========================
    plt.figure(figsize=(12, 5))
    plt.plot(df['Date'], df['Price'], linewidth=1)
    
    # Automatically set x-axis limits based on data
    plt.xlim(df['Date'].min(), df['Date'].max())
    
    # Create year ticks every 2 years based on actual data range
    start_year = df['Date'].min().year
    end_year = df['Date'].max().year
    year_ticks = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='2YS')
    plt.xticks(year_ticks, [t.year for t in year_ticks], rotation=45)
    
    plt.title("Daily Gold Price Movement")
    plt.xlabel("Year")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("static/daily.png")
    plt.close()

    # =========================
    # 6️⃣ MONTHLY VISUALIZATION (AUTO RANGE)
    # =========================
    monthly_df = df.resample('ME', on='Date')['Price'].mean()

    plt.figure(figsize=(12, 5))
    plt.plot(monthly_df.index, monthly_df.values, linewidth=1.5)
    plt.xlim(monthly_df.index.min(), monthly_df.index.max())
    
    # Create year ticks every 2 years based on actual data range
    start_year = monthly_df.index.min().year
    end_year = monthly_df.index.max().year
    year_ticks = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='2YS')
    plt.xticks(year_ticks, [t.year for t in year_ticks], rotation=45)
    
    plt.title("Monthly Gold Price Movement (Average)")
    plt.xlabel("Year")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("static/monthly.png")
    plt.close()

    # =========================
    # 7️⃣ YEARLY VISUALIZATION (AUTO RANGE)
    # =========================
    yearly_df = df.resample('YE', on='Date')['Price'].mean()

    plt.figure(figsize=(12, 5))
    plt.plot(
        yearly_df.index,
        yearly_df.values,
        marker='o',
        linestyle='-',
        linewidth=2,
        markersize=6
    )
    plt.xlim(yearly_df.index.min(), yearly_df.index.max())
    
    # Create year ticks every 2 years based on actual data range
    start_year = yearly_df.index.min().year
    end_year = yearly_df.index.max().year
    year_ticks = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='2YS')
    plt.xticks(year_ticks, [t.year for t in year_ticks], rotation=45)
    
    plt.title("Yearly Gold Price Movement (Average)")
    plt.xlabel("Year")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("static/yearly.png")
    plt.close()

    # =========================
    # 8️⃣ MODEL WITHOUT SPLIT
    # =========================
    X_full = df[['Days']]
    y_full = df['Price']

    full_model = LinearRegression()
    full_model.fit(X_full, y_full)
    
    a_full = full_model.coef_[0]
    b_full = full_model.intercept_

    y_full_pred = full_model.predict(X_full)

    mse_full = mean_squared_error(y_full, y_full_pred)
    r2_full = r2_score(y_full, y_full_pred)

    # =========================
    # 8️⃣.1️⃣ VISUALIZATION WITHOUT SPLIT (AUTO RANGE)
    # =========================
    plt.figure(figsize=(12, 5))

    plt.scatter(df['Date'], y_full, s=8, alpha=0.5, label="Actual Data", color='blue')
    plt.plot(df['Date'], y_full_pred, color='red', linewidth=2, label="Regression (Without Split)")
    plt.xlim(df['Date'].min(), df['Date'].max())
    
    # Create year ticks every 2 years based on actual data range
    start_year = df['Date'].min().year
    end_year = df['Date'].max().year
    year_ticks = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='2YS')
    plt.xticks(year_ticks, [t.year for t in year_ticks], rotation=45)

    plt.title("Linear Regression Model - Without Data Split (In-Sample)")
    plt.xlabel("Year")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("static/regression_no_split.png")
    plt.close()

    # =========================
    # 9️⃣ 80:20 DATA SPLIT
    # =========================
    split_index = int(len(df) * 0.8)

    X_train = X_full.iloc[:split_index]
    X_test = X_full.iloc[split_index:]

    y_train = y_full.iloc[:split_index]
    y_test = y_full.iloc[split_index:]

    # =========================
    # 🔟 TRAIN MODEL (SPLIT)
    # =========================
    model = LinearRegression()
    model.fit(X_train, y_train)

    a = model.coef_[0]
    b = model.intercept_

    y_test_pred = model.predict(X_test)
    y_pred_all = model.predict(X_full)

    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    # =========================
    # 🔟.1️⃣ VISUALIZATION WITH SPLIT (AUTO RANGE)
    # =========================
    plt.figure(figsize=(12, 5))

    plt.scatter(df['Date'][:split_index], y_train, s=8, label="Train Data (80%)", color='blue', alpha=0.6)
    plt.scatter(df['Date'][split_index:], y_test, s=8, label="Test Data (20%)", color='orange', alpha=0.6)
    plt.plot(df['Date'], y_pred_all, color='red', linewidth=2, label="Regression (80:20 Split)")
    plt.xlim(df['Date'].min(), df['Date'].max())
    
    # Create year ticks every 2 years based on actual data range
    start_year = df['Date'].min().year
    end_year = df['Date'].max().year
    year_ticks = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='2YS')
    plt.xticks(year_ticks, [t.year for t in year_ticks], rotation=45)

    plt.title("Linear Regression Model - 80:20 Data Split (Out-of-Sample)")
    plt.xlabel("Year")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("static/regression_split.png")
    plt.close()

    # =========================
    # 1️⃣1️⃣ PREDICTION 2026–2035
    # =========================
    prediction_years = pd.date_range(
        start='2026-01-01',
        end='2035-01-01',
        freq='YS'
    )

    pred_df = pd.DataFrame({'Date': prediction_years})
    pred_df['Days'] = (pred_df['Date'] - start_date).dt.days
    pred_df['Predicted_Price'] = model.predict(pred_df[['Days']])

    # =========================
    # 1️⃣2️⃣ REGRESSION GRAPH WITH PREDICTION (FULL RANGE)
    # =========================
    plt.figure(figsize=(14, 6))

    plt.scatter(df['Date'][:split_index], y_train, s=8, label="Train", alpha=0.6)
    plt.scatter(df['Date'][split_index:], y_test, s=8, label="Test", color='orange', alpha=0.6)
    plt.plot(df['Date'], y_pred_all, color='red', linewidth=2, label="Linear Regression")

    plt.plot(
        pred_df['Date'],
        pred_df['Predicted_Price'],
        color='green',
        linestyle='--',
        marker='o',
        markersize=8,
        linewidth=2,
        label="Prediction 2026–2035"
    )

    # Set x-axis limits from earliest data to 2035
    plt.xlim(df['Date'].min(), pd.Timestamp('2035-12-31'))
    
    # Create year ticks every 5 years from data start to 2035
    start_year = df['Date'].min().year
    year_ticks = pd.date_range(start=f'{start_year}-01-01', end='2035-12-31', freq='5YS')
    plt.xticks(year_ticks, [t.year for t in year_ticks], rotation=45)

    plt.title("Linear Regression and Gold Price Prediction 2026–2035")
    plt.xlabel("Year")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("static/regression.png")
    plt.close()

    # =========================
    # 1️⃣3️⃣ RETURN TO FLASK
    # =========================
    return (
        a, b, a_full, b_full,
        mse, r2, mse_full, r2_full,
        initial_preview,
        regression_preview,
        initial_info,
        after_clean_info,
        pred_df,
        statistics
    )