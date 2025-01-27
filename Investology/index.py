from flask import Flask, render_template, session, redirect, url_for, request
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg

from prediction import fetch_and_preprocess, create_sequences, build_model, predict_stock_prices

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session handling


def human_readable_large_number(value):
    try:
        value = float(value) / 1e7
        return f'{value:.2f} Cr'
    except (ValueError, TypeError):
        return value
    
def get_mutual_fund_holders(ticker):
    try:
        # Try to fetch mutual fund holders
        name = yf.Ticker(ticker)
        mutual_fund_holders = name.mutualfund_holders  # This might raise an error
    except yf.exceptions.YFinanceDataException:
        # If there's an issue fetching the data, assign None
        mutual_fund_holders = None
    except AttributeError:
        # If the attribute is not found
        mutual_fund_holders = None

    return mutual_fund_holders

# Register the filter with Jinja2
app.jinja_env.filters['human_readable_large_number'] = human_readable_large_number

@app.route('/navigation', methods=['GET', 'POST'])
def navigation():
    if request.method == 'POST':
        task = request.form.get('task')
        if task == 'home':
            # Redirect to the home page
            return redirect(url_for('search_page'))
        elif task == 'overview':
            # Redirect to the overview page
            return redirect(url_for('overview'))
        elif task == 'financials':
            # Redirect to the financials page
            return redirect(url_for('financials'))
        elif task == 'fundamentals':
            # Redirect to the fundamentals page
            return redirect(url_for('fundamentals'))
        elif task == 'projections':
            # Redirect to the projections page
            return redirect(url_for('projections'))

    return render_template('navigation.html')


# Route for the search page
@app.route('/', methods=['GET', 'POST'])
def search_page():
    csv_file = 'static/CSV/companies.csv'  # Path to your CSV file
    # Load CSV options into a list of dictionaries
    df = pd.read_csv(csv_file)
    options = [{'value': row['SYMBOL'], 'text': row['NAME OF COMPANY']} for _, row in df.iterrows()]

    if request.method == 'POST':
        c_name = request.form.get('c_name')
        if not c_name:  # Handle case when no company is selected
            alert_message = "Please select a company."
            return render_template('searchPage.html', options=options, alert_message=alert_message)
        # Store selected company in session and redirect to the overview page
        session['cname'] = c_name
        return redirect(url_for('overview'))
    
    return render_template('searchPage.html', options=options)

# Route for the overview page
@app.route('/overview', methods=['GET'])
def overview():
    c_name = session.get('cname', None)
    if not c_name:
        return redirect(url_for('search_page'))  # Redirect to search if no company is selected

    c_name_ns = f"{c_name}.NS"
    name = yf.Ticker(c_name_ns)
    
    mutual_fund_holders = get_mutual_fund_holders(c_name_ns)

    # Generate pie chart
    if mutual_fund_holders is not None:
        fund_names = mutual_fund_holders['Holder']
        fund_percentages = mutual_fund_holders['pctHeld']
        if fund_names is not None and fund_percentages is not None:
            pie = go.Figure(data=[go.Pie(labels=fund_names, values=fund_percentages)])
            pie.update_layout(title=f'Mutual Fund Holders Distribution for {c_name}', width=1200)
            pie_chart = pie.to_html(full_html=False)
        else:
            pie = go.Figure(data=[go.Pie(labels=['No Data'], values=[100])])
            pie.update_layout(title=f'No Data of Mutual Fund Holders Distribution for {c_name}', width=1200)
            pie_chart = pie.to_html(full_html=False)
    else:
        pie = go.Figure(data=[go.Pie(labels=['No Data'], values=[100])])
        pie.update_layout(title=f'No Data of Mutual Fund Holders Distribution for {c_name}', width=1200)
        pie_chart = pie.to_html(full_html=False)

    # Prepare context data
    context = {
        'summary': name.info.get('longBusinessSummary', 'No summary available'),
        'c_name': c_name,
        'Total_Employees': name.info.get('fullTimeEmployees', 'N/A'),
        'industry': name.info.get('industry', 'N/A'),
        'sector': name.info.get('sector', 'N/A'),
        'website': name.info.get('website', 'N/A'),
        'pie_chart': pie_chart,
    }

    return render_template('overview.html', **context)

@app.route('/home')
def home():
    # Home page logic here
    return "Home Page"

@app.route('/financials', methods=['GET'])
def financials():
    # Retrieve the company name from session
    cname = session.get('cname')
    if not cname:
        return redirect(url_for('search_page'))  # Redirect to search page if cname is not in session

    # Append '.NS' for NSE ticker
    cname += ".NS"
    stock = yf.Ticker(cname)

    # Financials and Balance Sheet
    balance_sheet = stock.balance_sheet
    financials = stock.financials

    # Gross Profit Calculation
    if not financials.empty:
        total_revenue = financials.loc['Total Revenue'][0] if 'Total Revenue' in financials.index else None
        cost_of_revenue = financials.loc['Cost Of Revenue'][0] if 'Cost Of Revenue' in financials.index else None
        gross_profit = total_revenue - cost_of_revenue if all([total_revenue, cost_of_revenue]) else 'NA'
    else:
        gross_profit = 'NA'

    # EBITDA Calculation
    if not financials.empty:
        operating_income = financials.loc['Operating Income'][0] if 'Operating Income' in financials.index else None
        depreciation_and_amortization = financials.loc['Depreciation & Amortization'][0] if 'Depreciation & Amortization' in financials.index else None
        ebitda = operating_income + depreciation_and_amortization if all([operating_income, depreciation_and_amortization]) else 'NA'
    else:
        ebitda = 'NA'

    # Plotting Total Revenue for Financial Years
    income_stmt = stock.financials
    if 'Total Revenue' in income_stmt.index:
        total_revenue = income_stmt.loc['Total Revenue']
        df_total_rev = total_revenue.reset_index()
        df_total_rev.columns = ['Date', 'Total Revenue']
        df_total_rev['Date'] = pd.to_datetime(df_total_rev['Date'])
        df_total_rev = df_total_rev.sort_values(by='Date').tail(4)
        df_total_rev['Total Revenue (INR Crores)'] = df_total_rev['Total Revenue'] / 1e7
        df_total_rev['Financial Year'] = (df_total_rev['Date'].dt.year - 1).astype(str) + '-' + df_total_rev['Date'].dt.year.astype(str).str[2:]
    else:
        df_total_rev = pd.DataFrame()

    if 'Net Income' in income_stmt.index:
        net_income = income_stmt.loc['Net Income']
        df_net_inc = net_income.reset_index()
        df_net_inc.columns = ['Date', 'Net Income']
        df_net_inc['Date'] = pd.to_datetime(df_net_inc['Date'])
        df_net_inc = df_net_inc.sort_values(by='Date').tail(4)
        df_net_inc['Net Income (INR Crores)'] = df_net_inc['Net Income'] / 1e7
        df_net_inc['Financial Year'] = (df_net_inc['Date'].dt.year - 1).astype(str) + '-' + df_net_inc['Date'].dt.year.astype(str).str[2:]
    else:
        df_net_inc = pd.DataFrame()

    # Plot Total Revenue and Net Income
    plots = make_subplots(rows=1, cols=2, shared_yaxes=True,
                          subplot_titles=[f'Total Revenue ({cname})', f'Net Income ({cname})'])
    if not df_total_rev.empty:
        plots.add_trace(go.Bar(x=df_total_rev['Financial Year'], y=df_total_rev['Total Revenue (INR Crores)'],
                               marker_color='#2ca02c', name='Total Revenue (INR Crores)'), row=1, col=1)
    if not df_net_inc.empty:
        plots.add_trace(go.Bar(x=df_net_inc['Financial Year'], y=df_net_inc['Net Income (INR Crores)'],
                               marker_color='skyblue', name='Net Income (INR Crores)'), row=1, col=2)
    plots.update_layout(
        title=f'{cname} Financial Performance',
        height=600,
        legend_title='Legend',
        title_x=0.5,
        showlegend=True,
    )
    plots.update_xaxes(title_text='Financial Year', row=1, col=1)
    plots.update_xaxes(title_text='Financial Year', row=1, col=2)
    plots.update_yaxes(title_text='Amount (INR Crores)', row=1, col=1)
    year_revenue = plots.to_html(full_html=False)

    # Quarterly Revenue
    quat_income_stmt = stock.quarterly_financials
    if 'Total Revenue' in quat_income_stmt.index:
        quat_total_revenue = quat_income_stmt.loc['Total Revenue']
        df_quat_rev = quat_total_revenue.reset_index()
        df_quat_rev = df_quat_rev.head(5)
        df_quat_rev.columns = ['Date', 'Total Revenue']
        df_quat_rev['Date'] = pd.to_datetime(df_quat_rev['Date'])
        df_quat_rev = df_quat_rev.sort_values(by='Date')
        df_quat_rev['Total Revenue (INR Crores)'] = df_quat_rev['Total Revenue'] / 1e7
        df_quat_rev['Quarter'] = df_quat_rev['Date'].dt.to_period('Q')
    else:
        df_quat_rev = pd.DataFrame()

    if 'Net Income' in quat_income_stmt.index:
        quat_net_income = quat_income_stmt.loc['Net Income']
        df_quat_inc = quat_net_income.reset_index()
        df_quat_inc = df_quat_inc.head(5)
        df_quat_inc.columns = ['Date', 'Net Income']
        df_quat_inc['Date'] = pd.to_datetime(df_quat_inc['Date'])
        df_quat_inc = df_quat_inc.sort_values(by='Date')
        df_quat_inc['Net Income (INR Crores)'] = df_quat_inc['Net Income'] / 1e7
        df_quat_inc['Quarter'] = df_quat_inc['Date'].dt.to_period('Q')
    else:
        df_quat_inc = pd.DataFrame()

    # Quarterly Plot
    plots1 = make_subplots(rows=1, cols=2, shared_yaxes=True,
                           subplot_titles=[f'Total Revenue ({cname})', f'Net Income ({cname})'])
    if not df_quat_rev.empty:
        plots1.add_trace(go.Bar(x=df_quat_rev['Quarter'].astype(str), y=df_quat_rev['Total Revenue (INR Crores)'],
                                marker_color='#2ca02c', name='Total Revenue (INR Crores)'), row=1, col=1)
    if not df_quat_inc.empty:
        plots1.add_trace(go.Bar(x=df_quat_inc['Quarter'].astype(str), y=df_quat_inc['Net Income (INR Crores)'],
                                marker_color='skyblue', name='Net Income (INR Crores)'), row=1, col=2)
    plots1.update_layout(
        title=f'{cname} Quarterly Financial Performance',
        height=600,
        legend_title='Legend',
        title_x=0.5,
        showlegend=True,
    )
    plots1.update_xaxes(title_text='Quarter', row=1, col=1)
    plots1.update_xaxes(title_text='Quarter', row=1, col=2)
    plots1.update_yaxes(title_text='Amount (INR Crores)', row=1, col=1)
    quat_plot = plots1.to_html(full_html=False)

    # Prepare data for rendering the template
    context = {
        'company_name': cname[:-3],  # Remove '.NS' for display
        'website': stock.info.get('website', 'N/A'),
        'Revenue': financials.loc['Total Revenue'][0] if 'Total Revenue' in financials.index else 'NA',
        'Net_Income': financials.loc['Net Income'][0] if 'Net Income' in financials.index else 'NA',
        'Gross_Profit': gross_profit,
        'year_revenue': year_revenue,
        'quat_plot': quat_plot,
    }

    return render_template('financials.html', **context)


@app.route('/fundamentals', methods=['GET'])
def fundamentals():
    # Get company name (cname) from session
    cname = session.get('cname')
    if not cname:
        return redirect(url_for('search_page'))  # Redirect to the search page if cname is missing

    cname += ".NS"  # Add the NSE ticker suffix
    stock = yf.Ticker(cname)

    # Fetch stock information
    info = stock.info
    balance_sheet = stock.balance_sheet
    financials = stock.financials

    # Earnings Per Share (EPS)
    if not financials.empty:
        net_income = financials.loc['Net Income'][0] if 'Net Income' in financials.index else None
        shares_outstanding = info.get('sharesOutstanding')
        earning_per_share = (net_income / shares_outstanding) if all([net_income, shares_outstanding]) else 0
    else:
        earning_per_share = 0

    # Debt to Equity Ratio
    if not financials.empty and not balance_sheet.empty:
        total_debt = info.get('totalDebt')
        total_equity = balance_sheet.loc['Stockholders Equity'][0] if 'Stockholders Equity' in balance_sheet.index else None
        debt_to_equity = (total_debt / total_equity) if all([total_debt, total_equity]) else 0
    else:
        debt_to_equity = 0

    # Profit Margin
    if not financials.empty:
        net_income = financials.loc['Net Income'][0] if 'Net Income' in financials.index else None
        total_revenue = financials.loc['Total Revenue'][0] if 'Total Revenue' in financials.index else None
        profit_margin = (net_income / total_revenue) * 100 if all([net_income, total_revenue]) else 0
    else:
        profit_margin = 0

    # Operating Margin
    if not financials.empty:
        operating_income = financials.loc['Operating Income'][0] if 'Operating Income' in financials.index else None
        total_revenue = financials.loc['Total Revenue'][0] if 'Total Revenue' in financials.index else None
        operating_margin = (operating_income / total_revenue) * 100 if all([operating_income, total_revenue]) else 0
    else:
        operating_margin = 0

    # Context for rendering the template
    context = {
        'company_name': cname[:-3],  # Remove ".NS" for display
        'website': info.get('website', 'N/A'),
        'market_cap': info.get('marketCap'),
        'forward_pe': info.get('forwardPE'),
        'price_sales': info.get('priceToSalesTrailing12Months'),
        'price_book': info.get('priceToBook'),
        'profit_margin': profit_margin or info.get('profitMargins', 0) * 100,
        'operating_margin': operating_margin or info.get('operatingMargins', 0) * 100,
        'return_on_equity': info.get('returnOnEquity', 0) * 100,
        'return_on_assets': info.get('returnOnAssets', 0) * 100,
        'total_debt': info.get('totalDebt'),
        'debtToEquity': debt_to_equity,
        'EPS': earning_per_share,
        'Dividend_Yield': info.get('dividendYield', 0) * 100,
        'P_FCF': info.get('priceToFreeCashFlows'),
        'EV_EBITDA': info.get('enterpriseToEbitda'),
        'EV_Revenue': info.get('enterpriseToRevenue'),
    }

    return render_template('fundamentals.html', **context)


@app.route('/projections', methods=['GET'])
def projections():
    # Get the company name from the session
    cname = session.get('cname')
    if not cname:
        return redirect(url_for('search_page'))  # Redirect to search page if cname is missing

    cname += ".NS"
    stock = yf.Ticker(cname)
    info = stock.info

    # Calculate the date 30 days ago from today
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=60)

    # Fetch historical data
    data = yf.download(cname, start=start_date, end=end_date)
    if data.empty:
        return "Error: Empty DataFrame returned by yfinance."

    # Ensure the index is a DatetimeIndex
    data.index = pd.to_datetime(data.index)

    # Create a custom market colors style for the candles
    my_style = mpf.make_mpf_style(base_mpf_style='charles', rc={'figure.figsize': (10, 6)})
    market_colors = mpf.make_marketcolors(up='blue', down='red', wick='i', edge='i', volume='in')
    my_style.update(marketcolors=market_colors)

    # Create a Matplotlib figure
    fig, _ = mpf.plot(data, type='candle', style=my_style, ylabel='Price', returnfig=True)

    # Render the figure as an image
    buffer = io.BytesIO()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buffer)
    image_data = buffer.getvalue()

    # Extract additional company info
    company_name = stock.info.get('longName', 'Unknown Company')
    website = stock.info.get('website', '')

    current_price = stock.history(period='1d')['Close'].iloc[-1]
    yesterday_price = stock.history(period='2d')['Close'].iloc[0]
    high_52_weeks = stock.info['fiftyTwoWeekHigh']
    low_52_weeks = stock.info['fiftyTwoWeekLow']

    # Get open, high, low, and close prices for today
    today_data = stock.history(period='1d')
    open_price = today_data['Open'].iloc[0]
    high_price = today_data['High'].iloc[0]
    low_price = today_data['Low'].iloc[0]
    close_price = today_data['Close'].iloc[0]

    # Creating a Candlestick Plotly figure
    fig1 = go.Figure()
    fig1.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='blue',
        decreasing_line_color='red',
        name='Candlestick'
    ))
    fig1.update_layout(
        title=f'{cname[:-3]} Candlestick Chart of 3 months',
        xaxis_title='Dates',
        yaxis_title='Prices',
        xaxis_rangeslider_visible=False,
        height=500
    )
    plotly_image1 = fig1.to_html(full_html=False)

    # Creating a Line Chart for the same data
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Line Chart'
    ))
    fig2.update_layout(
        title=f'{cname[:-3]} Line Chart for 3 months',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=500
    )
    plotly_image2 = fig2.to_html(full_html=False)

    # Prediction Model Building (assuming functions fetch_and_preprocess, create_sequences, build_model, and predict_stock_prices exist)
    df = stock.history(period="3mo")
    df = df.reset_index()
    dates, close_prices, scaled_close_prices, scaler = fetch_and_preprocess(cname)
    sequence_length = 5
    X = create_sequences(scaled_close_prices, sequence_length)
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y = scaled_close_prices[sequence_length:]
    y_train, y_test = y[:split_index], y[split_index:]
    model = build_model(X_train, y_train, sequence_length)
    predicted_prices = predict_stock_prices(model, scaled_close_prices, scaler, sequence_length)
    dates_predicted = pd.date_range(start=df['Date'].iloc[-1], periods=11)[1:]
    extended_dates = df['Date'].tolist() + [dates_predicted[0]]
    extended_prices = close_prices.tolist() + [predicted_prices.flatten()[0]]
    fig_pro = go.Figure()
    fig_pro.add_trace(go.Scatter(x=extended_dates, y=extended_prices, mode='lines', name='Original Data'))
    fig_pro.add_trace(go.Scatter(x=dates_predicted, y=predicted_prices.flatten(), mode='lines', name='Projected Data'))
    fig_pro.update_layout(
        title=f'{cname[:-3]} Stock Prices with Projections for 10 days',
        xaxis_title='Date',
        yaxis_title='Stock Price (Rs)',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        showlegend=True
    )
    project = fig_pro.to_html(full_html=False)

    # Context for template rendering
    context = {
        'company_name': company_name,
        'website': website[8:],  # Removing 'https://' from the beginning
        'current_price': current_price,
        'yesterday_price': yesterday_price,
        'year_week_high': high_52_weeks,
        'year   _week_low': low_52_weeks,
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close_price,
        'image_data': image_data,
        'plotly_image1': plotly_image1,
        'plotly_image2': plotly_image2,
        'project': project
    }

    return render_template('projections.html', **context)



if __name__ == '__main__':
    app.run(debug=True)
