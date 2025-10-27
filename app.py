import os
import pandas as pd
from flask import Flask, render_template, request, send_file, redirect, url_for 
import warnings
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import json 
from plotly.utils import PlotlyJSONEncoder
from io import BytesIO 

# Suppress sklearn warnings for clean output
warnings.filterwarnings("ignore")

# --- Configuration & Constants ---
app = Flask(__name__)

# Feature Columns (used for clustering and anomaly detection)
FEATURES = [
    "AVG. BAY HRS_HRS_FLOAT", 
    "AVG. BREAK HRS_HRS_FLOAT",
    "AVG. OOO HRS_HRS_FLOAT",
    "AVG. CAFETERIA HRS_HRS_FLOAT",
    "UNBILLED_HRS_FLOAT" 
]

# --- PROFESSIONAL COMPLIANCE CONSTANTS ---
BAY_HOUR_MANDATE = 7.0        
CRITICAL_BAY_LIMIT = 5.0      
HIGH_BREAK_LIMIT_CRITICAL = 3.0 
HIGH_BREAK_LIMIT_WARNING = 2.0  
HIGH_CHECKIN_LIMIT = 6        
BURN_OUT_LIMIT = 10.0         
# ---------------------------------------------

# Column Constants (MATCHED TO YOUR DATA HEADERS)
COL_FAKEID = "FAKE ID" 
COL_DESIGNATION = "DESIGNATION" 
COL_ACCOUNT = "ACCOUNT CODE" 
COL_IN = "AVG. IN TIME"
COL_OUT = "AVG. OUT TIME"
COL_BAY = "AVG. BAY HRS"
COL_BREAK = "AVG. BREAK HRS"
COL_CAFE = "AVG. CAFETERIA HRS"
COL_OOO = "AVG. OOO HRS"
COL_OFFICE = "AVG. OFFICE HRS"
COL_UNBILLED_FLAG = "UNBILLED" 
COL_UNALLOCATED = "UNALLOCATED" 

# Pie Chart Data Columns (MATCHED TO YOUR DATA HEADERS)
COL_HALF_DAY = "HALF-DAY LEAVE"
COL_FULL_DAY = "FULL-DAY LEAVE"
COL_CHECKIN = "ONLINE CHECK-IN"
COL_EXCEPTION = "EXCEMPTIONS" 

TIME_COLS = [COL_IN, COL_OUT, COL_OFFICE, COL_BAY, COL_BREAK, COL_CAFE, COL_OOO]

# Float Column Constants for easy reference
BAY_FLOAT = "AVG. BAY HRS_HRS_FLOAT"
BREAK_FLOAT = "AVG. BREAK HRS_HRS_FLOAT"
CAFE_FLOAT = "AVG. CAFETERIA HRS_HRS_FLOAT"
OOO_FLOAT = "AVG. OOO HRS_HRS_FLOAT"
OFFICE_FLOAT = "AVG. OFFICE HRS_HRS_FLOAT"
UNBILLED_FLOAT = "UNBILLED_HRS_FLOAT"

# Aesthetics (MU SIGMA INSPIRED COLORS - Updated from previous general colors)
ACCENT_BLUE_MU = "#99ccff"    # Baby Blue (Primary Mu Sigma Color)
ACCENT_GREEN = "#28a745"      # Compliance Pass
ACCENT_RED = "#dc3545"        # Financial Risk / Critical Fail
ACCENT_AMBER = "#ffb300"      # Warning / Break Time Attention
ACCENT_HEADER = "#1f77b4"     # Darker Blue for Header/Contrast (Original)
GRAPH_HEIGHT = 500 


# --- Utility Functions (unchanged) ---

def strip_formatting(text):
    """Removes all HTML tags (<b>, </b>) and Markdown bolding (**) for clean text output."""
    if isinstance(text, str):
        return text.replace('**', '').replace('<b>', '').replace('</b>', '')
    return str(text)

def time_to_float(time_str):
    """Converts a time string (HH:MM:SS) or Timedelta to float hours."""
    if pd.isna(time_str):
        return 0.0
    
    if isinstance(time_str, str) and len(time_str.split(':')) == 3:
        try:
            h, m, s = map(int, time_str.split(':'))
            return round(h + m / 60 + s / 3600, 2)
        except ValueError:
            return 0.0
    elif isinstance(time_str, pd.Timedelta):
        return round(time_str.total_seconds() / 3600, 2)
    
    return 0.0

def float_to_h_mm(float_hrs):
    """Converts float hours (e.g., 8.5) to a clean H:MM string format (e.g., "8:30")."""
    if float_hrs is None or float_hrs < 0:
        return "0:00"
    total_minutes = int(round(float_hrs * 60))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours}:{minutes:02d}"

# --- Data Loading (unchanged, includes dummy data) ---
def load_data():
    """Loads and preprocesses the attendance data."""
    global COL_HALF_DAY, COL_FULL_DAY, COL_CHECKIN, COL_EXCEPTION, COL_UNBILLED_FLAG, COL_UNALLOCATED
    
    FILE_PATH = os.path.join(os.getcwd(), 'data', 'attendance.xlsx')
    
    # --- Dummy Data Generation for testing if file doesn't exist ---
    if not os.path.exists(FILE_PATH):
        print("🚨 WARNING: Data file not found. Generating dummy data for demonstration.")
        data = {
            COL_FAKEID: ['EMP1001', 'EMP1002', 'EMP1003', 'EMP1004', 'EMP1005', 'EMP1006', 'EMP1007', 'EMP1008'], 
            COL_DESIGNATION: ['AL', 'Consultant', 'Manager', 'Analyst', 'Dev', 'Tester', 'Senior AL', 'Consultant'],
            "RECRUITMENT TYPE": ['Internal', 'Campus', 'Internal', 'Campus', 'Campus', 'Internal', 'Internal', 'Campus'],
            COL_ACCOUNT: ['SN', 'SN', 'B2', 'C3', 'SN', 'B2', 'B2', 'C3'],
            COL_IN: ['08:12:26', '09:05:00', '09:30:00', '08:45:00', '07:45:00', '10:00:00', '08:30:00', '09:15:00'],
            COL_OUT: ['20:43:46', '19:00:00', '18:15:00', '17:00:00', '16:00:00', '20:00:00', '18:00:00', '17:45:00'],
            COL_OFFICE: ['12:32:03', '09:55:00', '08:45:00', '08:15:00', '08:15:00', '10:00:00', '09:30:00', '08:30:00'],
            COL_BAY: ['11:32:06', '06:30:00', '05:00:00', '07:30:00', '04:00:00', '08:00:00', '06:45:00', '07:00:00'],
            COL_BREAK: ['00:59:56', '02:00:00', '02:30:00', '00:30:00', '03:00:00', '01:00:00', '01:15:00', '01:30:00'],
            COL_CAFE: ['00:12:20', '01:30:00', '01:00:00', '00:15:00', '01:00:00', '00:30:00', '00:45:00', '00:30:00'],
            COL_OOO: ['00:47:36', '00:30:00', '01:30:00', '00:15:00', '02:00:00', '00:30:00', '00:45:00', '00:30:00'],
            COL_UNBILLED_FLAG: ['BILLED', 'UNBILLED', 'BILLED', 'BILLED', 'UNBILLED', 'BILLED', 'UNBILLED', 'BILLED'],
            COL_HALF_DAY: [2, 2, 0, 0, 1, 0, 1, 0],
            COL_FULL_DAY: [8, 0, 1, 0, 0, 0, 0, 0],
            COL_CHECKIN: [10, 0, 0, 2, 7, 0, 1, 0], 
            COL_EXCEPTION: [5, 8, 1, 0, 6, 0, 0, 0], 
            COL_UNALLOCATED: ['NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO']
        }
        df = pd.DataFrame(data)
    # --- End Dummy Data ---
    else:
        try:
            df = pd.read_excel(FILE_PATH, engine='openpyxl') 
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load data from {FILE_PATH}. Error: {e}")
            return pd.DataFrame() 

    if not df.empty:
        # Standardize column names
        df.columns = df.columns.str.strip().str.replace('-', ' ').str.upper()

    # Re-map constants to the now-upper-cased, cleaned headers for internal use
    COL_HALF_DAY = COL_HALF_DAY.upper().replace('-', ' ')
    COL_FULL_DAY = COL_FULL_DAY.upper().replace('-', ' ')
    COL_CHECKIN = COL_CHECKIN.upper().replace('-', ' ')
    COL_EXCEPTION = COL_EXCEPTION.upper()
    COL_UNBILLED_FLAG = COL_UNBILLED_FLAG.upper()
    COL_UNALLOCATED = COL_UNALLOCATED.upper()

    if COL_FAKEID not in df.columns or COL_ACCOUNT not in df.columns:
        print(f"ERROR: Missing required columns.")
        return pd.DataFrame()
        
    df[COL_FAKEID] = df[COL_FAKEID].astype(str)
    df.set_index(COL_FAKEID, inplace=True)
    
    # 1. Convert time columns to float hours
    for col in TIME_COLS:
        col_cleaned = col.upper().replace('-', ' ')
        if col_cleaned in df.columns:
            new_col_name = f"{col_cleaned}_HRS_FLOAT" 
            df[new_col_name] = df[col_cleaned].apply(time_to_float)
        else:
            df[f"{col_cleaned}_HRS_FLOAT"] = 0.0
            
    # 2. Handle the UNBILLED status and penalty
    df['BILLED_STATUS'] = df[COL_UNBILLED_FLAG].apply(
        lambda x: 'UNBILLED' if isinstance(x, str) and x.strip().upper() == 'UNBILLED' else 'BILLED'
    )
    def unbilled_penalty_to_float(status):
        return 2.0 if isinstance(status, str) and status.strip().lower() == 'unbilled' else 0.0 

    if COL_UNBILLED_FLAG in df.columns:
        df[UNBILLED_FLOAT] = df[COL_UNBILLED_FLAG].apply(unbilled_penalty_to_float)
    else:
        df[UNBILLED_FLOAT] = 0.0
        
    # Check for the UNALLOCATED column (if not present, default to 'NO')
    if COL_UNALLOCATED not in df.columns:
        df[COL_UNALLOCATED] = 'NO'
    else:
        df[COL_UNALLOCATED] = df[COL_UNALLOCATED].astype(str).str.strip().str.upper()

    # 3. Handle PIE CHART data - Ensure columns exist and are numeric
    for col in [COL_HALF_DAY, COL_FULL_DAY, COL_CHECKIN, COL_EXCEPTION]:
        if col not in df.columns: 
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def run_clustering_and_anomaly_detection(df):
    """Performs K-Means clustering and Isolation Forest anomaly detection."""
    model_features = [f.upper() for f in FEATURES]
    
    if df.empty or not all(col in df.columns for col in model_features):
        return df, None, None
        
    try:
        data_for_model = df[model_features].fillna(0) 
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_model)
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["CLUSTER"] = kmeans.fit_predict(scaled_data)
        
        # Anomaly Detection (Isolation Forest)
        iso_forest = IsolationForest(random_state=42, contamination=0.1) 
        df["ANOMALY_SCORE"] = iso_forest.fit_predict(scaled_data)
        df["ANOMALY_FLAG"] = df["ANOMALY_SCORE"].apply(lambda x: x == -1)
        
        return df, kmeans, None 
    except Exception as e:
        print(f"Clustering/Anomaly detection failed: {e}")
        df["CLUSTER"] = -1
        df["ANOMALY_FLAG"] = False
        return df, None, None

def get_employee_row(emp_id):
    """Retrieves a single employee's data by ID."""
    global df
    if df.empty:
        return None
    try:
        emp_id = str(emp_id) 
        row_dict = df.loc[emp_id].to_dict()
        row_dict[COL_FAKEID] = emp_id 
        return row_dict
    except KeyError:
        return None

def get_account_data(account_code):
    """Retrieves all employee data for a specific account code."""
    global df
    if df.empty or account_code is None:
        return pd.DataFrame()
    try:
        account_code_upper = str(account_code).strip().upper()
        account_df = df[df[COL_ACCOUNT] == account_code_upper].copy()
        return account_df
    except Exception:
        return pd.DataFrame()


def calculate_comparison_details(emp_val, avg_val, metric_name):
    """
    Calculates difference, percentage, and returns a descriptive string. (Used by /employee)
    """
    emp_h_mm = float_to_h_mm(emp_val)
    
    if avg_val == 0 or avg_val < 0.01:
        if emp_val > 0.1: 
            return f"{emp_h_mm}. ⚠️ High disparity: Time recorded where team average is near zero."
        return f"{emp_h_mm}. ◉ In line with the team average of zero."

    diff = emp_val - avg_val
    time_display = f"{emp_h_mm}" 

    if abs(diff) < 0.1: # Threshold for 'in line' (e.g., within 6 minutes)
        return f"◉ {time_display}. Closely matches the average."

    percent_diff = (diff / avg_val) * 100
    
    status_word = "Higher" if diff > 0 else "Lower"
    adverb = ""
    
    if abs(percent_diff) > 25:
        adverb = "Significantly"
    elif abs(percent_diff) > 10:
        adverb = "Moderately"
    else:
        adverb = "Slightly"
    
    # Define indicators and impact based on metric type
    if metric_name == "Bay Hours":
        if diff > 0: indicator = "▲" # Upward trend is positive
        else: indicator = "▼" # Downward trend is negative
    else: # Break, Cafe, OOO (Less time is GOOD)
        if diff > 0: indicator = "🚨" # Alert symbol for deviation
        else: indicator = "✅" # Checkmark for efficiency

    impact = "Suggests strong focus and output." if metric_name == "Bay Hours" and diff > 0 else \
             "Suggests below-average concentration." if metric_name == "Bay Hours" and diff <= 0 else \
             "Potential drain on core productivity." if diff > 0 else \
             "Shows efficient time management."

    comparison = f"{indicator} {time_display} ({adverb} {status_word} the average). {impact}"
    return comparison

# --- CORE FUNCTION: generate_employee_story (unchanged) ---
def generate_employee_story(row):
    """
    Generates professional findings, assigns a main recommendation, and provides steps.
    """
    bay_hrs = row.get(BAY_FLOAT, 0)
    office_hrs = row.get(OFFICE_FLOAT, 0)
    break_hrs = row.get(BREAK_FLOAT, 0)
    unbilled_status = row.get('BILLED_STATUS', "BILLED").upper()
    unallocated_flag = row.get(COL_UNALLOCATED, "NO").upper() 
    checkin_count = row.get(COL_CHECKIN, 0)
    is_anomaly = row.get("ANOMALY_FLAG", False)
    
    bay_hrs_display = float_to_h_mm(bay_hrs)
    break_hrs_display = float_to_h_mm(break_hrs)
    
    findings = []
    recommendation_steps = []
    main_recommendation = "SUSTAIN PERFORMANCE."
    risk_tier = "" 

    is_unbilled_and_unallocated = (unbilled_status == 'UNBILLED' and unallocated_flag == 'YES')
    is_unbilled_but_allocated = (unbilled_status == 'UNBILLED' and unallocated_flag == 'NO')

    # --- TIER 1: CRITICAL INTERVENTION (High Risk to Company or Client) ---
    if is_unbilled_and_unallocated or bay_hrs < CRITICAL_BAY_LIMIT or is_anomaly:
        risk_tier = "CRITICAL"
        main_recommendation = "CRITICAL INTERVENTION REQUIRED: Immediate managerial review is mandatory to resolve policy breaches and mitigate financial risk."
        
        if is_unbilled_and_unallocated:
            findings.append(f"CRITICAL FINANCIAL RISK: Employee status is UNBILLED and UNALLOCATED. This represents a total loss of productivity and high financial exposure.")
            recommendation_steps.append("1. IMMEDIATE Allocation Directive: The Resource Manager must assign a critical, billable micro-project or clarify the resource status within 24 hours. The resource is idle.")
        
        if bay_hrs < CRITICAL_BAY_LIMIT:
            findings.append(f"SEVERE COMPLIANCE BREACH: Focused Floor Time is critically low at {bay_hrs_display} (less than 5 hours), indicating a profound failure to meet core presence requirements.")
            findings.append(f"BAY HOURS STATUS: Critically Low. This is the primary driver of the severe compliance breach.")
            step_num = len(recommendation_steps) + 1
            recommendation_steps.append(f"{step_num}. Mandatory 1-on-1: The supervisor must conduct a daily 15-minute sync to review time logs and implement a protected schedule for core floor hours.")
        
        if is_anomaly and "Critical Outlier Investigation" not in [r.split(':',1)[0].strip() for r in recommendation_steps]:
             step_num = len(recommendation_steps) + 1
             recommendation_steps.append(f"{step_num}. Critical Outlier Investigation (High Priority): Initiate a Root Cause Analysis (RCA) to determine if the significant behavioral deviation is due to high performance, policy violation, or environmental factors (360-degree scope).")

    # --- TIER 2: HIGH REVIEW (Compliance Breach or Company-Absorbed Cost) ---
    elif is_unbilled_but_allocated or bay_hrs < BAY_HOUR_MANDATE:
        risk_tier = "HIGH REVIEW"
        main_recommendation = "HIGH REVIEW REQUIRED: Employee is non-compliant or incurring company cost. Structured efficiency plan is necessary."

        if is_unbilled_but_allocated:
            findings.append(f"COMPANY COST RISK: Employee status is UNBILLED but ALLOCATED. The company is currently absorbing this resource cost. Investigate the billing issue.")
            recommendation_steps.append("1. Billing & Account Liaison Directive: The Manager must immediately verify the discrepancy in task codes and liaise with the Client Partner/Billing Desk to expedite billing clearance. The priority is to confirm allocation and move the resource to a billable status as quickly as possible, providing comprehensive reasons for the delay.")
            
        if bay_hrs < BAY_HOUR_MANDATE and bay_hrs >= CRITICAL_BAY_LIMIT:
            findings.append(f"COMPLIANCE WARNING: Focused Floor Time averages {bay_hrs_display}, a deficit of {float_to_h_mm(BAY_HOUR_MANDATE - bay_hrs)} against the 7-hour mandate.")
            findings.append(f"BAY HOURS STATUS: Below Mandate. Requires compliance plan.")
            step_num = len(recommendation_steps) + 1
            recommendation_steps.append(f"{step_num}. Compliance Plan: Implement a structured 2-week observation plan focused on enforcing core work blocks to meet the 7-hour floor mandate.")
        
    # --- TIER 3: BALANCE & EFFICIENCY (Compliant or High) ---
    else:
        risk_tier = "BALANCED"
        main_recommendation = "SUSTAIN PERFORMANCE: Employee is compliant with the floor mandate. Focus on optimizing time management and mitigating burnout risk."
        
        if bay_hrs > BAY_HOUR_MANDATE + 1.0: # Explicitly high Bay Hours (e.g., > 8.0 hrs)
             findings.append(f"POLICY COMPLIANCE: Employee meets the 7-hour floor mandate with an average of {bay_hrs_display}. This high floor time is a positive indicator of focus.")
             findings.append(f"BAY HOURS STATUS: High Compliance. Monitor for work-life balance issues.")
        else:
             findings.append(f"POLICY COMPLIANCE: Employee meets the 7-hour floor mandate with an average of {bay_hrs_display}.")
             findings.append(f"BAY HOURS STATUS: Compliant. Standard performance.")

        if office_hrs > BURN_OUT_LIMIT:
            findings.append(f"BURN-OUT RISK: Average Office Hours are high at {float_to_h_mm(office_hrs)}. This suggests a high-pressure workload or difficulty disconnecting.")
            recommendation_steps.append("1. Wellness Intervention: Manager must enforce structured breaks and confirm end-of-day workload closure to mitigate fatigue and prevent burnout.")
            
    # --- Secondary Findings (Regardless of Primary Tier) ---
    
    # Check for excessive break time (secondary aggravator)
    if break_hrs >= HIGH_BREAK_LIMIT_CRITICAL:
        findings.append(f"EXCESSIVE NON-PRODUCTIVE TIME: Break time is critically high at {break_hrs_display}. This is an additional risk factor.")
        step_num = len(recommendation_steps) + 1
        recommendation_steps.append(f"{step_num}. Time Audit Directive: Manager must instruct the employee to track all break time activities for 5 days to identify and curb time leakage sources.")
    elif break_hrs >= HIGH_BREAK_LIMIT_WARNING and risk_tier == "BALANCED":
        findings.append(f"TIME LEAKAGE WARNING: Break time is elevated at {break_hrs_display}. Optimize break usage for better focus.")
        step_num = len(recommendation_steps) + 1
        recommendation_steps.append(f"{step_num}. Break Optimization Review: Supervisor should review break patterns and advise on shorter, more frequent breaks (e.g., 5 min every hour) instead of long, excessive breaks.")
        
    if checkin_count >= HIGH_CHECKIN_LIMIT and 'HR Policy Refresher' not in [r.split(':',1)[0].strip() for r in recommendation_steps]:
        findings.append(f"POLICY DEVIATION: {checkin_count} Online Check-ins recorded. This pattern suggests inconsistent physical attendance.")
        step_num = len(recommendation_steps) + 1
        recommendation_steps.append(f"{step_num}. HR Policy Refresher: HR or Management must schedule the employee for a brief session clarifying the appropriate use of 'Online Check-in' vs. standard attendance policy.")


    if not recommendation_steps:
        recommendation_steps.append("1. Positive Reinforcement: Management should continue high performance acknowledgment; utilize this employee as a productivity benchmark for peers.")
    
    # Clean findings (story_lines) for template output
    cleaned_findings = [strip_formatting(f) for f in findings]
    
    # Clean recommendation steps for story output (though already mostly clean)
    cleaned_recommendation_steps = [strip_formatting(r) for r in recommendation_steps]


    return cleaned_findings, cleaned_recommendation_steps, main_recommendation

def generate_plotly_charts(df, emp_row):
    """
    Creates three Plotly charts for the employee analysis. (Uses original colors)
    """
    charts_json = {}
    
    # --- Data Extraction & Conversion for Bar Charts ---
    metrics = [BAY_FLOAT, BREAK_FLOAT, CAFE_FLOAT, OOO_FLOAT]
    metric_names = ["Bay Hours", "Break Hours", "Cafeteria Hours", "OOO Hours"]
    
    # --- Chart Generation Logic (UNCHANGED) ---
    emp_float_values = [emp_row.get(m, 0) for m in metrics]
    account_code = emp_row.get(COL_ACCOUNT)
    
    account_df = df[df[COL_ACCOUNT] == account_code]
    account_avg_float = account_df[metrics].mean().tolist() if not account_df.empty else [0] * len(metrics)
    company_avg_float = df[metrics].mean().tolist()
    
    # Convert float values to H:MM strings for the chart's labels
    emp_h_mm = [float_to_h_mm(v) for v in emp_float_values]
    account_avg_h_mm = [float_to_h_mm(v) for v in account_avg_float]
    company_avg_h_mm = [float_to_h_mm(v) for v in company_avg_float]
    
    # Create comparison DataFrames
    plot_df_account = pd.DataFrame({
        'Category': metric_names * 2,
        'Float Hours': emp_float_values + account_avg_float, 
        'Display Time': emp_h_mm + account_avg_h_mm, 
        'Type': ['Employee'] * len(metrics) + ['Account Average'] * len(metrics)
    })
    
    plot_df_company = pd.DataFrame({
        'Category': metric_names * 2,
        'Float Hours': emp_float_values + company_avg_float, 
        'Display Time': emp_h_mm + company_avg_h_mm, 
        'Type': ['Employee'] * len(metrics) + ['Company Average'] * len(metrics)
    })
    
    # --- Generate Dynamic Narratives (UNCHANGED) ---
    account_narrative = [strip_formatting(f"Comparison Status vs. Peers ({account_code} Account):")]
    company_narrative = [strip_formatting("Organizational Average Comparison Status:")]
    
    max_account_diff = 0
    account_annotations = []
    
    # ... (Narrative generation loop) ...
    for i, metric in enumerate(metric_names):
        acc_comp_str = calculate_comparison_details(emp_float_values[i], account_avg_float[i], metric)
        account_narrative.append(f"• {strip_formatting(metric)}: {strip_formatting(acc_comp_str)}")
        comp_comp_str = calculate_comparison_details(emp_float_values[i], company_avg_float[i], metric)
        company_narrative.append(f"• {strip_formatting(metric)}: {strip_formatting(comp_comp_str)}")
        if account_avg_float[i] > 0 and account_avg_float[i] != emp_float_values[i]:
            diff_pct = abs((emp_float_values[i] - account_avg_float[i]) / account_avg_float[i])
            if diff_pct > max_account_diff:
                max_account_diff = diff_pct
                annotation_text = strip_formatting(acc_comp_str).split(". ", 1)[-1].strip()
                account_annotations = [f"Primary Disparity: {metric} ({annotation_text})"]

    charts_json['narrative_1'] = "<br>".join(account_narrative)
    charts_json['narrative_2'] = "<br>".join(company_narrative)

    # --- Chart 1: Employee vs. Account Average Bar Chart (Uses old colors for consistency with original /employee code) ---
    fig1 = px.bar(
        plot_df_account, x='Category', y='Float Hours', color='Type', barmode='group', 
        title=f"Performance Benchmark: Employee vs. Account ({account_code}) Average",
        color_discrete_map={'Employee': ACCENT_HEADER, 'Account Average': ACCENT_AMBER},
        text='Display Time' 
    )
    
    if account_annotations:
        fig1.add_annotation(
            text=account_annotations[0],
            xref="paper", yref="paper", x=0.5, y=1.05, showarrow=False,
            font=dict(size=12, color="#333"), bgcolor="#FFF8DC", bordercolor="#FFD700", borderwidth=1, borderpad=4
        )
    
    fig1.update_layout(
        yaxis_title="Average Time (Hours:Minutes)", 
        xaxis_title="Time Metric", 
        template="plotly_white", 
        legend_title="Comparison Type",
        height=GRAPH_HEIGHT,
        font=dict(family="Arial, sans-serif", size=10, color="#444"), 
        margin=dict(t=100) 
    )
    fig1.update_traces(textposition='outside')
    
    charts_json['comparison_account'] = json.dumps(fig1.to_dict(), cls=PlotlyJSONEncoder) 

    # --- Chart 2: Employee vs. Company Average Bar Chart (UNCHANGED) ---
    fig2 = px.bar(
        plot_df_company, x='Category', y='Float Hours', color='Type', barmode='group',
        title="Organizational Benchmark: Employee vs. Overall Company Average",
        color_discrete_map={'Employee': ACCENT_HEADER, 'Company Average': '#9467BD'}, 
        text='Display Time' 
    )

    if company_narrative:
        fig2.add_annotation(
            text="Detailed organizational comparisons are listed below the chart.",
            xref="paper", yref="paper", x=0.5, y=1.05, showarrow=False,
            font=dict(size=12, color="#333"), bgcolor="#E6E6FA", bordercolor="#9467BD", borderwidth=1, borderpad=4
        )

    fig2.update_layout(
        yaxis_title="Average Time (Hours:Minutes)", 
        xaxis_title="Time Metric", 
        template="plotly_white", 
        legend_title="Comparison Type",
        height=GRAPH_HEIGHT,
        font=dict(family="Arial, sans-serif", size=10, color="#444"), 
        margin=dict(t=100)
    )
    fig2.update_traces(textposition='outside')
    
    charts_json['comparison_company'] = json.dumps(fig2.to_dict(), cls=PlotlyJSONEncoder) 
    
    # --- Chart 3: Leave and Exceptions Proportions Pie Chart (UNCHANGED) ---
    pie_labels = ['Half-Day Leaves', 'Full-Day Leaves', 'Online Check-ins', 'Exceptions']
    pie_values_raw = [
        emp_row.get(COL_HALF_DAY, 0),
        emp_row.get(COL_FULL_DAY, 0),
        emp_row.get(COL_CHECKIN, 0), 
        emp_row.get(COL_EXCEPTION, 0)
    ]
    
    pie_data = pd.DataFrame({'Label': pie_labels, 'Value': pie_values_raw})
    
    if pie_data['Value'].sum() == 0:
        charts_json['leave_exceptions_pie'] = "{}" 
        charts_json['narrative_3'] = (
            "No recorded leaves, exceptions, or check-ins were required for this employee. "
            "This indicates strong adherence to attendance policies."
        )
    else:
        fig3 = px.pie(
            pie_data, 
            names='Label',
            values='Value',
            title="Proportional Distribution of Non-Standard Attendance Events",
            color_discrete_sequence=px.colors.sequential.Agsunset, 
            hole=0.4 
        )
        fig3.update_traces(
            textinfo='percent+label', 
            pull=[0.05 if label in ['Exceptions', 'Full-Day Leaves'] else 0 for label in pie_data['Label']],
            marker=dict(line=dict(color='#FFFFFF', width=1.5)) 
        )
        fig3.update_layout(
            template="plotly_white", 
            legend_title="Category",
            height=GRAPH_HEIGHT,
            font=dict(family="Arial, sans-serif", size=10, color="#444")
        )
        charts_json['leave_exceptions_pie'] = json.dumps(fig3.to_dict(), cls=PlotlyJSONEncoder)
        
        charts_json['narrative_3'] = (
            "This Donut Chart illustrates the proportional distribution of leaves, required Online Check-ins, and other attendance events. "
            "It highlights the most frequent type of non-standard attendance behavior."
        )
    
    return charts_json

# --- Core FUNCTION: generate_dot_plot (unchanged) ---
def generate_dot_plot(account_df, account_code):
    """
    Generates a dot plot comparing employees' Bay Hours and Break Hours within the account.
    """
    if account_df.empty:
        return "{}"

    plot_df = account_df.copy()
    plot_df.reset_index(inplace=True)
    plot_df['Bay_H_MM'] = plot_df[BAY_FLOAT].apply(float_to_h_mm)
    plot_df['Break_H_MM'] = plot_df[BREAK_FLOAT].apply(float_to_h_mm)
    
    # Define compliance status for coloring
    plot_df['Compliance_Status'] = plot_df[BAY_FLOAT].apply(
        lambda x: 'Compliant (>=7h)' if x >= BAY_HOUR_MANDATE else 
                  'Low (<5h) / Critical Risk' if x < CRITICAL_BAY_LIMIT else
                  'Warning (<7h)'
    )
    
    # Define marker colors (using new constants)
    color_map = {
        'Compliant (>=7h)': ACCENT_GREEN,
        'Warning (<7h)': ACCENT_AMBER,
        'Low (<5h) / Critical Risk': ACCENT_RED
    }

    # Create the scatter plot (dot graph)
    fig = px.scatter(
        plot_df, 
        x=BAY_FLOAT, 
        y=BREAK_FLOAT, 
        color='Compliance_Status',
        title=f"Employee Behavioral Distribution in Account {account_code}",
        hover_data={
            COL_FAKEID: True,
            'Bay_H_MM': True,
            'Break_H_MM': True,
            BAY_FLOAT: False, # Hide raw float
            BREAK_FLOAT: False # Hide raw float
        },
        color_discrete_map=color_map,
        size_max=15 # Standardize marker size
    )
    
    # Add target lines
    fig.add_hline(y=HIGH_BREAK_LIMIT_WARNING, line_dash="dash", line_color=ACCENT_AMBER, annotation_text=f"Break Warning ({float_to_h_mm(HIGH_BREAK_LIMIT_WARNING)})", annotation_position="top left")
    fig.add_hline(y=HIGH_BREAK_LIMIT_CRITICAL, line_dash="dash", line_color=ACCENT_RED, annotation_text=f"Break Critical ({float_to_h_mm(HIGH_BREAK_LIMIT_CRITICAL)})", annotation_position="bottom right")
    fig.add_vline(x=BAY_HOUR_MANDATE, line_dash="dash", line_color=ACCENT_GREEN, annotation_text=f"Compliance Mandate ({BAY_HOUR_MANDATE}h)", annotation_position="top right")
    fig.add_vline(x=CRITICAL_BAY_LIMIT, line_dash="dash", line_color=ACCENT_RED, annotation_text=f"Critical Bay Limit ({CRITICAL_BAY_LIMIT}h)", annotation_position="bottom left")

    # Update axis labels and layout
    fig.update_layout(
        xaxis_title="Average Focused Floor Time (Bay Hours)",
        yaxis_title="Average Non-Productive Time (Break Hours)",
        template="plotly_white",
        height=GRAPH_HEIGHT,
        margin=dict(t=80, b=40, l=40, r=40)
    )
    
    return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)

def generate_account_report(account_df, account_code):
    """
    Generates summary metrics and a strategic narrative for the entire account team.
    REMOVED compliance_chart_json.
    """
    total_employees = len(account_df)
    
    if total_employees == 0:
        return {
            'summary': "No data available for this account.",
            'metrics': {},
            'dot_plot_json': "{}", 
            'unbilled_ids': []
        }

    # 1. METRICS CALCULATION
    
    # Compliance Rate (Bay Hours >= MANDATE)
    compliant_employees = account_df[account_df[BAY_FLOAT] >= BAY_HOUR_MANDATE]
    compliance_rate = round((len(compliant_employees) / total_employees) * 100, 1)

    # Financial Risk (Unbilled/Unallocated)
    unbilled_employees_df = account_df[account_df['BILLED_STATUS'] == 'UNBILLED']
    unbilled_employees = len(unbilled_employees_df)
    unbilled_ids = unbilled_employees_df.index.tolist() # Extract Unbilled IDs

    unallocated_employees = account_df[account_df[COL_UNALLOCATED] == 'YES']
    unallocated_count = len(unallocated_employees)
    
    # Average Times
    avg_bay_hrs = account_df[BAY_FLOAT].mean()
    avg_break_hrs = account_df[BREAK_FLOAT].mean()
    
    metrics = {
        'Total Employees': total_employees,
        'Compliance Rate (>=7h Bay)': f"{compliance_rate}%",
        'Avg. Focused Time (Bay)': float_to_h_mm(avg_bay_hrs),
        'Avg. Non-Productive Time (Break)': float_to_h_mm(avg_break_hrs),
        'Employees at Financial Risk (Unbilled)': unbilled_employees,
        'Unallocated Resources': unallocated_count
    }

    # 2. STRATEGIC NARRATIVE
    narrative = [f"## Strategic Overview for Account: {account_code}", "---"]
    main_directive = ""
    
    # Compliance Narrative
    if compliance_rate >= 90:
        narrative.append("✅ **EXCELLENT COMPLIANCE**: The team demonstrates exceptional adherence to floor time mandates. Focus should shift to talent retention.")
        main_directive = "Maintain high performance and initiate risk-based talent retention strategies."
    elif compliance_rate >= 75:
        narrative.append("🔶 **GOOD COMPLIANCE**: Compliance is strong but has pockets of deviation. Address individual non-compliance to reach target levels.")
        main_directive = "Target specific non-compliant individuals to raise the overall team average."
    else:
        narrative.append("🛑 **CRITICAL COMPLIANCE ISSUE**: The account compliance rate is low. A structural intervention plan is required immediately.")
        main_directive = "Immediate managerial intervention and a 30-day compliance remediation plan are mandatory."

    # Financial Risk Narrative
    if unbilled_employees > 0:
        risk_percentage = round((unbilled_employees / total_employees) * 100, 1)
        narrative.append(f"🚨 **FINANCIAL RISK ALERT**: {unbilled_employees} employees ({risk_percentage}%) are currently flagged as Unbilled. This is a critical revenue leakage point. {unallocated_count} are UNALLOCATED resources.")
    
    # Break Time Narrative
    if avg_break_hrs > HIGH_BREAK_LIMIT_CRITICAL:
        narrative.append(f"⚠️ **EFFICIENCY WARNING**: The average break time is {float_to_h_mm(avg_break_hrs)}, indicating potential widespread time leakage. Review team scheduling practices.")
    
    narrative.append(f"\n**PRIMARY DIRECTIVE**: {main_directive}")

    # **3. CHART GENERATION (Compliance Breakdown - REMOVED)**

    # 4. CHART GENERATION (Dot Plot)
    dot_plot_json = generate_dot_plot(account_df, account_code)
    
    # Use strip_formatting to clean the narrative for HTML pre tags, then preserve newlines
    clean_summary = strip_formatting("</br>".join(narrative)).replace("</br>", "\n")

    return {
        'summary': clean_summary,
        'metrics': metrics,
        'dot_plot_json': dot_plot_json, 
        'unbilled_ids': unbilled_ids
    }


# Global Data Initialization
df_raw = load_data()
df, kmeans, _ = run_clustering_and_anomaly_detection(df_raw.copy())


# --- Flask Routes ---



@app.route("/employee", methods=["GET", "POST"])
def employee():
    """
    Employee route is entirely unchanged, as requested. 
    It will use the original chart generation with old colors.
    """
    suggested_id = df.index[0] if not df.empty else "EMP1001" 
    
    # Initial load/error state handling
    if (request.method == "GET" and 'emp_id' not in request.args) or request.form.get("emp_id", "").strip() == "":
        return render_template("employee.html", 
                               emp_id="Enter ID", 
                               suggested_id=suggested_id,
                               is_initial_load=True, 
                               error="Please enter an Employee ID above to begin analysis.")
        
    emp_id_to_load = request.args.get("emp_id", "").strip()
    
    if request.method == "POST":
        emp_id_to_load = request.form.get("emp_id", "").strip()
    
    if not emp_id_to_load or emp_id_to_load == "Enter ID":
        return redirect(url_for('employee'))


    row = get_employee_row(emp_id_to_load)
    
    if row is None:
        return render_template("employee.html", 
                               emp_id=emp_id_to_load, 
                               suggested_id=suggested_id,
                               is_initial_load=True, 
                               error=f"Employee ID '{emp_id_to_load}' not found. Please try a valid ID (e.g., {suggested_id}).",
                               chart_json_1="{}", narrative_1="", chart_json_2="{}", narrative_2="", chart_json_3="{}", narrative_3="",
                               story=[], recommendation="")

    # Prepare Allocation Status display
    raw_allocation_status = row.get(COL_UNALLOCATED, "NO").upper()
    if raw_allocation_status == 'YES':
        allocation_display = "UNALLOCATED 🛑"
        allocation_class = "unallocated"
    else: # 'NO'
        allocation_display = "ALLOCATED ✅"
        allocation_class = "allocated"

    # Employee Details for display (7hr Compliance card removed)
    employee_details = {
        "ID": row[COL_FAKEID], 
        "Designation": row.get(COL_DESIGNATION, "N/A"),
        "Account Code": row.get(COL_ACCOUNT, "N/A"), 
        "Online Check-in Count": row.get(COL_CHECKIN, 0), 
        "Avg. In Time": row.get(COL_IN, "N/A"), 
        "Avg. Out Time": row.get(COL_OUT, "N/A"),
        "Billed Status (Client)": row.get('BILLED_STATUS', "N/A").upper(),
        # New Allocation Keys for HTML card
        "Allocation Display": allocation_display,
        "Allocation Class": allocation_class
    }
    
    # Time Data for the "Tree" (Detailed Breakdown)
    office_hrs = row.get(OFFICE_FLOAT, 0)
    bay_hrs = row.get(BAY_FLOAT, 0)
    cafe_hrs = row.get(CAFE_FLOAT, 0)
    ooo_hrs = row.get(OOO_FLOAT, 0)
    break_total = row.get(BREAK_FLOAT, 0) 
    
    tree_data = {
        "Office Hours": float_to_h_mm(office_hrs),
        "Bay Hours (Focused)": float_to_h_mm(bay_hrs),
        "Break Hours (Total)": float_to_h_mm(break_total),
        "Cafeteria Hours": float_to_h_mm(cafe_hrs),
        "OOO Hours": float_to_h_mm(ooo_hrs)
    }
    
    # Generate Analysis
    plotly_data = generate_plotly_charts(df, row)
    story_lines, recommendation_steps, main_rec = generate_employee_story(row)

    # Combine recommendation steps into a single HTML string, now entirely plain text
    formatted_recommendation = f"Primary Directive: {main_rec}<br><br>"
    formatted_recommendation += "<br>".join(f"{step}" for step in recommendation_steps)


    return render_template("employee.html",
                           emp_id=emp_id_to_load,
                           suggested_id=suggested_id,
                           is_initial_load=False,
                           employee_details=employee_details,
                           tree_data=tree_data, 
                           
                           chart_json_1=plotly_data['comparison_account'],
                           chart_json_2=plotly_data['comparison_company'],
                           chart_json_3=plotly_data['leave_exceptions_pie'],
                           
                           narrative_1=plotly_data['narrative_1'],
                           narrative_2=plotly_data['narrative_2'],
                           narrative_3=plotly_data['narrative_3'],
                           
                           story=story_lines,
                           recommendation=formatted_recommendation) 

@app.route("/download_report/<emp_id>")
def download_report(emp_id):
    """
    Route to generate and download the employee analysis report. (UNCHANGED)
    """
    row = get_employee_row(emp_id)
    if row is None:
        return "Employee ID not found.", 404
        
    findings, recommendations, main_rec = generate_employee_story(row) 
    
    def clean_text(text):
        return text.replace('**', '').replace('<b>', '').replace('</b>', '')

    output = f"EMPLOYEE ATTENDANCE COMPLIANCE & BEHAVIORAL ANALYSIS - ID: {emp_id}\n"
    output += "="*60 + "\n\n"
    
    # 1. MAIN RECOMMENDATION (EXECUTIVE SUMMARY)
    output += "[ I. EXECUTIVE SUMMARY AND PRIMARY DIRECTIVE ]\n"
    output += "-"*60 + "\n"
    output += f"MAIN RECOMMENDATION: {clean_text(main_rec)}\n"
    output += "-"*60 + "\n\n"
    
    # 2. CORE TIME METRICS
    BAY_FLOAT = "AVG. BAY HRS_HRS_FLOAT"
    OFFICE_FLOAT = "AVG. OFFICE HRS_HRS_FLOAT"
    BREAK_FLOAT = "AVG. BREAK HRS_HRS_FLOAT"
    COL_UNALLOCATED = "UNALLOCATED"
    
    output += "[ II. CORE TIME METRICS ]\n"
    output += f"Avg. Floor Time (Bay): {float_to_h_mm(row.get(BAY_FLOAT, 0))}\n"
    output += f"Avg. Office Time (Total): {float_to_h_mm(row.get(OFFICE_FLOAT, 0))}\n"
    output += f"Avg. Total Break Time: {float_to_h_mm(row.get(BREAK_FLOAT, 0))}\n"
    output += f"Billed Status (Client): {row.get('BILLED_STATUS', 'N/A').upper()}\n"
    
    allocation_display = "ALLOCATED" if row.get(COL_UNALLOCATED, 'NO').upper() == 'NO' else "UNALLOCATED"
    output += f"Allocation Status: {allocation_display}\n\n"

    # 3. BEHAVIORAL FINDINGS
    output += "[ III. BEHAVIORAL FINDINGS & NON-COMPLIANCE NOTES ]\n"
    if not findings:
         output += "• No critical or warning-level behavioral patterns detected.\n"
    else:
        for finding in findings:
            output += f"• {clean_text(finding)}\n" 
    
    output += "\n" + "-"*60 + "\n\n"
    
    # 4. STRATEGIC ACTION PLAN
    output += "[ IV. STRATEGIC ACTION PLAN (Management Directives) ]\n"
    for step in recommendations:
        output += f"{clean_text(step)}\n"
        
    # Return as in-memory file
    report_file = BytesIO(output.encode('utf-8'))
    
    return send_file(
        report_file,
        mimetype='text/plain',
        as_attachment=True,
        download_name=f'Attendance_Report_{emp_id}.txt'
    )
FULL_ACCOUNT_CODES = [
    "SN", "TM", "TS", "CJ", "SM", "LC", "CC", "TE",
    "TSO", "LD", "HN", "ET", "TH", "MA", "BB", "CG",
    "EA", "OM", "SA", "SC", "AC", "NR", "IS", "CL",
    "NH", "EW", "ED", "EJ", "NI", "NB", "AE", "XR",
    "EN", "OW", "PC", "CS", "UR", "YB", "DL", "DA",
    "NJ"
]
VALID_ACCOUNT_CODES_SET = set(FULL_ACCOUNT_CODES)

# ===================================================================
# Account Data Utility Functions
# ===================================================================

def get_actual_column(df, possible_names, fallback=None):
    """
    Return the first matching column from possible_names.
    If none found, return fallback or raise KeyError.
    """
    df_cols = [c.strip() for c in df.columns]
    for name in possible_names:
        if name in df_cols:
            return name
        # Case-insensitive match
        for col in df_cols:
            if col.lower() == name.lower():
                return col
    if fallback:
        return fallback
    raise KeyError(f"None of the expected columns {possible_names} found in DataFrame. Available columns: {df.columns.tolist()}")

def get_account_data(account_code):
    """
    Fetch data for a specific account, with dynamic column detection.
    """
    global df
    # Reset index to make FAKE ID available as a column
    df_reset = df.reset_index()
    df_reset.columns = df_reset.columns.str.strip()  # remove spaces
    account_col = get_actual_column(df_reset, [COL_ACCOUNT])
    filtered_df = df_reset[df_reset[account_col] == account_code].copy()
    return filtered_df

def create_comparison_charts(account_df, account_code):
    """Create comparison charts with company averages"""
    charts = {}
    
    print(f"DEBUG: Creating comparison charts for {account_code} with {len(account_df)} employees")
    
    # Get company averages and convert to Python floats
    company_avg_bay = float(df[BAY_FLOAT].mean())
    company_avg_break = float(df[BREAK_FLOAT].mean())
    company_avg_cafe = float(df[CAFE_FLOAT].mean())
    company_avg_ooo = float(df[OOO_FLOAT].mean())
    
    # Get account averages and convert to Python floats
    account_avg_bay = float(account_df[BAY_FLOAT].mean())
    account_avg_break = float(account_df[BREAK_FLOAT].mean())
    account_avg_cafe = float(account_df[CAFE_FLOAT].mean())
    account_avg_ooo = float(account_df[OOO_FLOAT].mean())
    
    # 1. Create Compliance Distribution Pie Chart
    try:
        # Convert counts to Python integers
        compliant_count = int((account_df[BAY_FLOAT] >= BAY_HOUR_MANDATE).sum())
        warning_count = int(((account_df[BAY_FLOAT] >= CRITICAL_BAY_LIMIT) & (account_df[BAY_FLOAT] < BAY_HOUR_MANDATE)).sum())
        critical_count = int((account_df[BAY_FLOAT] < CRITICAL_BAY_LIMIT).sum())
        
        compliance_labels = ['Compliant (≥7h)', 'Warning (5-7h)', 'Critical (<5h)']
        compliance_values = [compliant_count, warning_count, critical_count]
        compliance_colors = [ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED]
        
        compliance_trace = {
            'labels': compliance_labels,
            'values': compliance_values,
            'type': 'pie',
            'marker': {'colors': compliance_colors},
            'hole': 0.4,
            'textinfo': 'percent+label',
            'hovertemplate': '<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        }
        
        compliance_layout = {
            'title': f'Bay Hours Compliance Distribution - {account_code}',
            'height': 500,
            'showlegend': True,
            'annotations': [{
                'text': 'Compliance',
                'x': 0.5, 'y': 0.5,
                'font': {'size': 16},
                'showarrow': False
            }]
        }
        
        charts['compliance_distribution'] = {'data': [compliance_trace], 'layout': compliance_layout}
        print("DEBUG: Compliance distribution chart created successfully")
        
    except Exception as e:
        print(f"ERROR creating compliance distribution chart: {e}")
        charts['compliance_distribution'] = {'data': [], 'layout': {}}
    
    # 2. Create All Metrics Comparison Chart
    try:
        metrics = ['Bay Hours', 'Break Hours', 'Cafeteria Hours', 'OOO Hours']
        account_values = [account_avg_bay, account_avg_break, account_avg_cafe, account_avg_ooo]
        company_values = [company_avg_bay, company_avg_break, company_avg_cafe, company_avg_ooo]
        
        account_trace = {
            'x': metrics,
            'y': account_values,
            'type': 'bar',
            'name': f'{account_code} Account',
            'marker': {'color': ACCENT_HEADER},
            'text': [f'{val:.1f}h' for val in account_values],
            'textposition': 'auto'
        }
        
        company_trace = {
            'x': metrics,
            'y': company_values,
            'type': 'bar',
            'name': 'Company Average',
            'marker': {'color': ACCENT_AMBER},
            'text': [f'{val:.1f}h' for val in company_values],
            'textposition': 'auto'
        }
        
        all_metrics_layout = {
            'title': f'Time Metrics Comparison: {account_code} vs Company',
            'xaxis': {'title': 'Time Metrics'},
            'yaxis': {'title': 'Average Hours'},
            'height': 500,
            'barmode': 'group',
            'showlegend': True
        }
        
        charts['all_metrics_comparison'] = {'data': [account_trace, company_trace], 'layout': all_metrics_layout}
        print("DEBUG: All metrics comparison chart created successfully")
        
    except Exception as e:
        print(f"ERROR creating all metrics comparison chart: {e}")
        charts['all_metrics_comparison'] = {'data': [], 'layout': {}}
    
    # 3. Create Behavioral Distribution Dot Plot
    if not account_df.empty and BAY_FLOAT in account_df.columns and BREAK_FLOAT in account_df.columns:
        try:
            # Get employee IDs
            emp_col = get_actual_column(account_df, [COL_FAKEID], 'index')
            employee_ids = account_df[emp_col].tolist()
            
            # Convert numpy arrays to Python lists with float conversion
            bay_hours_list = [float(x) for x in account_df[BAY_FLOAT].tolist()]
            break_hours_list = [float(x) for x in account_df[BREAK_FLOAT].tolist()]
            
            # Create color coding based on compliance
            colors = []
            for bay_hrs in bay_hours_list:
                if bay_hrs >= BAY_HOUR_MANDATE:
                    colors.append(ACCENT_GREEN)  # Compliant
                elif bay_hrs >= CRITICAL_BAY_LIMIT:
                    colors.append(ACCENT_AMBER)  # Warning
                else:
                    colors.append(ACCENT_RED)  # Critical
            
            dot_trace = {
                'x': bay_hours_list,
                'y': break_hours_list,
                'mode': 'markers',
                'type': 'scatter',
                'marker': {
                    'size': 12, 
                    'color': colors,
                    'line': {'width': 2, 'color': 'darkgray'}
                },
                'text': employee_ids,
                'hovertemplate': '<b>Employee: %{text}</b><br>Bay Hours: %{x:.1f}<br>Break Hours: %{y:.1f}<extra></extra>'
            }
            
            dot_layout = {
                'title': f'Employee Behavioral Distribution - {account_code}',
                'xaxis': {'title': 'Focused Floor Time (Bay Hours)'},
                'yaxis': {'title': 'Non-Productive Time (Break Hours)'},
                'height': 500,
                'showlegend': False,
                'shapes': [
                    # Vertical line for 7-hour mandate
                    {'type': 'line', 'x0': 7, 'x1': 7, 'y0': 0, 'y1': 1, 'yref': 'paper', 
                     'line': {'color': ACCENT_GREEN, 'width': 2, 'dash': 'dash'}},
                    # Horizontal line for break warning
                    {'type': 'line', 'x0': 0, 'x1': 1, 'xref': 'paper', 'y0': 2, 'y1': 2,
                     'line': {'color': ACCENT_AMBER, 'width': 2, 'dash': 'dash'}},
                    # Horizontal line for break critical
                    {'type': 'line', 'x0': 0, 'x1': 1, 'xref': 'paper', 'y0': 3, 'y1': 3,
                     'line': {'color': ACCENT_RED, 'width': 2, 'dash': 'dash'}}
                ]
            }
            
            charts['behavioral_distribution'] = {'data': [dot_trace], 'layout': dot_layout}
            print("DEBUG: Behavioral distribution chart created successfully")
            
        except Exception as e:
            print(f"ERROR creating behavioral distribution chart: {e}")
            charts['behavioral_distribution'] = {'data': [], 'layout': {}}
    
    return charts

def generate_account_report(account_df, account_code):
    """
    Prepare all metrics and charts JSON for a given account.
    """
    # Detect required columns
    emp_col = get_actual_column(account_df, [COL_FAKEID])
    unbilled_col = get_actual_column(account_df, [COL_UNBILLED_FLAG])

    # ---------------------------
    # KPI Cards
    # ---------------------------
    total_employees_account = len(account_df)
    
    print(f"DEBUG: Processing account {account_code} with {total_employees_account} employees")
    
    # Calculate compliance rate (Bay Hours >= 7.0)
    compliant_count = (account_df[BAY_FLOAT] >= BAY_HOUR_MANDATE).sum()
    compliance_rate = round((compliant_count / total_employees_account) * 100, 1) if total_employees_account > 0 else 0

    # Calculate unbilled employees count
    unbilled_count = 0
    unbilled_ids = []
    
    if account_df[unbilled_col].dtype == 'object':  # String values
        unbilled_mask = account_df[unbilled_col].str.upper() == 'UNBILLED'
        unbilled_count = unbilled_mask.sum()
        unbilled_ids = account_df[unbilled_mask][emp_col].tolist()
    else:  # Numeric values (1/0)
        unbilled_mask = account_df[unbilled_col] == 1
        unbilled_count = unbilled_mask.sum()
        unbilled_ids = account_df[unbilled_mask][emp_col].tolist()
    
    print(f"DEBUG: Found {unbilled_count} unbilled employees: {unbilled_ids}")

    # Calculate unallocated count
    unallocated_count = 0
    unallocated_ids = []
    if COL_UNALLOCATED in account_df.columns:
        if account_df[COL_UNALLOCATED].dtype == 'object':  # String values
            unallocated_mask = account_df[COL_UNALLOCATED].str.upper() == 'YES'
            unallocated_count = unallocated_mask.sum()
            unallocated_ids = account_df[unallocated_mask][emp_col].tolist()
        else:  # Numeric values
            unallocated_mask = account_df[COL_UNALLOCATED] == 1
            unallocated_count = unallocated_mask.sum()
            unallocated_ids = account_df[unallocated_mask][emp_col].tolist()
    
    print(f"DEBUG: Found {unallocated_count} unallocated resources: {unallocated_ids}")

    # Calculate averages
    avg_bay_hrs = account_df[BAY_FLOAT].mean()
    avg_break_hrs = account_df[BREAK_FLOAT].mean()

    metrics = {
        "Total Employees": total_employees_account,
        "Compliance Rate (>=7h Bay)": f"{compliance_rate}%",
        "Avg. Focused Time (Bay)": float_to_h_mm(avg_bay_hrs),
        "Employees at Financial Risk (Unbilled)": unbilled_count,
        "Avg. Non-Productive Time (Break)": float_to_h_mm(avg_break_hrs),
        "Unallocated Resources": unallocated_count
    }

    # ---------------------------
    # Strategic Narrative
    # ---------------------------
    narrative = [f"## Strategic Overview for Account: {account_code}", "---"]
    main_directive = ""
    
    # Compliance Narrative
    if compliance_rate >= 90:
        narrative.append("✅ **EXCELLENT COMPLIANCE**: The team demonstrates exceptional adherence to floor time mandates. Focus should shift to talent retention.")
        main_directive = "Maintain high performance and initiate risk-based talent retention strategies."
    elif compliance_rate >= 75:
        narrative.append("🔶 **GOOD COMPLIANCE**: Compliance is strong but has pockets of deviation. Address individual non-compliance to reach target levels.")
        main_directive = "Target specific non-compliant individuals to raise the overall team average."
    else:
        narrative.append("🛑 **CRITICAL COMPLIANCE ISSUE**: The account compliance rate is low. A structural intervention plan is required immediately.")
        main_directive = "Immediate managerial intervention and a 30-day compliance remediation plan are mandatory."

    # Financial Risk Narrative
    if unbilled_count > 0:
        risk_percentage = round((unbilled_count / total_employees_account) * 100, 1)
        narrative.append(f"🚨 **FINANCIAL RISK ALERT**: {unbilled_count} employees ({risk_percentage}%) are currently flagged as Unbilled. This is a critical revenue leakage point.")
        if unallocated_count > 0:
            narrative.append(f"🚨 **RESOURCE ALLOCATION ISSUE**: {unallocated_count} employees are currently unallocated and generating no revenue.")
    else:
        narrative.append("✅ **FINANCIAL STATUS**: All employees are properly billed with no financial risk identified.")
        if unallocated_count > 0:
            narrative.append(f"⚠️ **ALLOCATION NOTE**: {unallocated_count} employees are unallocated but properly billed.")
        else:
            narrative.append("✅ **RESOURCE UTILIZATION**: All employees are properly allocated to projects.")
    
    # Break Time Narrative
    if avg_break_hrs > HIGH_BREAK_LIMIT_CRITICAL:
        narrative.append(f"⚠️ **EFFICIENCY WARNING**: The average break time is {float_to_h_mm(avg_break_hrs)}, indicating potential widespread time leakage. Review team scheduling practices.")
    elif avg_break_hrs > HIGH_BREAK_LIMIT_WARNING:
        narrative.append(f"🔶 **BREAK TIME MONITORING**: Average break time of {float_to_h_mm(avg_break_hrs)} is elevated and should be monitored.")
    else:
        narrative.append(f"✅ **BREAK TIME MANAGEMENT**: Average break time of {float_to_h_mm(avg_break_hrs)} is within acceptable limits.")
    
    # Bay Hours Narrative
    if avg_bay_hrs >= BAY_HOUR_MANDATE:
        narrative.append(f"✅ **FOCUSED WORK**: Average bay hours of {float_to_h_mm(avg_bay_hrs)} meet or exceed the compliance mandate.")
    else:
        narrative.append(f"⚠️ **FOCUS CONCERN**: Average bay hours of {float_to_h_mm(avg_bay_hrs)} are below the 7-hour mandate.")
    
    # Get company averages for comparison insights
    company_avg_bay = df[BAY_FLOAT].mean()
    company_avg_break = df[BREAK_FLOAT].mean()
    
    # Add comparison insights
    narrative.extend(["", "## Comparative Performance Analysis", "---"])
    
    # Bay hours comparison
    bay_diff = avg_bay_hrs - company_avg_bay
    if bay_diff > 1.0:
        narrative.append(f"✅ **STRONG FOCUS**: Account bay hours are {bay_diff:.1f}h higher than company average, indicating excellent focus.")
    elif bay_diff > 0:
        narrative.append(f"🔶 **GOOD FOCUS**: Account bay hours are slightly higher than company average.")
    elif bay_diff < -1.0:
        narrative.append(f"⚠️ **FOCUS CONCERN**: Account bay hours are {abs(bay_diff):.1f}h lower than company average.")
    else:
        narrative.append(f"⚖️ **AVERAGE FOCUS**: Account bay hours are in line with company average.")
    
    # Break hours comparison
    break_diff = avg_break_hrs - company_avg_break
    if break_diff > 1.0:
        narrative.append(f"🚨 **BREAK CONCERN**: Account break hours are {break_diff:.1f}h higher than company average, indicating potential time leakage.")
    elif break_diff > 0:
        narrative.append(f"🔶 **BREAK MONITORING**: Account break hours are slightly higher than company average.")
    elif break_diff < -1.0:
        narrative.append(f"✅ **EFFICIENT BREAKS**: Account break hours are {abs(break_diff):.1f}h lower than company average, showing good time management.")
    else:
        narrative.append(f"⚖️ **STANDARD BREAKS**: Account break hours are in line with company average.")
    
    narrative.append(f"\n**PRIMARY DIRECTIVE**: {main_directive}")

    # ---------------------------
    # Create Comparison Charts
    # ---------------------------
    charts = create_comparison_charts(account_df, account_code)
    
    # Use strip_formatting to clean the narrative
    clean_summary = strip_formatting("</br>".join(narrative)).replace("</br>", "\n")

    print(f"DEBUG: Final metrics - {metrics}")
    
    return {
        "summary": clean_summary,
        "metrics": metrics,
        "behavioral_distribution_json": json.dumps(charts.get('behavioral_distribution', {})),
        "compliance_distribution_json": json.dumps(charts.get('compliance_distribution', {})),
        "all_metrics_comparison_json": json.dumps(charts.get('all_metrics_comparison', {})),
        "unbilled_ids": unbilled_ids,
        "unallocated_ids": unallocated_ids,
        # Pass comparison data for insights
        "account_avg_bay": avg_bay_hrs,
        "company_avg_bay": company_avg_bay,
        "account_avg_break": avg_break_hrs,
        "company_avg_break": company_avg_break
    }


# ===================================================================
# /account Route
# ===================================================================
@app.route("/account", methods=["GET", "POST"])
def account():
    unique_accounts = FULL_ACCOUNT_CODES
    suggested_account = unique_accounts[0]

    account_code_to_load = request.args.get("account_code", "").strip()
    if request.method == "POST":
        account_code_to_load = request.form.get("account_code", "").strip()

    if not account_code_to_load or account_code_to_load not in VALID_ACCOUNT_CODES_SET:
        error_message = "Please select a valid Account Code from the dropdown to begin analysis."
        if account_code_to_load and account_code_to_load not in VALID_ACCOUNT_CODES_SET:
            error_message = f"Invalid Account Code selected: '{account_code_to_load}'. Please choose one from the list."
        return render_template(
            "account.html",
            account_code="",
            suggested_account=suggested_account,
            unique_accounts=unique_accounts,
            is_initial_load=True,
            error=error_message
        )

    # Load account data
    account_df = get_account_data(account_code_to_load)
    if account_df.empty:
        return render_template(
            "account.html",
            account_code=account_code_to_load,
            suggested_account=suggested_account,
            unique_accounts=unique_accounts,
            is_initial_load=True,
            error=f"Account Code '{account_code_to_load}' has no data. Try another code (e.g., {suggested_account})."
        )

    report_data = generate_account_report(account_df, account_code_to_load)

    return render_template(
        "account.html",
        account_code=account_code_to_load,
        suggested_account=suggested_account,
        unique_accounts=unique_accounts,
        is_initial_load=False,
        report_summary=report_data['summary'],
        metrics=report_data['metrics'],
        behavioral_distribution_json=report_data['behavioral_distribution_json'],
        compliance_distribution_json=report_data['compliance_distribution_json'],
        all_metrics_comparison_json=report_data['all_metrics_comparison_json'],
        unbilled_ids=report_data['unbilled_ids'],
        unallocated_ids=report_data['unallocated_ids'],
        account_avg_bay=report_data['account_avg_bay'],
        company_avg_bay=report_data['company_avg_bay'],
        account_avg_break=report_data['account_avg_break'],
        company_avg_break=report_data['company_avg_break']
    )

# ===============================


# --- New Overview Page Constants ---
OVERVIEW_DESIGNATIONS = ['AL', 'Consultant', 'Manager', 'Analyst', 'Dev', 'Tester', 'Senior AL']
@app.route("/overview", methods=["GET", "POST"])
def overview():
    # Get filter parameters
    selected_designation = request.form.get("designation_filter", "All")
    selected_account = request.form.get("account_filter", "All")
    
    # Filter data based on selections
    filtered_df = df.copy()
    
    if selected_designation != "All":
        filtered_df = filtered_df[filtered_df[COL_DESIGNATION] == selected_designation]
    
    if selected_account != "All":
        filtered_df = filtered_df[filtered_df[COL_ACCOUNT] == selected_account]
    
    # Calculate overview metrics for cards
    total_employees = len(filtered_df)
    
    # Average times
    avg_in_time = filtered_df[COL_IN].mode()[0] if not filtered_df.empty and not filtered_df[COL_IN].mode().empty else "N/A"
    avg_out_time = filtered_df[COL_OUT].mode()[0] if not filtered_df.empty and not filtered_df[COL_OUT].mode().empty else "N/A"
    avg_office_hrs = float_to_h_mm(filtered_df[OFFICE_FLOAT].mean()) if not filtered_df.empty else "0:00"
    
    # Average leaves
    avg_full_day_leaves = round(filtered_df[COL_FULL_DAY].mean(), 1) if not filtered_df.empty else 0
    avg_half_day_leaves = round(filtered_df[COL_HALF_DAY].mean(), 1) if not filtered_df.empty else 0
    
    # Unbilled and Unallocated counts with hover data
    unbilled_count = len(filtered_df[filtered_df['BILLED_STATUS'] == 'UNBILLED'])
    unbilled_ids = filtered_df[filtered_df['BILLED_STATUS'] == 'UNBILLED'].index.tolist()
    
    unallocated_count = len(filtered_df[filtered_df[COL_UNALLOCATED] == 'YES'])
    unallocated_ids = filtered_df[filtered_df[COL_UNALLOCATED] == 'YES'].index.tolist()
    
    # Bay hour maintenance percentage (employees with >= 7 hours)
    bay_compliant_count = len(filtered_df[filtered_df[BAY_FLOAT] >= BAY_HOUR_MANDATE])
    bay_maintenance_percentage = round((bay_compliant_count / total_employees) * 100, 1) if total_employees > 0 else 0
    
    # Prepare card data with hover information
    cards_data = {
        'total_employees': {
            'value': total_employees,
            'title': 'Total Employees',
            'hover_info': f"Filtered employees count"
        },
        'avg_in_time': {
            'value': avg_in_time,
            'title': 'Avg. In Time',
            'hover_info': f"Most common check-in time"
        },
        'avg_out_time': {
            'value': avg_out_time,
            'title': 'Avg. Out Time', 
            'hover_info': f"Most common check-out time"
        },
        'avg_office_hrs': {
            'value': avg_office_hrs,
            'title': 'Avg. Office Hours',
            'hover_info': f"Average total office presence"
        },
        'avg_full_day_leaves': {
            'value': avg_full_day_leaves,
            'title': 'Avg. Full Day Leaves',
            'hover_info': f"Average full day leaves per employee"
        },
        'avg_half_day_leaves': {
            'value': avg_half_day_leaves,
            'title': 'Avg. Half Day Leaves',
            'hover_info': f"Average half day leaves per employee"
        },
        'unbilled_count': {
            'value': unbilled_count,
            'title': 'Unbilled Resources',
            'hover_info': f"IDs: {', '.join(unbilled_ids) if unbilled_ids else 'None'}"
        },
        'unallocated_count': {
            'value': unallocated_count,
            'title': 'Unallocated Resources',
            'hover_info': f"IDs: {', '.join(unallocated_ids) if unallocated_ids else 'None'}"
        },
        'bay_maintenance': {
            'value': f"{bay_maintenance_percentage}%",
            'title': 'Bay Hours Compliance',
            'hover_info': f"{bay_compliant_count}/{total_employees} employees meet 7h mandate"
        }
    }
    
    # Generate charts
    charts_data = generate_overview_charts(filtered_df)
    
    # Prepare high and low bay hours tables - FIXED: Use the actual index (Employee ID)
    high_bay_employees = filtered_df.nlargest(5, BAY_FLOAT)[[COL_DESIGNATION, COL_ACCOUNT, BAY_FLOAT]].copy()
    high_bay_employees['Bay_Hours_Display'] = high_bay_employees[BAY_FLOAT].apply(float_to_h_mm)
    high_bay_employees = high_bay_employees.reset_index().rename(columns={COL_FAKEID: 'Employee ID'})
    
    low_bay_employees = filtered_df.nsmallest(5, BAY_FLOAT)[[COL_DESIGNATION, COL_ACCOUNT, BAY_FLOAT]].copy()
    low_bay_employees['Bay_Hours_Display'] = low_bay_employees[BAY_FLOAT].apply(float_to_h_mm)
    low_bay_employees = low_bay_employees.reset_index().rename(columns={COL_FAKEID: 'Employee ID'})
    
    # Get unique values for filters
    unique_designations = ['All'] + sorted(filtered_df[COL_DESIGNATION].unique().tolist())
    unique_accounts = ['All'] + sorted(filtered_df[COL_ACCOUNT].unique().tolist())
    
    return render_template("overview.html",
                         cards_data=cards_data,
                         charts_data=charts_data,
                         high_bay_employees=high_bay_employees.to_dict('records'),
                         low_bay_employees=low_bay_employees.to_dict('records'),
                         unbilled_ids=unbilled_ids,
                         unallocated_ids=unallocated_ids,
                         filters={
                             'designations': unique_designations,
                             'accounts': unique_accounts,
                             'selected_designation': selected_designation,
                             'selected_account': selected_account
                         },
                         is_initial_load=False)

def generate_overview_charts(filtered_df):
    """Generate charts for the overview page"""
    charts = {}
    
    if filtered_df.empty:
        # Return empty chart JSONs if no data
        charts['time_metrics_bar'] = "{}"
        charts['productivity_scatter'] = "{}"
        charts['compliance_distribution'] = "{}"
        return charts
    
    # Chart 1: Time Metrics Bar Chart (Bay, Break, OOO, Cafe Hours)
    try:
        avg_bay = filtered_df[BAY_FLOAT].mean()
        avg_break = filtered_df[BREAK_FLOAT].mean()
        avg_ooo = filtered_df[OOO_FLOAT].mean()
        avg_cafe = filtered_df[CAFE_FLOAT].mean()
        
        metrics = ['Bay Hours', 'Break Hours', 'Out of Office', 'Cafeteria']
        values = [avg_bay, avg_break, avg_ooo, avg_cafe]
        display_values = [float_to_h_mm(v) for v in values]
        
        # Create bar chart
        bar_trace = {
            'x': metrics,
            'y': values,
            'type': 'bar',
            'marker': {
                'color': [ACCENT_HEADER, ACCENT_AMBER, ACCENT_RED, ACCENT_BLUE_MU],
                'line': {'width': 2, 'color': 'darkgray'}
            },
            'text': display_values,
            'textposition': 'auto',
            'hovertemplate': '<b>%{x}</b><br>Average: %{text}<extra></extra>'
        }
        
        bar_layout = {
            'title': 'Average Time Distribution Across Key Metrics',
            'xaxis': {'title': 'Time Metrics'},
            'yaxis': {'title': 'Average Hours'},
            'height': 400,
            'showlegend': False,
            'template': 'plotly_white'
        }
        
        charts['time_metrics_bar'] = json.dumps({'data': [bar_trace], 'layout': bar_layout})
        
    except Exception as e:
        print(f"Error creating time metrics bar chart: {e}")
        charts['time_metrics_bar'] = "{}"
    
    # Chart 2: Productivity vs Break Time Scatter Plot
    try:
        scatter_trace = {
            'x': filtered_df[BAY_FLOAT].tolist(),
            'y': filtered_df[BREAK_FLOAT].tolist(),
            'mode': 'markers',
            'type': 'scatter',
            'marker': {
                'size': 10,
                'color': filtered_df[BAY_FLOAT].apply(
                    lambda x: ACCENT_GREEN if x >= BAY_HOUR_MANDATE else 
                             ACCENT_AMBER if x >= CRITICAL_BAY_LIMIT else ACCENT_RED
                ).tolist(),
                'line': {'width': 1, 'color': 'darkgray'}
            },
            'text': filtered_df.index.tolist(),
            'hovertemplate': '<b>Employee: %{text}</b><br>Bay Hours: %{x:.1f}<br>Break Hours: %{y:.1f}<extra></extra>'
        }
        
        scatter_layout = {
            'title': 'Productivity vs Break Time Analysis',
            'xaxis': {'title': 'Bay Hours (Productivity)'},
            'yaxis': {'title': 'Break Hours (Non-Productive Time)'},
            'height': 400,
            'showlegend': False,
            'shapes': [
                {
                    'type': 'line',
                    'x0': BAY_HOUR_MANDATE, 'x1': BAY_HOUR_MANDATE,
                    'y0': 0, 'y1': filtered_df[BREAK_FLOAT].max() if not filtered_df.empty else 5,
                    'line': {'color': ACCENT_GREEN, 'width': 2, 'dash': 'dash'}
                }
            ]
        }
        
        charts['productivity_scatter'] = json.dumps({'data': [scatter_trace], 'layout': scatter_layout})
        
    except Exception as e:
        print(f"Error creating productivity scatter chart: {e}")
        charts['productivity_scatter'] = "{}"
    
    # Chart 3: Compliance Distribution Pie Chart
    try:
        # Calculate compliance distribution
        compliant_count = len(filtered_df[filtered_df[BAY_FLOAT] >= BAY_HOUR_MANDATE])
        warning_count = len(filtered_df[(filtered_df[BAY_FLOAT] >= CRITICAL_BAY_LIMIT) & (filtered_df[BAY_FLOAT] < BAY_HOUR_MANDATE)])
        critical_count = len(filtered_df[filtered_df[BAY_FLOAT] < CRITICAL_BAY_LIMIT])
        
        labels = ['Compliant (≥7h)', 'Warning (5-7h)', 'Critical (<5h)']
        values = [compliant_count, warning_count, critical_count]
        colors = [ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED]
        
        # Create pie chart
        pie_trace = {
            'labels': labels,
            'values': values,
            'type': 'pie',
            'marker': {'colors': colors},
            'hole': 0.4,
            'textinfo': 'percent+label',
            'hovertemplate': '<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        }
        
        pie_layout = {
            'title': 'Bay Hours Compliance Distribution',
            'height': 400,
            'showlegend': True,
            'annotations': [{
                'text': 'Compliance',
                'x': 0.5, 'y': 0.5,
                'font': {'size': 16},
                'showarrow': False
            }]
        }
        
        charts['compliance_distribution'] = json.dumps({'data': [pie_trace], 'layout': pie_layout})
            
    except Exception as e:
        print(f"Error creating compliance distribution chart: {e}")
        charts['compliance_distribution'] = "{}"
    
    return charts



@app.route("/")
def index():
    return render_template("index.html")


# Add this route to handle overview page redirection
@app.route("/overview_redirect")
def overview_redirect():
    return redirect(url_for("overview"))

@app.route("/employee_selection_redirect")
def employee_selection_redirect():
    # Simply redirect to the Employee Deep Dive page
    return redirect(url_for("employee"))

@app.route("/account_selection_redirect")
def account_selection_redirect():
    # Redirect to Account Analysis page
    return redirect(url_for("account"))

    
if __name__ == "__main__":
    # Create a dummy data directory if not exists
    if not os.path.exists('data'):
        os.makedirs('data')
        
    # Re-read global data and run models for the app instance
    df_raw = load_data()
    df, kmeans, _ = run_clustering_and_anomaly_detection(df_raw.copy())
    
    app.run(debug=True)