# Credit Risk Analysis & Portfolio Monitoring Dashboard
## Project Overview
This project analyzes loan portfolio performance, credit default behavior, repayment trends, and collection efficiency to identify high-risk segments and support proactive portfolio management decisions.
The analysis combines:
- Exploratory Data Analysis (EDA)
- Portfolio risk monitoring
- Branch performance evaluation
- Customer repayment behavior analysis
- Predictive risk indicators
- Collection recovery insights
- Data visualization dashboards
---
# Business Problem
Financial institutions face increasing challenges in:
- Managing rising defaults
- Monitoring portfolio exposure
- Identifying high-risk borrowers early
- Improving collection efficiency
- Reducing non-performing loans (NPLs)
This project aims to:
- Detect risky loan segments
- Monitor outstanding balances
- Track repayment behavior
- Improve collection prioritization
- Support data-driven lending decisions
---
# Objectives
- Analyze portfolio exposure across branches
- Identify drivers of default risk
- Evaluate repayment performance
- Detect concentration risks
- Support collection strategy improvements
- Build predictive indicators for potential defaults
---
# Dataset Description
The dataset includes:
- Customer loan details
- Outstanding balances
- Amount paid
- Loan cycles
- Branch information
- Loan officer performance
- Repayment status
- Active vs defaulted clients
---
# Tools & Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Seaborn
- Jupyter Notebook
- Scikit-learn
---
# Key Analysis Performed
- Outstanding balance distribution
- Default trend analysis
- Branch risk comparison
- Loan cycle analysis
- Recovery performance
- Portfolio concentration analysis
- Correlation analysis
- Collection efficiency monitoring
---
# Key Insights
## Portfolio Risk
- High outstanding balances indicate elevated exposure risk.
- Certain branches contribute disproportionately to defaults.
## Collection Performance
- Partial-paying clients represent major recovery opportunities.
- Faster follow-up significantly improves closure rates.
## Customer Behavior
- Higher loan cycles often show improved repayment reliability.
- Newly onboarded clients exhibit higher early-stage risk.
---
# Recommendations
## Immediate Actions
- Intensify follow-up on partial-payment clients
- Prioritize high outstanding balances
- Escalate long overdue accounts
## Branch-Level Actions
- Increase monitoring on high-risk branches
- Improve collection accountability
- Review underwriting quality
---
# Future Improvements
- Build real-time dashboard
- Deploy predictive default model
- Add automated risk alerts
- Integrate customer scoring system
- Implement SHAP explainability
---
# Project Structure
Credit-risk-analysis/
│
├── data/
├── notebooks/
├── images/
├── reports/
├── models/
├── README.md
└── requirements.txt
---
# How to Run
```bash
# Clone repository
git clone https://github.com/johnthuo-analytics/Credit-risk-analysis.git
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook
