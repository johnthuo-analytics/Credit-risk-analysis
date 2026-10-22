# Data

`data/raw/loans.csv` is **synthetic demo data** produced by `python -m src.generate_data`.
It lets the pipeline run end to end but contains no real clients. To analyse real data,
replace the file with your export (same column names) and re-run the pipeline.
Do not commit real client data; `.gitignore` is set up to help.

## Schema

| Column | Type | Description | Available at application time? |
|---|---|---|---|
| Client_ID | str | Unique borrower ID | n/a |
| Age | int | Borrower age | Yes |
| Income | float | Monthly income | Yes |
| Loan_Amount | float | Amount borrowed | Yes |
| Credit_Score | float | Credit rating (300-850) | Yes |
| Branch | str | Issuing branch | Yes |
| Loan_Officer | str | Officer managing the loan | Yes (not used in model: high cardinality) |
| Loan_Cycle | int | Number of previous loans | Yes |
| Outstanding_Balance | float | Remaining unpaid amount | **No** - post-disbursement |
| Total_Paid | float | Amount repaid so far | **No** - post-disbursement |
| Loan_Status | str | Target: `Paid` or `Default` | **No** - outcome |

**Leakage note:** `Outstanding_Balance` and `Total_Paid` describe repayment, which is the
outcome being predicted. They are used for portfolio monitoring and collections only and are
deliberately excluded from the scoring model (enforced by a unit test).
