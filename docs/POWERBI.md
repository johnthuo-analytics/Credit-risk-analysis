# Power BI Dashboard Guide

Power BI Desktop is free (Windows). This guide builds a 3-page dashboard from the files
produced by `python -m src.export_powerbi` in `data/powerbi/`.

Preview images of all three pages are in `images/` (`dashboard_1_overview.png`, `dashboard_2_drivers.png`, `dashboard_3_collections_model.png`). Use them as your target design.

## 1. Load the data
1. Power BI Desktop → **Home → Get data → Text/CSV**.
2. Load `fact_loans.csv`, `dim_branch_summary.csv`, `collection_priority.csv`,
   `model_comparison.csv`, `feature_importance.csv` (from `data/powerbi/`).
3. In Model view, link `fact_loans[Branch]` → `dim_branch_summary[Branch]` (many-to-one).
   Use `dim_branch_summary[Branch]` for slicers and axes.
4. **View → Themes → Browse for themes** → select `docs/credit_risk_theme.json`.

> `Predicted_PD` is out-of-fold: every loan was scored by a model that never saw it.
> `Total_Paid` / `Outstanding_Balance` are monitoring fields, not model inputs.

## 2. DAX measures (Modeling → New measure)
```DAX
Total Loans = COUNTROWS(fact_loans)
Total Disbursed = SUM(fact_loans[Loan_Amount])
Outstanding = SUM(fact_loans[Outstanding_Balance])
Total Paid = SUM(fact_loans[Total_Paid])
Defaults = SUM(fact_loans[Default_Flag])
Default Rate = DIVIDE([Defaults], [Total Loans])
Collection Efficiency = DIVIDE([Total Paid], [Total Paid] + [Outstanding])
Avg Predicted PD = AVERAGE(fact_loans[Predicted_PD])
High Risk Share =
    DIVIDE(CALCULATE([Total Loans], fact_loans[Risk_Band] = "High"), [Total Loans])
Share of Outstanding =
    DIVIDE([Outstanding], CALCULATE([Outstanding], ALL(fact_loans[Branch])))
Share of Defaults =
    DIVIDE([Defaults], CALCULATE([Defaults], ALL(fact_loans[Branch])))
Default Rate vs Portfolio = [Default Rate] - CALCULATE([Default Rate], ALL(fact_loans))
```
Format `Default Rate`, `Collection Efficiency`, `High Risk Share`, `Share of ...` as percentages.

## 3. Page layout

**Page 1 - Executive Overview**
- KPI cards (top row): Total Loans, Total Disbursed, Outstanding, Default Rate, Collection Efficiency.
- Clustered bar: `Default Rate` by `Branch`, with a constant line at the portfolio rate.
- Column chart: `Outstanding` by `Branch`.
- Donut: `Total Loans` by `Risk_Band` (Low green, Medium amber, High red).
- Slicers: Branch, Loan_Cycle band (`Cycle_Band`), `Score_Band`.

**Page 2 - Risk Drivers**
- Column chart: `Default Rate` by `Score_Band` (expect it to fall as score rises).
- Column chart: `Default Rate` by `Cycle_Band`.
- Column chart: `Default Rate` by `Age_Band`.
- Scatter: X = `Loan_to_Income`, Y = `Predicted_PD`, colour = `Risk_Band`.
- Bar: `feature_importance[Importance]` by `Feature`.

**Page 3 - Collections & Model**
- Table: `collection_priority` (Client_ID, Branch, Loan_Officer, Outstanding_Balance, paid_share), sorted by balance.
- Matrix: Branch (rows) x `Risk_Band` (columns), values = `Total Loans`, with conditional formatting.
- Table: `model_comparison` (test_auc, test_gini, test_ks).
- Text box: explain the cost-based threshold and the leakage exclusions.

## 4. Polish checklist
- Use one font (Segoe UI) and the theme colours; no more than 3 colours per page.
- Align visuals to a grid (View → Gridlines + Snap to grid).
- Title every visual with a plain-English message, e.g. "Mombasa defaults run above the portfolio average".
- Hide unused columns (Model view → right-click → Hide in report view).
- Add a 1-line takeaway text box on each page.

## 5. Share it
- Save as `Credit_Risk_Dashboard.pbix`. Keep the `.pbix` out of Git if it embeds real data.
- Screenshot each page into `images/` and link them in the README (this is what visitors will see on GitHub).
- Publishing to the Power BI Service needs a work/school account; a public GitHub README with screenshots works without one.
