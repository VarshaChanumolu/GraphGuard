# GraphGuard — Tableau Dashboard Spec

Connect each sheet to the CSV in the same folder as this file.

## Sheet 1: Model Comparison (model_comparison.csv)
**Chart type:** Grouped bar chart
**Rows:** run_name  **Columns:** test_auc, test_pr_auc (side by side)
**Color:** stage (baseline = teal, gnn = coral)
**Title:** "Model Performance: AUC and PR-AUC by Run"
**Note:** PR-AUC is the more honest metric at 3.5% fraud rate -- label it prominently.

## Sheet 2: Fraud Score Distribution (transactions_scored.csv)
**Chart type:** Histogram of fraud_probability, colored by actual_label (0=legit, 1=fraud)
**Title:** "Fraud Probability Distribution -- Legitimate vs Actual Fraud"
**Insight to highlight:** Fraud transactions cluster near probability 1.0;
a bimodal distribution here confirms the model separates the classes well.

## Sheet 3: SHAP Global Feature Importance (shap_global.csv)
**Chart type:** Horizontal bar chart, sorted descending by mean_abs_shap
**Color:** is_graph_feature (True = different color to highlight graph contribution)
**Title:** "Global Feature Importance (Mean |SHAP Value|)"
**Note:** Graph features (degree, clustering_coeff, component_size) should be
visually distinct -- these are the features the graph construction stage added.

## Sheet 4: GNN Training Curve (gnn_training.csv)
**Chart type:** Line chart
**Filter:** metric IN ('val_auc', 'val_pr_auc', 'train_loss') -- show all three
**X-axis:** epoch  **Y-axis:** value  **Color:** metric
**Mark the best checkpoint epoch (14) with a reference line.**
**Title:** "GNN Training Curve -- Val PR-AUC peaked at epoch 14 (0.4677)"

## Dashboard Assembly
Arrange all four sheets on a single dashboard in a 2x2 grid.
Add a text box title: "GraphGuard: Production Fraud Detection System"
Add a subtitle: "IEEE-CIS dataset | 590,540 transactions | 3.5% fraud rate"
Publish to Tableau Public (File -> Save to Tableau Public).
