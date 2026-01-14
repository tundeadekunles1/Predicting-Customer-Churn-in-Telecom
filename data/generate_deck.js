const pptxgen = require("pptxgenjs");

// Output: Telco_Churn_5min_Overview_v2.pptx
// 7 slides total (still within ~5 minutes): 6 core + 1 appendix

function addTitle(slide, title) {
  slide.addText(title, {
    x: 0.6, y: 0.35, w: 12.1, h: 0.8,
    fontFace: "Calibri", fontSize: 34, bold: true,
    color: "1F2937"
  });
}

function addKicker(slide, text) {
  slide.addText(text, {
    x: 0.6, y: 1.2, w: 12.1, h: 0.4,
    fontFace: "Calibri", fontSize: 16, color: "374151"
  });
}

function addSectionHeader(slide, text, y) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.6, y, w: 12.1, h: 0.55,
    fill: { color: "EEF2FF" },
    line: { color: "C7D2FE", width: 1 },
    radius: 0.2
  });
  slide.addText(text, {
    x: 0.85, y: y + 0.11, w: 11.6, h: 0.35,
    fontFace: "Calibri", fontSize: 18, bold: true, color: "1F2937"
  });
}

function addBullets(slide, bullets, x, y, w, h) {
  slide.addText(bullets.map(b => `• ${b}`).join("\n"), {
    x, y, w, h,
    fontFace: "Calibri", fontSize: 20, color: "111827",
    valign: "top",
    lineSpacingMultiple: 1.15
  });
}

function addCallout(slide, title, body, x, y, w, h) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    fill: { color: "F9FAFB" },
    line: { color: "E5E7EB", width: 1 },
    radius: 0.2
  });
  slide.addText(title, {
    x: x + 0.25, y: y + 0.15, w: w - 0.5, h: 0.3,
    fontFace: "Calibri", fontSize: 16, bold: true, color: "1F2937"
  });
  slide.addText(body, {
    x: x + 0.25, y: y + 0.5, w: w - 0.5, h: h - 0.65,
    fontFace: "Calibri", fontSize: 14, color: "374151",
    valign: "top"
  });
}

function addMiniBars(slide, x, y, w, h, labels, values) {
  // values expected 0..1
  const n = Math.min(labels.length, values.length);
  const rowH = h / n;
  for (let i = 0; i < n; i++) {
    const barW = Math.max(0.01, (w - 3.0) * Math.min(1, Math.max(0, values[i])));
    slide.addText(labels[i], {
      x, y: y + i * rowH, w: 2.6, h: rowH,
      fontFace: "Calibri", fontSize: 14, color: "111827", valign: "mid"
    });
    slide.addShape(pptx.ShapeType.roundRect, {
      x: x + 2.8, y: y + i * rowH + 0.12, w: w - 3.0, h: rowH - 0.24,
      fill: { color: "F3F4F6" },
      line: { color: "E5E7EB", width: 1 },
      radius: 0.15
    });
    slide.addShape(pptx.ShapeType.roundRect, {
      x: x + 2.8, y: y + i * rowH + 0.12, w: barW, h: rowH - 0.24,
      fill: { color: "2563EB" },
      line: { color: "2563EB", width: 0 },
      radius: 0.15
    });
  }
}

function addFooter(slide, text) {
  slide.addText(text, {
    x: 0.6, y: 7.15, w: 12.1, h: 0.25,
    fontFace: "Calibri", fontSize: 12, color: "6B7280"
  });
}

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Bilikisu";

// ------------------- Slide 1: Overview -------------------
{
  const s = pptx.addSlide();
  s.background = { color: "FFFFFF" };
  addTitle(s, "Telco Churn Retention Targeting — 5-minute Overview");
  addKicker(s, "Feature engineering → training → evaluation → weighted model selection → Streamlit deployment");
  addSectionHeader(s, "What this project delivers", 1.75);

  addBullets(s, [
    "A churn probability score per customer (0 to 1)",
    "A ranked targeting list (Top 10% or Top 20%) with an optional probability threshold",
    "Operational export (CSV): CustomerID, churn_probability, Action",
    "Metrics that prove value: Baseline, Precision@k, Recall@k, Lift@k"
  ], 0.9, 2.45, 12.0, 3.6);

  addCallout(
    s,
    "One-line business meaning",
    "“We spend retention effort on the customers most likely to leave, instead of guessing.”",
    0.9, 6.1, 12.0, 0.85
  );

  addFooter(s, "5 minutes | Audience: mixed | Version v2");
}

// ------------------- Slide 2: Feature engineering -------------------
{
  const s = pptx.addSlide();
  s.background = { color: "FFFFFF" };
  addTitle(s, "1) Feature engineering: clean, consistent, non-leaky inputs");
  addKicker(s, "Goal: build features that reflect what we know BEFORE churn happens (no ‘answer sheet’).");
  addSectionHeader(s, "Data quality and leakage controls", 1.75);

  addBullets(s, [
    "Fixed missing values (e.g., TotalCharges NaNs) so training and evaluation are valid",
    "Kept label (Churn_Yes) out of model inputs to prevent leakage",
    "One-hot encoding for categories (Contract_*, PaymentMethod_*, InternetService_*, etc.)",
    "Engineered features: charges_ratio, HighSpender, HighChurnRisk (must be train-fitted if threshold-based)"
  ], 0.9, 2.45, 7.3, 3.6);

  addCallout(
    s,
    "Leakage (kid-simple)",
    "Leakage is like peeking at the answer during an exam. It looks great in testing, but fails in real life.",
    8.4, 2.45, 4.5, 2.1
  );

  addCallout(
    s,
    "Key control (important)",
    "If a feature uses a threshold (like top 25% charges), learn that threshold on TRAIN only, then apply to test and production.",
    8.4, 4.75, 4.5, 1.35
  );

  addFooter(s, "Step 1/5");
}

// ------------------- Slide 3: Model training -------------------
{
  const s = pptx.addSlide();
  s.background = { color: "FFFFFF" };
  addTitle(s, "2) Model training: multiple candidates, same split, same schema");
  addKicker(s, "Goal: rank churners above non-churners—especially at the top of the list.");
  addSectionHeader(s, "Candidates trained", 1.75);

  addBullets(s, [
    "Logistic Regression (interpretable baseline)",
    "Logistic Regression (class_weight=balanced)",
    "Random Forest + Random Forest (balanced)",
    "HistGradientBoosting (non-linear model, often strong for top-k ranking)"
  ], 0.9, 2.45, 12.0, 2.6);

  addSectionHeader(s, "Training setup (simple)", 5.2);
  addBullets(s, [
    "Stratified holdout split (keeps churn/non-churn proportion consistent)",
    "Same processed dataset and same feature contract across models",
    "Saved artifacts for deployment (joblib)"
  ], 0.9, 5.85, 12.0, 1.2);

  addFooter(s, "Step 2/5");
}

// ------------------- Slide 4: Model evaluation (Lift@k) -------------------
{
  const s = pptx.addSlide();
  s.background = { color: "FFFFFF" };
  addTitle(s, "3) Model evaluation: Lift@k is the targeting KPI");
  addKicker(s, "Lift@k answers: “Is Top 10% / Top 20% targeting better than random selection?”");
  addSectionHeader(s, "Definitions (kid-simple but correct)", 1.75);

  addBullets(s, [
    "Baseline churn rate: average churn in the evaluation set",
    "Precision@k: churn rate inside the Top k% targeted list",
    "Lift@k = Precision@k ÷ Baseline (how many times better than random)",
    "Recall@k: % of all churners captured in the Top k% list"
  ], 0.9, 2.45, 7.3, 3.4);

  addCallout(
    s,
    "Example",
    "If baseline = 0.265 and Precision@10% = 0.773,\nLift@10% = 0.773 / 0.265 ≈ 2.91×.\nMeaning: Top 10% is ~3× more churn-heavy than average.",
    8.4, 2.45, 4.5, 2.1
  );

  addCallout(
    s,
    "Lift curve behavior (expected)",
    "Lift is highest at very small k (top 1–5%) and decreases as we target more customers.\nThis is normal because we include more lower-risk customers as k grows.",
    8.4, 4.75, 4.5, 1.9
  );

  addFooter(s, "Step 3/5");
}

// ------------------- Slide 5: Weighted scoring (min–max normalization) -------------------
{
  const s = pptx.addSlide();
  s.background = { color: "FFFFFF" };
  addTitle(s, "4) Weighted model scoring: why min–max normalization");
  addKicker(s, "We need one score that reflects business priorities across multiple metrics.");
  addSectionHeader(s, "Problem: metrics are on different numeric scales", 1.75);

  addBullets(s, [
    "Lift@10% and Lift@20% are multipliers (~2–3× here)",
    "PR-AUC is between 0 and 1",
    "If we add raw numbers, one metric can dominate unfairly"
  ], 0.9, 2.45, 7.3, 2.1);

  addSectionHeader(s, "Solution: normalize each metric to 0–1 and apply weights", 4.65);

  addCallout(
    s,
    "Min–max normalization",
    "norm(x) = (x − min) / (max − min)\n0 = worst model on that metric, 1 = best model on that metric.",
    0.9, 5.35, 6.2, 1.15
  );

  addCallout(
    s,
    "Weighted score",
    "Score = 0.60·norm(Lift@10%) + 0.30·norm(Lift@20%) + 0.10·norm(PR-AUC)",
    7.5, 5.35, 5.4, 1.15
  );

  addMiniBars(
    s,
    0.9, 6.65, 12.0, 0.65,
    ["Lift@10% (60%)", "Lift@20% (30%)", "PR-AUC (10%)"],
    [0.60, 0.30, 0.10]
  );

  addFooter(s, "Step 4/5");
}

// ------------------- Slide 6: Deployment + pilot -------------------
{
  const s = pptx.addSlide();
  s.background = { color: "FFFFFF" };
  addTitle(s, "5) Deployment: Streamlit app + controlled retention pilot");
  addKicker(s, "Goal: convert scores into actions safely, then prove business impact.");
  addSectionHeader(s, "What the Streamlit app does", 1.75);

  addBullets(s, [
    "Loads processed dataset (or uploaded file) + loads saved model",
    "Scores churn probability per customer",
    "Builds Top 10% / Top 20% list + optional probability threshold",
    "Exports CSV for operations: CustomerID, churn_probability, Action",
    "Shows Baseline, Precision@k, Recall@k, Lift@k (label fixed to Churn_Yes)"
  ], 0.9, 2.45, 7.3, 3.8);

  addCallout(
    s,
    "Controlled pilot (simple)",
    "Within the targeted list, randomly split customers into:\n• Treatment: receives the retention action\n• Control: receives no action\nCompare churn + ROI to prove incremental value.",
    8.4, 2.45, 4.5, 2.5
  );

  addCallout(
    s,
    "Why this matters",
    "Offline metrics show “the list is good.”\nPilot shows “the list makes money.”",
    8.4, 5.1, 4.5, 1.2
  );

  addFooter(s, "Step 5/5");
}

// ------------------- Slide 7: Appendix (Champion + Challenger + formula as single visual) -------------------
{
  const s = pptx.addSlide();
  s.background = { color: "FFFFFF" };
  addTitle(s, "Appendix: Champion + Challenger decision (single visual)");
  addKicker(s, "Decision uses weighted score with min–max normalization.");

  addCallout(
    s,
    "Champion (Production default)",
    "Logistic Regression (balanced)\nWhy: best overall weighted score (strong Lift@20% + strong PR-AUC + high Lift@10%).\nBest all-round choice for Top 10% and Top 20% targeting.",
    0.9, 2.0, 6.2, 2.1
  );

  addCallout(
    s,
    "Challenger (Pilot comparison)",
    "HistGradientBoosting\nWhy: best Lift@10% (“Top 10% sniper”).\nTest head-to-head vs Champion in pilot for incremental ROI.",
    0.9, 4.25, 6.2, 1.65
  );

  // single formula visual card
  s.addShape(pptx.ShapeType.roundRect, {
    x: 7.5, y: 2.0, w: 5.4, h: 3.9,
    fill: { color: "EEF2FF" },
    line: { color: "C7D2FE", width: 1 },
    radius: 0.25
  });

  s.addText("Weighted score formula", {
    x: 7.8, y: 2.2, w: 4.9, h: 0.4,
    fontFace: "Calibri", fontSize: 18, bold: true, color: "1F2937"
  });

  s.addText("norm(x) = (x − min) / (max − min)", {
    x: 7.8, y: 2.75, w: 4.9, h: 0.5,
    fontFace: "Calibri", fontSize: 20, bold: true, color: "111827"
  });

  s.addText("Score = 0.60·norm(Lift@10%) + 0.30·norm(Lift@20%) + 0.10·norm(PR-AUC)", {
    x: 7.8, y: 3.35, w: 4.9, h: 1.0,
    fontFace: "Calibri", fontSize: 18, color: "111827"
  });

  s.addText(
    "Meaning (kid-simple):\nWe convert each metric to a fair 0–1 grade, then add them using the business weights.",
    {
      x: 7.8, y: 4.55, w: 4.9, h: 1.1,
      fontFace: "Calibri", fontSize: 14, color: "374151"
    }
  );

  addFooter(s, "Appendix (still within the 5-minute deck)");
}

pptx.writeFile({ fileName: "Telco_Churn_5min_Overview_v2.pptx" });
