from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"
OUTPUT = REPORTS / "freight_rate_assessment_report.pdf"
DECEMBER_CHART = ROOT / "scorer_results" / "candidate_december.png"
FEATURE_CHART = REPORTS / "feature_importance.png"

NAVY = colors.HexColor("#102A43")
TEAL = colors.HexColor("#0B6670")
GOLD = colors.HexColor("#E4A936")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#5C6B73")
PALE = colors.HexColor("#F2F6F7")
LINE = colors.HexColor("#D7E2E5")
WHITE = colors.white


def build_feature_chart() -> None:
    features = pd.read_csv(ARTIFACTS / "final_feature_importance.csv").head(10).iloc[::-1]
    figure, axis = plt.subplots(figsize=(8.2, 4.1), dpi=180)
    axis.barh(features["feature"], features["importance"], color="#0B6670")
    axis.set_xlabel("Mean CatBoost importance")
    axis.grid(axis="x", color="#D7E2E5", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#9FB3BA")
    axis.tick_params(axis="y", length=0)
    figure.tight_layout()
    figure.savefig(FEATURE_CHART, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=29,
            leading=34,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=sample["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=18,
            textColor=MUTED,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=23,
            textColor=NAVY,
            spaceBefore=2,
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=16,
            textColor=TEAL,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=14,
            textColor=INK,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8.1,
            leading=11,
            textColor=MUTED,
        ),
        "metric": ParagraphStyle(
            "Metric",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=20,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = letter
    if doc.page > 1:
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawRightString(width - 0.65 * inch, 0.38 * inch, f"{doc.page}")
    canvas.restoreState()


def metric_card(value: str, label: str, style: dict[str, ParagraphStyle]) -> Table:
    table = Table(
        [[Paragraph(value, style["metric"])], [Paragraph(label, style["metric_label"])]],
        colWidths=[1.5 * inch],
        rowHeights=[0.37 * inch, 0.26 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def styled_table(data: list[list], widths: list[float], header: bool = True) -> Table:
    header_style = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=7.6,
        leading=9.4,
        textColor=WHITE,
    )
    cell_style = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.2,
        textColor=INK,
    )
    wrapped = [
        [Paragraph(str(value), header_style if header and row_index == 0 else cell_style) for value in row]
        for row_index, row in enumerate(data)
    ]
    table = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    for row in range(1 if header else 0, len(data)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), PALE))
    table.setStyle(TableStyle(commands))
    return table


def p(text: str, style: dict[str, ParagraphStyle], name: str = "body") -> Paragraph:
    return Paragraph(text, style[name])


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    build_feature_chart()
    metric_payload = json.loads((ARTIFACTS / "validation_metrics.json").read_text(encoding="utf-8"))
    model = metric_payload["model"]
    baseline = metric_payload["baseline"]
    equipment = pd.read_csv(ARTIFACTS / "error_by_equipment.csv")
    style = styles()

    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.82 * inch,
        bottomMargin=0.58 * inch,
        title="Freight Rate Prediction Assessment",
        author="eraj22",
        subject="Machine learning approach, validation, and final December chart",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPageEnd=header_footer)])
    story = []

    story.extend(
        [
            Spacer(1, 0.65 * inch),
            Table([["MACHINE LEARNING ENGINEER ASSESSMENT"]], colWidths=[3.55 * inch], style=[
                ("BACKGROUND", (0, 0), (-1, -1), TEAL),
                ("TEXTCOLOR", (0, 0), (-1, -1), WHITE),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]),
            Spacer(1, 0.35 * inch),
            p("Freight Rate<br/>Prediction", style, "title"),
            p("A leakage-aware forecasting pipeline for 12,000 future loads", style, "subtitle"),
            Spacer(1, 0.55 * inch),
            Table(
                [
                    [p("Candidate", style, "small"), p("GitHub", style, "small")],
                    [p("eraj22", style, "h2"), p("github.com/eraj22", style, "h2")],
                    [p("Development data", style, "small"), p("Final scoring data", style, "small")],
                    [p("48,000 loads | Jan-Oct 2025", style, "body"), p("12,000 loads | Nov-Dec 2025", style, "body")],
                ],
                colWidths=[3.35 * inch, 3.35 * inch],
                style=[
                    ("BACKGROUND", (0, 0), (-1, -1), PALE),
                    ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                    ("INNERGRID", (0, 0), (-1, -1), 0.45, LINE),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ],
            ),
            Spacer(1, 1.0 * inch),
            p(
                "This report documents the data audit, chronological validation design, model choice, final holdout performance, and the required fixed December prediction chart.",
                style,
                "body",
            ),
            Spacer(1, 0.3 * inch),
            Table([["Reproducible code", "Exact ID alignment", "Scorer validated"]], colWidths=[2.2 * inch] * 3, style=[
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, -1), WHITE),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Spacer(1, 0.28 * inch),
            p("Executive summary", style, "h1"),
            p(
                "The final solution is a three-seed CatBoost ensemble trained on the logarithm of posted rate. It was selected because the dataset combines continuous signals, high-cardinality route categories, missing values, unseen lanes, nonlinear distance effects, and rare extreme rates.",
                style,
            ),
            Spacer(1, 0.08 * inch),
            Table(
                [[
                    metric_card(f"${model['mae']:,.2f}", "OCTOBER MAE", style),
                    metric_card(f"{model['mape_pct']:.2f}%", "OCTOBER MAPE", style),
                    metric_card(f"{model['r2']:.3f}", "OCTOBER R-SQUARED", style),
                    metric_card(f"{metric_payload['mae_improvement_pct']:.1f}%", "MAE IMPROVEMENT", style),
                ]],
                colWidths=[1.65 * inch] * 4,
                style=[("VALIGN", (0, 0), (-1, -1), "TOP")],
            ),
            Spacer(1, 0.22 * inch),
            p("Decision summary", style, "h2"),
            p(
                "- Use a chronological split because the scoring rows occur after the labeled period.<br/>"
                "- Use CatBoost to preserve categorical route information while remaining robust to unseen combinations.<br/>"
                "- Predict log1p(posted_rate) to limit the leverage of rare rates above $10,000 without deleting them.<br/>"
                "- Average three deterministic seeds to reduce model variance.<br/>"
                "- Train the final ensemble on all 48,000 labeled rows after the October holdout was evaluated.",
                style,
            ),
            p("Validation design", style, "h2"),
            styled_table(
                [
                    ["Stage", "Training window", "Evaluation window", "Purpose"],
                    ["Development", "2025-01-01 to 2025-08-31", "September 2025", "Model and feature selection"],
                    ["Final holdout", "2025-01-01 to 2025-09-30", "October 2025", "One-month forward simulation"],
                    ["Final fit", "2025-01-01 to 2025-10-31", "November-December 2025", "Submission predictions"],
                ],
                [1.05 * inch, 1.75 * inch, 1.45 * inch, 2.45 * inch],
            ),
            Spacer(1, 0.13 * inch),
            p(
                "Why not random cross-validation? Random folds would mix future observations into past training windows. The chronological design better represents the deployment decision and provides a more credible estimate of forward performance.",
                style,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Spacer(1, 0.28 * inch),
            p("Data audit", style, "h1"),
            p(
                "The labeled data contains 48,000 loads from January through October 2025. The submission data contains 12,000 loads from November through December. IDs are unique and the rows are not duplicated.",
                style,
            ),
            styled_table(
                [
                    ["Check", "Labeled", "Final", "Treatment"],
                    ["Duplicate rows", "0", "0", "No action"],
                    ["Duplicate load_id", "0", "0", "Hard validation failure"],
                    ["Missing weight", "300", "165", "Missing flag and native null handling"],
                    ["Negative weight", "292", "145", "Absolute value plus anomaly flag"],
                    ["Missing market index", "374", "249", "Missing flag and native null handling"],
                    ["Unseen cities", "-", "8", "Use coordinates and other numeric features"],
                    ["Unseen lanes", "-", "736", "Do not rely on lane memorization"],
                ],
                [1.45 * inch, 0.82 * inch, 0.82 * inch, 3.55 * inch],
            ),
            p("Target behavior", style, "h2"),
            p(
                "Posted rate is right-skewed. The median is $2,030.76, the mean is $2,373.98, the 99th percentile is $5,972.83, and the maximum is $25,533. Extreme values were retained because the brief does not identify them as bad records. A log target makes the model robust while preserving every row.",
                style,
            ),
            p("Generalization risk", style, "h2"),
            p(
                "Eight cities do not occur in the labeled data, and 736 final lanes are unseen. The feature set therefore combines lane identity with distance, coordinates, equipment, weight, market and quote signals, and calendar variables. An unseen lane can still be priced from its physical and market characteristics.",
                style,
            ),
            p("Leakage controls", style, "h2"),
            p(
                "Feature construction never uses posted_rate. All transformations are row-local or based only on training history. The fixed December scenario does not contain market index, quote signal, or coordinates, so coordinates are mapped from labeled city history and the two market signals use the most recent 30-day training medians. Those assumptions remain fixed while only the date changes.",
                style,
            ),
            Spacer(1, 0.18 * inch),
            Table(
                [[p("Source: provided train_test.csv, validation.csv, and assessment brief. Audit counts are generated by the submitted pipeline.", style, "small")]],
                colWidths=[6.7 * inch],
                style=[
                    ("BACKGROUND", (0, 0), (-1, -1), PALE),
                    ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Spacer(1, 0.28 * inch),
            Table(
                [[p("Model and feature engineering", style, "h1")]],
                colWidths=[6.7 * inch],
                style=TableStyle(
                    [
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 0),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ]
                ),
            ),
            p(
                "CatBoost is a strong fit for this problem because it learns nonlinear interactions, processes categorical features directly, and handles missing numeric values. The final ensemble uses depth 8, 400 boosting iterations, learning rate 0.07, L2 regularization 8, and seeds 42, 137, and 2026.",
                style,
            ),
            p("Feature groups", style, "h2"),
            styled_table(
                [
                    ["Group", "Examples", "Reason"],
                    ["Route", "pickup, delivery, lane, coordinates", "Capture geographic and directional effects"],
                    ["Load", "distance, equipment, cleaned weight", "Represent the physical movement"],
                    ["Market", "market index, quote signal", "Represent observed price conditions"],
                    ["Interactions", "distance x quote, distance x market", "Convert per-mile signals into load-scale effects"],
                    ["Calendar", "month, weekday, day of year, cyclic terms", "Capture seasonal and weekly patterns"],
                    ["Quality flags", "weight missing, negative weight, market missing", "Preserve information about imperfect records"],
                ],
                [1.05 * inch, 2.35 * inch, 3.25 * inch],
            ),
            p("Top feature importance", style, "h2"),
            Image(str(FEATURE_CHART), width=6.55 * inch, height=3.28 * inch),
            p(
                "Importance is averaged across the final three models. Distance and its nonlinear or market interactions dominate, while equipment and geographic displacement provide additional structure. Importance is descriptive, not causal.",
                style,
                "small",
            ),
            PageBreak(),
        ]
    )

    comparison_rows = [
        ["Metric", "Final ensemble", "Rate-per-mile baseline", "Change"],
        ["MAE", f"${model['mae']:,.2f}", f"${baseline['mae']:,.2f}", f"-{metric_payload['mae_improvement_pct']:.1f}%"],
        ["RMSE", f"${model['rmse']:,.2f}", f"${baseline['rmse']:,.2f}", f"-{(baseline['rmse']-model['rmse'])/baseline['rmse']*100:.1f}%"],
        ["MAPE", f"{model['mape_pct']:.2f}%", f"{baseline['mape_pct']:.2f}%", f"-{baseline['mape_pct']-model['mape_pct']:.2f} pp"],
        ["Median absolute error", f"${model['median_absolute_error']:,.2f}", f"${baseline['median_absolute_error']:,.2f}", f"-{(baseline['median_absolute_error']-model['median_absolute_error'])/baseline['median_absolute_error']*100:.1f}%"],
        ["R-squared", f"{model['r2']:.3f}", f"{baseline['r2']:.3f}", f"+{model['r2']-baseline['r2']:.3f}"],
    ]
    equipment_rows = [["Equipment", "Rows", "MAE", "Median AE"]] + [
        [row.equipment, f"{int(row.rows):,}", f"${row.mae:,.2f}", f"${row.median_ae:,.2f}"]
        for row in equipment.itertuples(index=False)
    ]
    story.extend(
        [
            Spacer(1, 0.28 * inch),
            p("Final holdout results", style, "h1"),
            p(
                "The final holdout contains 4,853 October loads that were not used to fit the evaluated models. The comparison baseline multiplies each distance by the median historical rate per mile from the training window.",
                style,
            ),
            styled_table(comparison_rows, [1.65 * inch, 1.55 * inch, 1.8 * inch, 1.35 * inch]),
            p("Error by equipment", style, "h2"),
            styled_table(equipment_rows, [1.65 * inch, 1.05 * inch, 1.55 * inch, 1.55 * inch]),
            p("Interpretation", style, "h2"),
            p(
                "Typical-load accuracy is strong: the median absolute error is $39.73 and 90% of absolute errors are below $166.64. RMSE remains much larger than MAE because a small number of extreme rates create large squared errors. This gap supports the log-target design and continued human review for unusually expensive loads.",
                style,
            ),
            p("Reproducibility", style, "h2"),
            p(
                "The repository provides fixed dependency ranges, deterministic seeds, schema checks, output validation, unit tests, and GitHub Actions CI. The provided scorer successfully validates the 12,000 final IDs and all 31 fixed December rows.",
                style,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Spacer(1, 0.28 * inch),
            Table(
                [[p("Fixed December prediction", style, "h1")]],
                colWidths=[6.7 * inch],
                style=TableStyle(
                    [
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 0),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ]
                ),
            ),
            p(
                "The chart below is generated by the employer-provided score.py after validating the completed December input file. Pickup, delivery, distance, equipment, and weight remain fixed. Only the date changes from 2025-12-01 through 2025-12-31.",
                style,
            ),
            Spacer(1, 0.05 * inch),
            Image(str(DECEMBER_CHART), width=6.75 * inch, height=2.58 * inch),
            Spacer(1, 0.12 * inch),
            p("Operational assumptions", style, "h2"),
            p(
                "- Lexington pickup and Fort Wayne delivery coordinates use median city coordinates from the labeled data.<br/>"
                "- Missing December market index and quote signal use the latest 30-day medians available in the labeled data.<br/>"
                "- The resulting date-only variation reflects learned calendar effects and does not assume knowledge of future market signals.",
                style,
            ),
            PageBreak(),
            Spacer(1, 0.28 * inch),
            p("Production considerations", style, "h1"),
            p(
                "Monitor MAE, MAPE, and error by equipment and lane after actual rates arrive. Retrain when the market regime or feature distributions drift. Keep a manual review threshold for predictions outside the observed rate and distance ranges. The model supports pricing analysis; it should not replace commercial judgment for exceptional loads.",
                style,
            ),
            p("Repository run sequence", style, "h2"),
            styled_table(
                [
                    ["Step", "Command"],
                    ["1", "python -m pip install -r requirements.txt"],
                    ["2", "python evaluate.py"],
                    ["3", "python train_predict.py"],
                    ["4", "python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv"],
                    ["5", "python -m unittest discover -s tests -v"],
                ],
                [0.55 * inch, 6.1 * inch],
            ),
            Spacer(1, 0.16 * inch),
            p(
                "Final artifacts: validation_predictions.csv, data/december_chart_inputs.csv, scorer_results/candidate_december.png, and this report.",
                style,
                "small",
            ),
        ]
    )

    doc.build(story)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
