"""
Automated PDF Executive Summary Generator.
Uses ReportLab to produce publication-grade institutional investment memos
with KPI matrices, asset tables, sentiment analysis, and 98% confidence scenarios.
"""

import io
from datetime import datetime
from typing import Dict, Any
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable
)


def generate_executive_pdf(
    portfolio_data: Dict[str, Any],
    sim_data: Dict[str, Any],
    sentiment_data: Dict[str, Any],
    client_name: str = "Individual Investor",
    horizon_years: int = 10
) -> bytes:
    """
    Generates an executive PDF report and returns the raw bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Brand Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A")
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569")
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6,
        spaceBefore=10
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E293B")
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E293B")
    )

    currency = portfolio_data.get("currency", "₹")
    total_capital = portfolio_data.get("total_capital", 0.0)

    story = []

    # 1. Header Banner
    story.append(Paragraph("SMART ANALYSIS & INVESTMENT SYSTEM", title_style))
    story.append(Paragraph("Executive Quantitative Portfolio & Stochastic Wealth Scenario Memo", subtitle_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"<b>Prepared for:</b> {client_name} &nbsp;|&nbsp; "
        f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y - %H:%M')} &nbsp;|&nbsp; "
        f"<b>Academic Capstone:</b> Final-Year Major Project Architecture",
        body_style
    ))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=12))

    # 2. Executive KPI Summary Cards
    story.append(Paragraph("1. EXECUTIVE QUANTITATIVE SUMMARY", section_heading))
    kpi_data = [
        [
            Paragraph("<b>Total Capital</b>", table_cell_bold),
            Paragraph(f"{currency} {total_capital:,.2f}", table_cell_style),
            Paragraph("<b>Time Horizon</b>", table_cell_bold),
            Paragraph(f"{horizon_years} Years", table_cell_style)
        ],
        [
            Paragraph("<b>Expected CAGR</b>", table_cell_bold),
            Paragraph(f"{portfolio_data.get('expected_cagr', 14.0):.1f}% p.a.", table_cell_style),
            Paragraph("<b>Portfolio Volatility</b>", table_cell_bold),
            Paragraph(f"{portfolio_data.get('expected_volatility', 15.0):.1f}%", table_cell_style)
        ],
        [
            Paragraph("<b>Optimal Sharpe Ratio</b>", table_cell_bold),
            Paragraph(f"{portfolio_data.get('sharpe_ratio', 1.0):.2f}", table_cell_style),
            Paragraph("<b>98% Value-at-Risk (1Y)</b>", table_cell_bold),
            Paragraph(f"{sim_data.get('var_1y_98_pct', 0.0):.1f}% Downside", table_cell_style)
        ]
    ]
    t_kpi = Table(kpi_data, colWidths=[130, 140, 130, 140])
    t_kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 10))

    # 3. Market Sentiment Momentum Section
    story.append(Paragraph("2. LIVE FINANCIAL NEWS SENTIMENT MOMENTUM", section_heading))
    sent_score = sentiment_data.get("composite_score", 0.0)
    sent_label = sentiment_data.get("overall_label", "Neutral")
    cagr_adj = sentiment_data.get("cagr_adjustment_pct", 0.0)
    mood_desc = sentiment_data.get("mood_description", "")

    sent_table_data = [
        [
            Paragraph("<b>Composite Sentiment Score:</b>", table_cell_bold),
            Paragraph(f"{sent_score:+.2f} ({sent_label})", table_cell_style),
            Paragraph("<b>CAGR Momentum Tilt:</b>", table_cell_bold),
            Paragraph(f"{cagr_adj:+.2f}% applied to Monte Carlo", table_cell_style)
        ],
        [
            Paragraph("<b>Market Sentiment Context:</b>", table_cell_bold),
            Paragraph(mood_desc, table_cell_style),
            Paragraph("", table_cell_style),
            Paragraph("", table_cell_style)
        ]
    ]
    t_sent = Table(sent_table_data, colWidths=[140, 150, 120, 130])
    t_sent.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
        ("SPAN", (1, 1), (3, 1)),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_sent)
    story.append(Spacer(1, 10))

    # 4. Recommended Asset Allocation Breakdown Table
    story.append(Paragraph("3. MODERN PORTFOLIO THEORY ALLOCATION BREAKDOWN", section_heading))
    alloc_headers = [
        Paragraph("Asset Description", table_header_style),
        Paragraph("Category", table_header_style),
        Paragraph("Allocated Capital", table_header_style),
        Paragraph("Weight %", table_header_style),
        Paragraph("5Y CAGR", table_header_style)
    ]
    alloc_rows = [alloc_headers]

    # Add stocks
    for s in portfolio_data.get("stocks_picks", []):
        alloc_rows.append([
            Paragraph(f"{s['name']} ({s['symbol']}) - {s.get('quantity', 0)} shares", table_cell_style),
            Paragraph("Direct Equity", table_cell_style),
            Paragraph(f"{currency} {s['allocated_amount']:,.2f}", table_cell_style),
            Paragraph(f"{s.get('weight_pct', 0.0):.1f}%", table_cell_style),
            Paragraph(f"{s.get('cagr_5y', 0.0):.1f}%", table_cell_style)
        ])

    # Add mutual funds
    for m in portfolio_data.get("mfs_picks", []):
        alloc_rows.append([
            Paragraph(m["name"], table_cell_style),
            Paragraph(m.get("category", "Mutual Fund"), table_cell_style),
            Paragraph(f"{currency} {m['allocated_amount']:,.2f}", table_cell_style),
            Paragraph(f"{m.get('weight_pct', 0.0):.1f}%", table_cell_style),
            Paragraph(f"{m.get('cagr_5y', 0.0):.1f}%", table_cell_style)
        ])

    # Add SIP tranche
    if portfolio_data.get("sip_total", 0) > 0:
        sip_info = portfolio_data.get("sip_info", {})
        alloc_rows.append([
            Paragraph(f"Systematic Investment Plan (Monthly: {currency} {sip_info.get('monthly_recommended', 0):,.2f}/mo)", table_cell_style),
            Paragraph("DCA Reserve", table_cell_style),
            Paragraph(f"{currency} {portfolio_data['sip_total']:,.2f}", table_cell_style),
            Paragraph(f"{portfolio_data.get('sip_pct', 0.0):.1f}%", table_cell_style),
            Paragraph(f"{portfolio_data.get('expected_cagr', 14.0):.1f}%", table_cell_style)
        ])

    t_alloc = Table(alloc_rows, colWidths=[200, 110, 100, 65, 65])
    t_alloc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_alloc)
    story.append(Spacer(1, 10))

    # 5. Stochastic Monte Carlo Wealth Projections (98% Statistical Confidence)
    story.append(Paragraph("4. STOCHASTIC WEALTH SCENARIOS (98% STATISTICAL CONFIDENCE)", section_heading))
    scen_headers = [
        Paragraph("Horizon Year", table_header_style),
        Paragraph("Invested Capital", table_header_style),
        Paragraph("Worst Case (2% Bear)", table_header_style),
        Paragraph("Average Expected (50%)", table_header_style),
        Paragraph("Best Case (98% Bull)", table_header_style)
    ]
    scen_rows = [scen_headers]

    df_proj = sim_data.get("df_projections")
    if df_proj is not None and not df_proj.empty:
        # Pick milestone years: 1, 3, 5, 7, 10, end
        years = df_proj["Year"].tolist()
        milestones = [y for y in [1, 3, 5, 10, 15, 20, 25] if y in years]
        if years[-1] not in milestones:
            milestones.append(years[-1])

        for y in milestones:
            row = df_proj[df_proj["Year"] == y].iloc[0]
            scen_rows.append([
                Paragraph(f"Year {int(y)}", table_cell_style),
                Paragraph(f"{currency} {row['Invested_Capital']:,.2f}", table_cell_style),
                Paragraph(f"{currency} {row['Worst_Case']:,.2f}", table_cell_style),
                Paragraph(f"{currency} {row['Average_Case']:,.2f}", table_cell_style),
                Paragraph(f"{currency} {row['Best_Case']:,.2f}", table_cell_style)
            ])

    t_scen = Table(scen_rows, colWidths=[90, 110, 115, 115, 110])
    t_scen.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0FDFA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_scen)
    story.append(Spacer(1, 14))

    # 6. Academic Methodology & Disclaimers
    disclaimer_text = (
        "<b>Quantitative Methodology:</b> This portfolio has been optimized using Markowitz Modern Portfolio Theory "
        "(MPT) with dynamic Sharpe Ratio maximization and quadratic variance minimization. Projections are computed via "
        "5,000 Geometric Brownian Motion (GBM) stochastic Monte Carlo simulation paths bounded by a 2.05-sigma (98%) "
        "two-tailed statistical confidence interval. Past performance does not guarantee future results."
    )
    story.append(Paragraph(disclaimer_text, ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748B")
    )))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
