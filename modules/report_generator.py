"""Builds the downloadable PDF report from whichever signals the session holds.
The overall-verdict ranking here mirrors the one in templates/verdict.html.
"""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

TEAL = colors.HexColor('#2AB38E')
DARK = colors.HexColor('#0D1B2A')
RED = colors.HexColor('#FF4757')
MUTED = colors.HexColor('#6B7686')


def _compute_overall(signal1, signal1b, signal2, signal4):
    """Mirrors verdict.html's ranking. Coherence ranks by (100 - score) since high coherence is safe."""
    scores = []
    if signal1:
        scores.append((signal1['score'], signal1['verdict'], signal1['score']))
    if signal1b:
        scores.append((signal1b['score'], signal1b['verdict'], signal1b['score']))
    if signal2:
        scores.append((signal2['score'], signal2['verdict'], signal2['score']))
    if signal2 and signal2.get('manipulation_verdict'):
        scores.append((signal2['manipulation_score'], signal2['manipulation_verdict'], signal2['manipulation_score']))
    if signal4:
        scores.append((100 - signal4['score'], signal4['verdict'], signal4['score']))

    if not scores:
        return None
    worst = max(scores, key=lambda s: s[0])
    return {'verdict': worst[1], 'score': worst[2], 'active_count': len(scores)}


def generate_report_pdf(signal1, signal1b, signal2, signal4):
    """Returns a BytesIO of the rendered PDF. Signals left as None render as "Not tested"."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DGTitle', parent=styles['Title'], textColor=DARK, fontSize=22, spaceAfter=4)
    subtitle_style = ParagraphStyle('DGSubtitle', parent=styles['Normal'], textColor=MUTED, fontSize=10, spaceAfter=20)
    h2_style = ParagraphStyle('DGH2', parent=styles['Heading2'], textColor=DARK, fontSize=14, spaceBefore=16, spaceAfter=6)
    body_style = ParagraphStyle('DGBody', parent=styles['Normal'], fontSize=10, leading=14)
    muted_style = ParagraphStyle('DGMuted', parent=styles['Normal'], fontSize=10, textColor=MUTED, leading=14)
    # A Paragraph in a cell ignores the table's TEXTCOLOR, so the header needs its own style.
    header_style = ParagraphStyle('DGHeader', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.white, fontName='Helvetica-Bold')

    story = []
    story.append(Paragraph('DeepGuard Analysis Report', title_style))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        subtitle_style,
    ))

    overall = _compute_overall(signal1, signal1b, signal2, signal4)
    story.append(Paragraph('Overall Verdict', h2_style))
    if overall:
        verdict_color = TEAL if overall['verdict'] in ('Real', 'Non-manipulative', 'Coherent') else RED
        overall_style = ParagraphStyle('DGOverall', parent=styles['Heading1'], textColor=verdict_color, fontSize=18, spaceAfter=4)
        story.append(Paragraph(f"{overall['verdict']} &mdash; {overall['score']}% confidence", overall_style))
        story.append(Paragraph(
            f"Based on {overall['active_count']} active signal"
            f"{'s' if overall['active_count'] != 1 else ''} out of 5. "
            "Full weighted fusion across all 5 signals not yet implemented - "
            "this is the single most concerning signal among those tested.",
            muted_style,
        ))
    else:
        story.append(Paragraph('No signals have been tested yet in this session.', muted_style))

    story.append(Paragraph('Signal Breakdown', h2_style))

    def verdict_cell(verdict):
        if verdict is None:
            return Paragraph('Not tested', muted_style)
        color = TEAL if verdict in ('Real', 'Non-manipulative', 'Coherent') else RED
        return Paragraph(f'<font color="{color.hexval()}"><b>{verdict}</b></font>', body_style)

    rows = [[
        Paragraph('SIGNAL', header_style),
        Paragraph('VERDICT', header_style),
        Paragraph('SCORE', header_style),
        Paragraph('DETAILS', header_style),
    ]]

    def add_row(name, verdict, score, details):
        rows.append([
            Paragraph(name, body_style),
            verdict_cell(verdict),
            Paragraph(f'{score}%' if score is not None else '--', body_style),
            Paragraph(details or '', muted_style),
        ])

    add_row(
        'Video Face (deepfake detection)',
        signal1['verdict'] if signal1 else None,
        signal1['score'] if signal1 else None,
        signal1.get('explanation') if signal1 else 'Not tested in this session.',
    )
    add_row(
        'Image Check (AI-generated / deepfake)',
        signal1b['verdict'] if signal1b else None,
        signal1b['score'] if signal1b else None,
        signal1b.get('explanation') if signal1b else 'Not tested in this session.',
    )
    add_row(
        'Voice Clone',
        signal2['verdict'] if signal2 else None,
        signal2['score'] if signal2 else None,
        signal2.get('explanation') if signal2 else 'Not tested in this session.',
    )
    add_row(
        'Manipulation (voice script)',
        signal2.get('manipulation_verdict') if signal2 else None,
        signal2.get('manipulation_score') if signal2 else None,
        'Coercive/manipulative language scoring, from the same audio as Voice Clone.' if signal2 and signal2.get('manipulation_verdict') else 'Not tested in this session.',
    )
    add_row(
        'Caption Coherence',
        signal4['verdict'] if signal4 else None,
        signal4['score'] if signal4 else None,
        'How well a post\'s caption matches what\'s actually said in its audio.' if signal4 else 'Not tested in this session.',
    )

    # Verdict/Score widened so "Non-manipulative" and the "Score" header stop wrapping.
    table = Table(rows, colWidths=[1.4 * inch, 1.3 * inch, 0.9 * inch, 2.4 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DDDDDD')),
        ('LINEBELOW', (0, 0), (-1, 0), 1, DARK),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F7FA')]),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(table)

    story.append(Spacer(1, 24))
    story.append(Paragraph(
        'Generated by DeepGuard - a multi-signal deepfake and manipulation detection tool. '
        'This report reflects the results present in this browser session at the time of download.',
        muted_style,
    ))

    doc.build(story)
    buf.seek(0)
    return buf
