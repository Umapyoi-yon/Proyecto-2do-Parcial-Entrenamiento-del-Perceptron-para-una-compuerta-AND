"""pdf_generator.py
Generador dinámico del reporte PDF (tabla de épocas + procedimiento paso a paso).
Depende solo de perceptron.py (para los tipos de datos) y de reportlab.
"""
from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from perceptron import Perceptron

# --------------------------------------------------------------------- fuentes
_FONT, _FONT_B = "Helvetica", "Helvetica-Bold"
_UNICODE = False


def _register_fonts() -> None:
    """Intenta registrar DejaVu (soporta η, Σ, subíndices). Si no, usa Helvetica."""
    global _FONT, _FONT_B, _UNICODE
    candidates = []
    try:
        import matplotlib
        d = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        candidates.append((os.path.join(d, "DejaVuSans.ttf"), os.path.join(d, "DejaVuSans-Bold.ttf")))
    except Exception:
        pass
    candidates += [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
        ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
    ]
    for reg, bold in candidates:
        if os.path.exists(reg) and os.path.exists(bold):
            try:
                pdfmetrics.registerFont(TTFont("RptFont", reg))
                pdfmetrics.registerFont(TTFont("RptFont-B", bold))
                pdfmetrics.registerFontFamily("RptFont", normal="RptFont", bold="RptFont-B",
                                              italic="RptFont", boldItalic="RptFont-B")
                _FONT, _FONT_B, _UNICODE = "RptFont", "RptFont-B", True
                return
            except Exception:
                continue


_register_fonts()


def _eta() -> str:
    return "η" if _UNICODE else "eta"


def _sum() -> str:
    return "Σ" if _UNICODE else "suma"


def _num(v: float) -> str:
    """Número completo, sin truncar, sin ceros de relleno."""
    s = f"{v:.10f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# ---------------------------------------------------------------------- reporte
def generate_pdf(model: Perceptron, title: str = "Perceptrón Monocapa – Compuerta AND") -> bytes:
    """Construye el PDF en memoria y devuelve sus bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm, title=title)
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName=_FONT_B, fontSize=18, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName=_FONT_B, fontSize=13,
                        spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#1f3a5f"))
    h3 = ParagraphStyle("h3", parent=ss["Heading3"], fontName=_FONT_B, fontSize=10.5,
                        spaceBefore=8, spaceAfter=3)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName=_FONT, fontSize=9.5, leading=14)
    mono = ParagraphStyle("mono", parent=body, leftIndent=12, fontSize=9, leading=13)

    story = [Paragraph(title, h1),
             Paragraph(f"Generado el {datetime.now():%d/%m/%Y %H:%M}", body), Spacer(1, 6)]

    # 1. Parámetros
    story.append(Paragraph("1. Parámetros de entrenamiento", h2))
    params = [
        ["Parámetro", "Valor"],
        ["Peso inicial w1", _num(model.w1_init)],
        ["Peso inicial w2", _num(model.w2_init)],
        ["Bias inicial b", _num(model.b_init)],
        [f"Tasa de aprendizaje {_eta()}", _num(model.eta)],
        ["Máximo de épocas", str(model.max_epochs)],
        ["Épocas ejecutadas", str(len(model.epochs))],
        ["Convergió", "Sí" if model.converged else "No"],
        ["Pesos finales", f"w1 = {_num(model.w1)},  w2 = {_num(model.w2)},  b = {_num(model.b)}"],
    ]
    story.append(_table(params, [6 * cm, 11 * cm], align_left=True))

    # 2. Tabla de épocas
    story.append(Paragraph("2. Resumen por épocas", h2))
    rows = [["Época", "w1", "w2", "b", "Errores"]]
    for e in model.epochs:
        rows.append([str(e.epoch), _num(e.w1), _num(e.w2), _num(e.b), str(e.errors)])
    story.append(_table(rows, [2.5 * cm, 3.8 * cm, 3.8 * cm, 3.8 * cm, 3 * cm]))

    # 3. Procedimiento
    story.append(Paragraph("3. Procedimiento paso a paso", h2))
    story.append(Paragraph(
        f"Para cada muestra (x1, x2) con salida deseada t se calcula: "
        f"z = x1·w1 + x2·w2 + b;  y = f(z) (escalón: 1 si z ≥ 0, 0 si z &lt; 0);  e = t − y;  "
        f"w<sub>nuevo</sub> = w<sub>previo</sub> + {_eta()}·e·x  (para el bias, x = 1).", body))

    for ep in model.epochs:
        block = [Paragraph(f"Época {ep.epoch}  (errores: {ep.errors})", h3)]
        for s in model.steps_of_epoch(ep.epoch):
            lines = [
                f"<b>Muestra {s.sample}:</b> x1 = {_num(s.x1)}, x2 = {_num(s.x2)}, t = {s.t}",
                f"z = ({_num(s.x1)}·{_num(s.w1_prev)}) + ({_num(s.x2)}·{_num(s.w2_prev)}) + {_num(s.b_prev)} = <b>{_num(s.z)}</b>",
                f"y = f({_num(s.z)}) = <b>{s.y}</b>,   e = {s.t} − {s.y} = <b>{s.e}</b>",
                f"w1 = {_num(s.w1_prev)} + {_num(model.eta)}·{s.e}·{_num(s.x1)} = <b>{_num(s.w1_new)}</b>",
                f"w2 = {_num(s.w2_prev)} + {_num(model.eta)}·{s.e}·{_num(s.x2)} = <b>{_num(s.w2_new)}</b>",
                f"b  = {_num(s.b_prev)} + {_num(model.eta)}·{s.e}·1 = <b>{_num(s.b_new)}</b>",
            ]
            para = [Paragraph(l, mono if i else body) for i, l in enumerate(lines)]
            block.append(KeepTogether(para + [Spacer(1, 4)]))
        story.extend(block)

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def _table(rows, widths, align_left: bool = False) -> Table:
    t = Table(rows, colWidths=widths, repeatRows=1)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), _FONT),
        ("FONTNAME", (0, 0), (-1, 0), _FONT_B),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2cf")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT" if align_left else "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(style))
    return t


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(_FONT, 8)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Página {doc.page}")
    canvas.drawString(1.8 * cm, 1.0 * cm, "Perceptrón Monocapa – Compuerta AND")
    canvas.restoreState()