"""Build the free-form solution presentation for the LG Aimers Phase 2 entry."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "LG_Aimers_솔루션_PPT_Phase2.pptx"

W = Inches(13.333333)
H = Inches(7.5)

BG = "0B1020"
PANEL = "151C31"
PANEL_2 = "1C2640"
INK = "F5F7FB"
MUTED = "AEB8CC"
TEAL = "39D8C6"
TEAL_DARK = "166E6B"
CORAL = "FF6B6B"
GOLD = "F4C95D"
BLUE = "6EA8FE"
GRID = "2A3655"

FONT = "Noto Sans CJK KR"
FONT_MONO = "Noto Sans Mono CJK KR"


def rgb(hex_code: str) -> RGBColor:
    return RGBColor.from_string(hex_code)


def add_rect(slide, x, y, w, h, fill, radius=False, line=None, width=1):
    kind = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    if line:
        shape.line.color.rgb = rgb(line)
        shape.line.width = Pt(width)
    else:
        shape.line.fill.background()
    if radius:
        try:
            shape.adjustments[0] = 0.12
        except (IndexError, ValueError):
            pass
    return shape


def add_line(slide, x1, y1, x2, y2, color=GRID, width=1.2, dash=None):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    line.line.color.rgb = rgb(color)
    line.line.width = Pt(width)
    if dash is not None:
        line.line.dash_style = dash
    return line


def add_text(
    slide,
    text,
    x,
    y,
    w,
    h,
    size=20,
    color=INK,
    bold=False,
    font=FONT,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin=0.02,
    line_spacing=1.0,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    p.line_spacing = line_spacing
    for run in p.runs:
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = rgb(color)
    return box


def add_rich_text(slide, runs, x, y, w, h, size=20, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    for item in runs:
        r = p.add_run()
        r.text = item[0]
        r.font.name = item[4] if len(item) > 4 else FONT
        r.font.size = Pt(item[1] if len(item) > 1 else size)
        r.font.bold = item[2] if len(item) > 2 else False
        r.font.color.rgb = rgb(item[3] if len(item) > 3 else INK)
    return box


def add_label(slide, text, x, y, w, color=TEAL, fill=PANEL_2):
    add_rect(slide, x, y, w, 0.32, fill, radius=True)
    add_text(slide, text, x, y + 0.01, w, 0.27, size=10, color=color, bold=True, align=PP_ALIGN.CENTER)


def add_bullet_list(slide, items, x, y, w, h, size=16, color=INK, bullet_color=TEAL, gap=0.43):
    for idx, item in enumerate(items):
        yy = y + idx * gap
        add_text(slide, "•", x, yy - 0.01, 0.25, 0.3, size=size + 2, color=bullet_color, bold=True)
        add_text(slide, item, x + 0.27, yy, w - 0.27, min(gap, h), size=size, color=color)


def add_header(slide, number, eyebrow, title, subtitle=None):
    add_text(slide, f"{number:02d}", 0.55, 0.40, 0.45, 0.28, size=11, color=TEAL, bold=True)
    add_text(slide, eyebrow.upper(), 1.05, 0.40, 3.4, 0.28, size=10, color=MUTED, bold=True)
    add_text(slide, title, 0.55, 0.80, 12.2, 0.60, size=28, color=INK, bold=True)
    if subtitle:
        add_text(slide, subtitle, 0.58, 1.43, 12.0, 0.40, size=12, color=MUTED)
    add_line(slide, 0.55, 1.92, 12.78, 1.92, GRID, 0.8)


def add_footer(slide, number):
    add_text(slide, "LG Aimers · 제구 성공 확률 예측", 0.55, 7.15, 4.4, 0.20, size=8, color="74809A")
    add_text(slide, f"{number:02d}", 12.30, 7.15, 0.45, 0.20, size=8, color="74809A", align=PP_ALIGN.RIGHT)


def set_bg(slide):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(BG)


def add_card_title(slide, title, x, y, w, color=TEAL):
    add_text(slide, title, x, y, w, 0.34, size=13, color=color, bold=True)


def build_deck():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    blank = prs.slide_layouts[6]

    # 01 — Cover
    slide = prs.slides.add_slide(blank)
    set_bg(slide)
    add_rect(slide, 8.50, -0.15, 5.10, 7.80, "10182C")
    add_rect(slide, 8.92, 0.46, 3.65, 3.65, BG, line=GRID, width=1.4)
    diamond = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.DIAMOND, Inches(9.55), Inches(1.06), Inches(2.4), Inches(2.4))
    diamond.fill.background()
    diamond.line.color.rgb = rgb(TEAL_DARK)
    diamond.line.width = Pt(2.3)
    for bx, by in [(10.67, 0.92), (11.83, 2.06), (10.67, 3.20), (9.52, 2.06)]:
        b = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(bx), Inches(by), Inches(0.26), Inches(0.26))
        b.fill.solid(); b.fill.fore_color.rgb = rgb(TEAL); b.line.fill.background()
    ball = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(10.30), Inches(4.64), Inches(1.05), Inches(1.05))
    ball.fill.solid(); ball.fill.fore_color.rgb = rgb(INK); ball.line.color.rgb = rgb("C8D1E2")
    add_line(slide, 10.55, 4.78, 11.06, 5.55, CORAL, 1.4)
    add_line(slide, 11.12, 4.80, 10.63, 5.54, CORAL, 1.4)
    add_label(slide, "LG AIMERS · PHASE 2", 0.63, 0.57, 2.05)
    add_text(slide, "투구 직전 정보로\n제구 성공 확률을 읽다", 0.63, 1.35, 7.45, 1.75, size=36, bold=True)
    add_text(slide, "시간 순 검증과 행 독립 피처 엔지니어링을 기반으로 한\n73개 피처 CatBoost 확률 모델", 0.68, 3.38, 7.35, 0.78, size=17, color=MUTED)
    add_rect(slide, 0.65, 4.55, 6.95, 0.04, TEAL)
    add_text(slide, "제출자  김재호   |   팀명  나란차", 0.68, 5.03, 6.6, 0.36, size=13, color=INK, bold=True)
    add_text(slide, "최종 학습: 2019–2024  ·  예측 대상: 2025  ·  81 features", 0.68, 5.60, 7.1, 0.34, size=11, color=MUTED)
    add_rect(slide, 0.65, 6.30, 7.10, 0.62, PANEL, radius=True)
    add_text(slide, "Phase 3 오프라인 해커톤 참가 여부  |  아니요", 0.88, 6.48, 6.62, 0.27, size=13, color=CORAL, bold=True)
    add_text(slide, "SOLUTION PRESENTATION", 9.05, 6.68, 3.33, 0.23, size=9, color="74809A", bold=True, align=PP_ALIGN.CENTER)

    # 02 — Problem
    slide = prs.slides.add_slide(blank); set_bg(slide)
    add_header(slide, 2, "Problem framing", "확률을 맞히는 문제 — 순위보다 ‘잘 보정된 확률’")
    cards = [
        ("TARGET", "control_success", "각 투구의 제구 성공 확률\n0–1 실수 출력", TEAL),
        ("METRIC", "Brier Skill Score", "확률 오차를 직접 평가\n높을수록 우수", GOLD),
        ("UNIT", "1 pitch = 1 row", "평가 행별 독립 예측\n다른 test 행 참조 금지", BLUE),
    ]
    for i, (tag, value, note, color) in enumerate(cards):
        x = 0.58 + i * 4.15
        add_rect(slide, x, 2.25, 3.78, 2.10, PANEL, radius=True, line=GRID)
        add_text(slide, tag, x + 0.24, 2.50, 3.15, 0.25, size=10, color=color, bold=True)
        add_text(slide, value, x + 0.24, 2.88, 3.25, 0.42, size=20, bold=True)
        add_text(slide, note, x + 0.24, 3.45, 3.18, 0.60, size=12, color=MUTED)
    add_rect(slide, 0.58, 4.72, 12.15, 1.72, "10182C", radius=True, line=GRID)
    add_text(slide, "설계 원칙", 0.88, 4.98, 1.55, 0.30, size=13, color=CORAL, bold=True)
    principles = [
        "미래 시즌 예측 구조를 반영한 시간 순 검증",
        "공식 데이터만 사용 · 외부 API/외부 데이터 없음",
        "현재 행 + 고정 모델만으로 완결되는 추론",
    ]
    for i, text_value in enumerate(principles):
        x = 2.35 + i * 3.32
        add_text(slide, f"0{i+1}", x, 4.98, 0.35, 0.30, size=11, color=TEAL, bold=True)
        add_text(slide, text_value, x + 0.45, 4.94, 2.62, 0.72, size=12, color=INK, bold=True)
    add_text(slide, "핵심 질문  |  ‘현재 투구 직전까지 알고 있는 정보만으로, 미래 시즌에서도 확률 품질을 유지할 수 있는가?’", 0.88, 6.00, 11.25, 0.25, size=11, color=MUTED)
    add_footer(slide, 2)

    # 03 — Data & validation
    slide = prs.slides.add_slide(blank); set_bg(slide)
    add_header(slide, 3, "Data & validation", "랜덤 분할 대신 ‘과거 → 미래’로 검증")
    add_text(slide, "개발 검증", 0.63, 2.22, 1.3, 0.30, size=12, color=TEAL, bold=True)
    y = 2.83
    x0 = 0.90
    seasons = list(range(2019, 2025))
    for i, season in enumerate(seasons):
        x = x0 + i * 1.73
        fill = TEAL_DARK if season <= 2023 else CORAL
        add_rect(slide, x, y, 1.28, 0.72, fill, radius=True)
        add_text(slide, str(season), x, y + 0.17, 1.28, 0.30, size=14, bold=True, align=PP_ALIGN.CENTER)
        if i < len(seasons)-1:
            add_line(slide, x + 1.28, y + 0.36, x + 1.70, y + 0.36, GRID, 1.4)
    add_text(slide, "TRAIN  |  1,475,092 rows", 1.15, 3.76, 8.20, 0.28, size=12, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "VALID\n253,507", 10.55, 3.70, 1.28, 0.55, size=11, color=CORAL, bold=True, align=PP_ALIGN.CENTER)
    add_line(slide, 0.64, 4.35, 12.68, 4.35, GRID, 0.8)
    add_text(slide, "최종 학습 / 제출", 0.63, 4.70, 1.55, 0.30, size=12, color=GOLD, bold=True)
    for i, season in enumerate(seasons):
        x = x0 + i * 1.45
        add_rect(slide, x, 5.20, 1.05, 0.62, TEAL_DARK, radius=True)
        add_text(slide, str(season), x, 5.36, 1.05, 0.24, size=12, bold=True, align=PP_ALIGN.CENTER)
        if i < len(seasons)-1:
            add_line(slide, x + 1.05, 5.51, x + 1.42, 5.51, GRID, 1.3)
    add_text(slide, "→", 9.68, 5.26, 0.52, 0.36, size=22, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
    add_rect(slide, 10.25, 5.08, 1.95, 0.86, PANEL_2, radius=True, line=GOLD, width=1.5)
    add_text(slide, "2025\nPREDICT", 10.25, 5.22, 1.95, 0.52, size=13, color=GOLD, bold=True, align=PP_ALIGN.CENTER)
    add_rect(slide, 0.66, 6.28, 11.97, 0.48, PANEL, radius=True)
    add_text(slide, "검증 원칙  |  시즌 변화(drift)를 숨기는 random K-fold를 피하고, 운영 시점과 같은 방향으로 성능을 측정", 0.87, 6.40, 11.52, 0.24, size=11, color=MUTED)
    add_footer(slide, 3)

    # 04 — Features
    slide = prs.slides.add_slide(blank); set_bg(slide)
    add_header(slide, 4, "Feature engineering", "47개 공식 입력 + 24개 파생 + 2개 집계 + 8개 Trackman 편차 = 81개")
    add_rect(slide, 0.60, 2.22, 3.15, 4.35, PANEL, radius=True, line=GRID)
    add_text(slide, "47", 0.86, 2.52, 1.10, 0.70, size=38, color=TEAL, bold=True)
    add_text(slide, "공식 입력 피처", 1.83, 2.72, 1.60, 0.32, size=14, bold=True)
    add_bullet_list(slide, ["경기·카운트·점수", "주자·상황 중요도", "선수·팀·손잡이", "공식 asof_* 과거 이력"], 0.85, 3.54, 2.52, 2.20, size=13, gap=0.53)
    add_text(slide, "+", 3.86, 3.80, 0.50, 0.60, size=32, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
    add_rect(slide, 4.40, 2.22, 3.52, 4.35, PANEL, radius=True, line=GRID)
    add_text(slide, "24", 4.68, 2.52, 1.10, 0.70, size=38, color=GOLD, bold=True)
    add_text(slide, "상태·상호작용 파생", 5.72, 2.72, 1.85, 0.32, size=14, bold=True)
    feature_groups = [
        ("상황", "count_state · scoring_pos_runner\nlate_inning · leverage pressure"),
        ("매치업", "좌/우 platoon · batter pressure"),
        ("추세", "직전 1·3경기 성공률 delta"),
        ("Cold-start", "결측 개수 · 결측 여부 · 이력 0"),
    ]
    for i, (tag, desc) in enumerate(feature_groups):
        yy = 3.48 + i * 0.69
        add_label(slide, tag, 4.69, yy, 0.78, color=GOLD, fill="222C47")
        add_text(slide, desc, 5.62, yy - 0.01, 1.96, 0.55, size=10, color=INK, bold=True)
    add_text(slide, "=", 8.06, 3.80, 0.50, 0.60, size=32, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
    add_rect(slide, 8.62, 2.22, 4.10, 4.35, "11222D", radius=True, line=TEAL_DARK, width=1.5)
    add_text(slide, "81", 8.98, 2.42, 1.78, 0.90, size=52, color=TEAL, bold=True)
    add_text(slide, "FINAL FEATURES", 10.55, 2.80, 1.60, 0.30, size=11, color=MUTED, bold=True)
    add_line(slide, 8.98, 3.42, 12.33, 3.42, TEAL_DARK, 1.1)
    add_bullet_list(slide, ["row_id · target 제외", "결측 자체를 정보로 유지", "test 통계·빈도·순위 미사용", "학습·추론 동일 함수/순서"], 9.00, 3.78, 3.10, 2.25, size=12, gap=0.52)
    add_text(slide, "모든 파생값은 해당 행의 입력 컬럼만으로 계산", 9.00, 6.03, 3.08, 0.28, size=10, color=TEAL, bold=True)
    add_footer(slide, 4)

    # 05 — Architecture
    slide = prs.slides.add_slide(blank); set_bg(slide)
    add_header(slide, 5, "Model architecture", "ASOF 상태·상호작용·조건부 집계를 단일 CatBoost에 결합")
    add_rect(slide, 0.62, 2.65, 2.15, 1.16, PANEL, radius=True, line=GRID)
    add_text(slide, "81 FEATURES", 0.62, 2.88, 2.15, 0.28, size=12, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "현재 행 단위", 0.62, 3.22, 2.15, 0.24, size=10, color=MUTED, align=PP_ALIGN.CENTER)
    add_line(slide, 2.78, 3.23, 3.48, 3.23, TEAL_DARK, 2.0)
    add_line(slide, 3.48, 3.23, 3.48, 2.72, TEAL_DARK, 1.5)
    add_line(slide, 3.48, 3.23, 3.48, 4.44, TEAL_DARK, 1.5)
    add_rect(slide, 3.70, 2.15, 2.44, 1.17, PANEL_2, radius=True, line=BLUE)
    add_text(slide, "ASOF + 상태", 3.70, 2.44, 2.44, 0.32, size=17, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "73 features", 3.70, 2.86, 2.44, 0.22, size=10, color=MUTED, align=PP_ALIGN.CENTER)
    add_rect(slide, 3.70, 4.02, 2.44, 1.17, PANEL_2, radius=True, line=CORAL)
    add_text(slide, "조건부 집계", 3.70, 4.30, 2.44, 0.32, size=17, color=CORAL, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "+ 8 Trackman", 3.70, 4.72, 2.44, 0.22, size=10, color=MUTED, align=PP_ALIGN.CENTER)
    add_line(slide, 6.15, 2.74, 7.02, 2.74, BLUE, 1.6)
    add_line(slide, 6.15, 4.60, 7.02, 4.60, CORAL, 1.6)
    add_rect(slide, 7.02, 2.28, 2.05, 2.78, "10182C", radius=True, line=GOLD, width=1.4)
    add_text(slide, "feature merge", 7.02, 2.57, 2.05, 0.28, size=11, color=GOLD, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "A", 7.35, 3.14, 0.40, 0.40, size=22, color=CORAL, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "→ 상태", 7.70, 3.19, 1.05, 0.28, size=11, color=INK, bold=True)
    add_line(slide, 7.30, 3.75, 8.80, 3.75, GRID, 0.8)
    add_text(slide, "B", 7.35, 4.05, 0.40, 0.40, size=22, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "→ 집계", 7.70, 4.10, 1.05, 0.28, size=11, color=INK, bold=True)
    add_line(slide, 9.10, 3.67, 9.75, 3.67, GOLD, 2.0)
    add_rect(slide, 9.78, 2.95, 2.94, 1.45, "11222D", radius=True, line=TEAL, width=1.6)
    add_text(slide, "CatBoost · 300 trees", 9.78, 3.27, 2.94, 0.32, size=15, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "0 ≤ p ≤ 1", 9.78, 3.76, 2.94, 0.24, size=11, color=MUTED, align=PP_ALIGN.CENTER)
    add_rect(slide, 0.65, 5.72, 12.02, 0.72, PANEL, radius=True)
    add_text(slide, "규칙은 검증 전에 고정  ·  연속 가중치 탐색 없음  ·  test 예측 분포를 이용한 선택/보정 없음", 0.90, 5.94, 11.55, 0.26, size=12, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, 5)

    # 06 — Results
    slide = prs.slides.add_slide(blank); set_bg(slide)
    add_header(
        slide,
        6,
        "Temporal validation",
        "시간 전이 검증으로 보정식의 재현성을 확인",
        "SUB-002 공식 886.2488171351점  ·  신규 후보 867.3538231619점으로 롤백",
    )
    # chart panel
    add_rect(slide, 0.62, 2.22, 7.18, 4.47, PANEL, radius=True, line=GRID)
    add_text(slide, "Brier Skill Score", 0.88, 2.46, 2.4, 0.28, size=13, bold=True)
    add_text(slide, "높을수록 우수", 2.73, 2.48, 1.30, 0.23, size=9, color=MUTED)
    chart_x, chart_y, chart_w, chart_h = 1.18, 3.06, 6.15, 2.76
    min_v, max_v = -0.0045, 0.0080
    zero_y = chart_y + chart_h * (max_v / (max_v - min_v))
    for val in [-0.004, 0.0, 0.004, 0.008]:
        yy = chart_y + chart_h * ((max_v - val) / (max_v - min_v))
        add_line(slide, chart_x, yy, chart_x + chart_w, yy, GRID if val != 0 else MUTED, 0.8 if val != 0 else 1.2)
        add_text(slide, f"{val:.3f}", 0.73, yy - 0.11, 0.40, 0.22, size=8, color=MUTED, align=PP_ALIGN.RIGHT)
    series = {
        "2023": [("SUB1", 0.001142785, CORAL), ("SUB2", 0.004089905, TEAL)],
        "2024": [("SUB1", 0.007384811, CORAL), ("SUB2", 0.007593591, TEAL)],
    }
    gx = [1.55, 4.55]
    for group_idx, (season, vals) in enumerate(series.items()):
        for i, (name, val, color) in enumerate(vals):
            x = gx[group_idx] + i * 0.67
            val_y = chart_y + chart_h * ((max_v - val) / (max_v - min_v))
            if val >= 0:
                top, height = val_y, zero_y - val_y
            else:
                top, height = zero_y, val_y - zero_y
            add_rect(slide, x, top, 0.46, max(height, 0.02), color, radius=True)
            add_text(slide, name, x - 0.10, chart_y + chart_h + 0.08, 0.65, 0.22, size=8, color=MUTED, bold=(name == "SUB2"), align=PP_ALIGN.CENTER)
        add_text(slide, season, gx[group_idx] + 0.41, 6.18, 1.10, 0.26, size=11, color=INK, bold=True, align=PP_ALIGN.CENTER)
    # right-side metrics
    add_rect(slide, 8.08, 2.22, 4.64, 4.47, "10182C", radius=True, line=GRID)
    add_text(slide, "정확한 검증 수치", 8.40, 2.47, 2.10, 0.30, size=13, bold=True)
    headers = ["SEASON", "MODEL", "BRIER ↓", "BSS ↑"]
    xs = [8.38, 9.28, 10.22, 11.37]
    ws = [0.80, 0.86, 1.05, 0.95]
    for t, x, w in zip(headers, xs, ws):
        add_text(slide, t, x, 2.93, w, 0.24, size=8, color=MUTED, bold=True, align=PP_ALIGN.RIGHT if "↓" in t or "↑" in t else PP_ALIGN.LEFT)
    rows = [
        ("2023", "SUB-001", "0.249714", "0.001143", CORAL),
        ("", "SUB-002", "0.248978", "0.004090", TEAL),
        ("2024", "SUB-001", "0.247962", "0.007385", CORAL),
        ("", "SUB-002", "0.247910", "0.007594", TEAL),
    ]
    for i, row in enumerate(rows):
        yy = 3.38 + i * 0.58
        if i in (1, 3):
            add_rect(slide, 8.27, yy - 0.06, 4.14, 0.44, "142C34", radius=True)
        for j, (value, x, w) in enumerate(zip(row[:4], xs, ws)):
            align = PP_ALIGN.RIGHT if j >= 2 else PP_ALIGN.LEFT
            add_text(slide, value, x, yy, w, 0.24, size=9, color=row[4] if j == 1 else INK, bold=(i in (1,3)), align=align)
    add_line(slide, 8.34, 5.91, 12.39, 5.91, GRID, 0.8)
    add_text(slide, "SUB-001 대비 Brier 개선", 8.38, 6.08, 2.18, 0.24, size=9, color=MUTED)
    add_text(slide, "2023  −0.000737", 10.50, 6.02, 1.82, 0.22, size=10, color=TEAL, bold=True, align=PP_ALIGN.RIGHT)
    add_text(slide, "2024  −0.000052", 10.50, 6.30, 1.82, 0.22, size=10, color=TEAL, bold=True, align=PP_ALIGN.RIGHT)
    add_footer(slide, 6)

    # 07 — Experiment decisions
    slide = prs.slides.add_slide(blank); set_bg(slide)
    add_header(slide, 7, "Experiment discipline", "좋았던 아이디어보다 ‘미래 시즌에 반복된 개선’을 채택")
    add_text(slide, "채택", 0.65, 2.24, 1.0, 0.30, size=13, color=TEAL, bold=True)
    adopted = [
        ("FE-001", "24개 상태·상호작용 파생", "행 독립성 검증 통과"),
        ("CatBoost", "범주형 변수의 안정적 처리", "2023·2024에서 강한 단일 모델"),
        ("COMBO-TM-FULL-006", "81개 피처 전체 학습", "ZIP·전이·감사 통과"),
    ]
    for i, (tag, title, note) in enumerate(adopted):
        y = 2.66 + i * 1.18
        add_rect(slide, 0.65, y, 5.70, 0.92, "11292C", radius=True, line=TEAL_DARK)
        add_label(slide, tag, 0.88, y + 0.17, 1.20, color=TEAL, fill="183C3D")
        add_text(slide, title, 2.27, y + 0.13, 2.70, 0.30, size=13, bold=True)
        add_text(slide, note, 2.27, y + 0.50, 3.60, 0.22, size=9, color=MUTED)
    add_text(slide, "미채택", 6.93, 2.24, 1.0, 0.30, size=13, color=CORAL, bold=True)
    rejected = [
        ("Season weighting", "감쇠 0.85/0.70", "2024 성능 저하"),
        ("Target aggregates", "투수·타자 조건부 통계", "2023 개선 미확인"),
        ("Recent blend", "전체 75% + 최근 25%", "추가 개선 0.0000028"),
        ("Support gate", "선택적 임계값", "전 시즌 일관성 부족"),
    ]
    for i, (tag, title, note) in enumerate(rejected):
        y = 2.66 + i * 0.88
        add_rect(slide, 6.93, y, 5.73, 0.67, PANEL, radius=True, line=GRID)
        add_text(slide, tag, 7.18, y + 0.14, 1.45, 0.25, size=10, color=CORAL, bold=True)
        add_text(slide, title, 8.70, y + 0.11, 2.08, 0.25, size=11, bold=True)
        add_text(slide, note, 10.80, y + 0.14, 1.48, 0.22, size=9, color=MUTED, align=PP_ALIGN.RIGHT)
    add_rect(slide, 6.93, 6.02, 5.73, 0.58, "241D28", radius=True)
    add_text(slide, "SUB-002 공식 886.2488171351  |  신규 867.3538231619", 7.12, 6.12, 5.35, 0.20, size=8, color=GOLD, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "신규 후보 −18.8949949382  ·  SUB-002 롤백", 7.12, 6.36, 5.35, 0.18, size=8, color=CORAL, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, 7)

    # 08 — Independence & compliance
    slide = prs.slides.add_slide(blank); set_bg(slide)
    add_header(slide, 8, "Inference integrity", "평가 데이터의 각 행을 완전히 독립적으로 예측")
    add_text(slide, "test.csv", 0.65, 2.27, 1.50, 0.32, size=13, color=TEAL, bold=True)
    for i in range(4):
        y = 2.79 + i * 0.64
        add_rect(slide, 0.68, y, 2.42, 0.44, PANEL_2, radius=True, line=GRID)
        add_text(slide, f"row {i+1:02d}  ·  current inputs", 0.86, y + 0.10, 1.98, 0.20, size=9, color=INK, font=FONT_MONO)
        add_line(slide, 3.11, y + 0.22, 3.67, y + 0.22, TEAL_DARK, 1.2)
        add_rect(slide, 3.67, y, 2.20, 0.44, "11292C", radius=True)
        add_text(slide, "build_features(row)", 3.78, y + 0.10, 1.95, 0.20, size=9, color=TEAL, bold=True, font=FONT_MONO)
        add_line(slide, 5.88, y + 0.22, 6.42, y + 0.22, TEAL_DARK, 1.2)
        add_rect(slide, 6.42, y, 2.15, 0.44, PANEL, radius=True)
        add_text(slide, "fixed models", 6.42, y + 0.10, 2.15, 0.20, size=9, color=INK, bold=True, align=PP_ALIGN.CENTER)
        add_line(slide, 8.58, y + 0.22, 9.12, y + 0.22, TEAL_DARK, 1.2)
        add_rect(slide, 9.12, y, 1.47, 0.44, "11222D", radius=True)
        add_text(slide, f"p{i+1}", 9.12, y + 0.10, 1.47, 0.20, size=10, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_rect(slide, 10.92, 2.60, 1.71, 3.05, "241A25", radius=True, line=CORAL, width=1.2)
    add_text(slide, "NO", 10.92, 2.93, 1.71, 0.42, size=23, color=CORAL, bold=True, align=PP_ALIGN.CENTER)
    add_bullet_list(slide, ["test groupby", "rolling / lag", "batch statistics", "inference fit", "distribution calibration"], 11.12, 3.63, 1.34, 1.70, size=9, bullet_color=CORAL, gap=0.31)
    add_rect(slide, 0.67, 5.78, 9.91, 0.72, PANEL, radius=True)
    checks = [("배치=단독", "max diff 0.0"), ("순서 변경", "동일 예측"), ("무관행 추가", "동일 예측")]
    for i, (title, note) in enumerate(checks):
        x = 0.92 + i * 3.20
        add_text(slide, "✓", x, 5.96, 0.32, 0.24, size=14, color=TEAL, bold=True)
        add_text(slide, title, x + 0.34, 5.92, 1.15, 0.26, size=10, color=INK, bold=True)
        add_text(slide, note, x + 1.50, 5.94, 1.20, 0.22, size=9, color=MUTED)
    add_footer(slide, 8)

    # 09 — Reproducibility
    slide = prs.slides.add_slide(blank); set_bg(slide)
    add_header(slide, 9, "Reproducibility", "실행 환경·모델 계약·성능 게이트를 제출물 안에 고정")
    metrics = [
        ("1,475,092 rows", "전체 학습 행", TEAL),
        ("81 features", "모델 피처 계약", GOLD),
        ("8 members", "제출 ZIP 구성", BLUE),
        ("AUDIT VERIFIED", "내부 감사 상태", CORAL),
    ]
    for i, (value, label, color) in enumerate(metrics):
        x = 0.62 + i * 3.04
        add_rect(slide, x, 2.22, 2.70, 1.18, PANEL, radius=True, line=GRID)
        add_text(slide, value, x + 0.20, 2.52, 2.28, 0.34, size=18, color=color, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, label, x + 0.20, 2.96, 2.28, 0.22, size=9, color=MUTED, align=PP_ALIGN.CENTER)
    add_rect(slide, 0.62, 3.78, 5.92, 2.62, "10182C", radius=True, line=GRID)
    add_card_title(slide, "제출 실행 환경", 0.92, 4.06, 2.20)
    env_rows = [
        ("Language", "Python 3.11.15"),
        ("CatBoost", "1.2.10"),
        ("Model", "CatBoost · 300 trees · 81 features"),
        ("Network", "미사용 · 외부 API 없음"),
    ]
    for i, (key, value) in enumerate(env_rows):
        yy = 4.53 + i * 0.42
        add_text(slide, key, 0.95, yy, 1.18, 0.22, size=9, color=MUTED, font=FONT_MONO)
        add_text(slide, value, 2.25, yy, 3.83, 0.22, size=10, color=INK, bold=True)
    add_rect(slide, 6.82, 3.78, 5.90, 2.62, "10182C", radius=True, line=GRID)
    add_card_title(slide, "Fail-fast 제출 계약", 7.12, 4.06, 2.40, color=GOLD)
    add_bullet_list(slide, [
        "입력·피처 개수/순서/ID 계약 검증",
        "모델 및 feature_columns SHA-256 검증",
        "NaN·Inf·범위 밖 확률 즉시 실패",
        "출력 2개 컬럼 및 입력 row 순서 보존",
    ], 7.11, 4.51, 5.02, 1.75, size=11, bullet_color=GOLD, gap=0.41)
    add_rect(slide, 0.64, 6.64, 12.03, 0.31, "12262B", radius=True)
    add_text(slide, "ZIP 감사: AUDIT_VERIFIED  ·  행 독립성/재현성 검증 통과  ·  공식 점수 확인 전 HOLD", 0.87, 6.69, 11.60, 0.19, size=9, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, 9)

    # 10 — Conclusion / participation
    slide = prs.slides.add_slide(blank); set_bg(slide)
    add_text(slide, "10", 0.58, 0.45, 0.45, 0.28, size=11, color=TEAL, bold=True)
    add_text(slide, "CONCLUSION", 1.07, 0.45, 2.0, 0.28, size=10, color=MUTED, bold=True)
    add_text(slide, "미래 시즌에도 흔들리지 않는 확률을 위해", 0.60, 1.03, 8.20, 0.62, size=29, bold=True)
    add_text(slide, "복잡도보다 검증 방향, 일관성, 추론 무결성을 우선했습니다.", 0.62, 1.72, 8.10, 0.40, size=15, color=MUTED)
    takeaways = [
        ("01", "TIME-AWARE", "2019–2023 → 2024\n미래 방향 검증"),
        ("02", "ROW-INDEPENDENT", "현재 행만으로 생성하는\n상태·상호작용 피처"),
        ("03", "COMBO-TM-FULL-006", "81개 피처 전체 학습\nTrackman train-only 번들"),
        ("04", "REPRODUCIBLE", "ZIP·행 독립성 검증\n공식 점수 대기"),
    ]
    for i, (num, title, note) in enumerate(takeaways):
        x = 0.62 + i * 3.03
        add_rect(slide, x, 2.55, 2.70, 1.58, PANEL, radius=True, line=GRID)
        add_text(slide, num, x + 0.20, 2.78, 0.42, 0.28, size=11, color=TEAL, bold=True)
        add_text(slide, title, x + 0.63, 2.77, 1.80, 0.28, size=10, color=INK, bold=True)
        add_text(slide, note, x + 0.20, 3.30, 2.28, 0.54, size=11, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
    add_rect(slide, 0.62, 4.63, 12.10, 1.70, "241A25", radius=True, line=CORAL, width=1.6)
    add_text(slide, "오프라인 해커톤(Phase 3) 참가 여부", 0.96, 4.99, 7.15, 0.36, size=17, color=INK, bold=True)
    add_text(slide, "아니요", 9.08, 4.87, 2.76, 0.66, size=34, color=CORAL, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "본 PPT는 온라인 해커톤(Phase 2) 솔루션 설명 및 코드 검증용 제출 자료입니다. 신규 ZIP은 승인 전 보류합니다.", 0.96, 5.62, 8.25, 0.28, size=11, color=MUTED)
    add_text(slide, "THANK YOU", 0.63, 6.72, 2.2, 0.27, size=10, color=TEAL, bold=True)
    add_text(slide, "제출자  김재호   |   팀명  나란차", 8.12, 6.69, 4.34, 0.28, size=10, color=MUTED, bold=True, align=PP_ALIGN.RIGHT)
    add_footer(slide, 10)

    # Ensure all slides have deterministic title metadata.
    for idx, s in enumerate(prs.slides, start=1):
        s.name = f"Solution {idx:02d}"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_deck()
