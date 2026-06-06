"""VQ-Font 파인튜닝 분석 보고서 PDF 생성"""
import json, os, glob
from fpdf import FPDF

# ── 데이터 수집 ──
with open('build_dataset/cr_mapping_v2.json') as f:
    cr = json.load(f)

imgs = glob.glob('datasets/train_font_image/reference_images_v2/*.png')
chars = [os.path.basename(p).split('.')[0] for p in imgs]
hex_set = set(hex(ord(c))[2:].upper() for c in chars)
n_chars = len(chars)
n_cr_keys = len(cr)
n_inferable = sum(1 for uni, deps in cr.items() if set(deps).issubset(hex_set))

# 부족한 구성요소 분석
all_needed = set()
for deps in cr.values():
    all_needed.update(deps)
missing_components = all_needed - hex_set
missing_chars_display = [chr(int(h, 16)) for h in sorted(missing_components)]

# ── PDF 생성 ──
FONT_PATH = 'datasets/train_font_ttf/NanumBarunpenR.ttf'

class PDF(FPDF):
    def header(self):
        self.set_font('NanumBarunpen', 'B', 14)
        self.cell(0, 10, 'VQ-Font 붓글씨 파인튜닝 분석 보고서', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font('NanumBarunpen', '', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def section_title(self, title):
        self.set_font('NanumBarunpen', 'B', 12)
        self.set_fill_color(230, 230, 250)
        self.cell(0, 8, title, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font('NanumBarunpen', '', 10)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def bullet(self, text, indent=10):
        self.set_font('NanumBarunpen', '', 10)
        self.set_x(self.l_margin + indent)
        self.multi_cell(0, 6, f'• {text}')

    def table_row(self, col1, col2, widths=(55, 125), bold_first=False):
        self.set_font('NanumBarunpen', 'B' if bold_first else '', 10)
        x = self.get_x()
        y = self.get_y()
        self.multi_cell(widths[0], 6, col1, border=1)
        h1 = self.get_y() - y
        self.set_xy(x + widths[0], y)
        self.set_font('NanumBarunpen', '', 10)
        self.multi_cell(widths[1], 6, col2, border=1)
        h2 = self.get_y() - y
        # 행 높이 맞춤
        if h1 > h2:
            self.set_y(y + h1)


pdf = PDF()
pdf.alias_nb_pages()
pdf.add_font('NanumBarunpen', '', FONT_PATH, uni=True)
pdf.add_font('NanumBarunpen', 'B', FONT_PATH, uni=True)
pdf.add_page()

# ═══ 1. 현황 요약 ═══
pdf.section_title('1. 현황 요약')
pdf.body_text(
    f'• reference_images_v2 이미지 수: {n_chars}장\n'
    f'• cr_mapping_v2.json 전체 유니코드 키: {n_cr_keys}개\n'
    f'• 실제 추론 가능 글자 수: {n_inferable} / 11,172  ({n_inferable/11172*100:.1f}%)\n'
    f'• 부족한 구성요소 수: {len(missing_components)}개'
)

# ═══ 2. 추론 결과가 안 좋을 수 있는 원인 ═══
pdf.section_title('2. 추론 결과 부진 원인 분석')

pdf.set_font('NanumBarunpen', 'B', 11)
pdf.cell(0, 7, '2-1. 데이터 문제 (가장 가능성 높음)', new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)

pdf.table_row('원인', '설명', bold_first=True)
pdf.table_row(
    '레퍼런스 이미지 수 부족',
    f'reference_images_v2에 {n_chars}자만 존재. cr_mapping의 모든 구성요소를 커버하지 못하여 '
    f'추론 가능 글자가 {n_inferable}자로 제한됨. 11,172자 전체 생성 불가.'
)
pdf.table_row(
    '이미지 품질',
    '스캔/촬영 노이즈, 해상도 불일치, 배경 처리 미비 시 학습 품질 저하.'
)
pdf.table_row(
    'content reference 매핑 부족',
    'cr_mapping_v2.json에 모든 11,172자에 대한 분해 매핑이 정의되어야 함. '
    '빠진 글자는 추론 자체가 불가능.'
)
pdf.ln(3)

pdf.set_font('NanumBarunpen', 'B', 11)
pdf.cell(0, 7, '2-2. 학습 설정 문제', new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)

pdf.table_row('원인', '설명', bold_first=True)
pdf.table_row(
    '50,000 iter 부족 가능성',
    '붓글씨는 일반 폰트보다 획 변형이 커서 더 많은 학습 반복이 필요할 수 있음. '
    '100,000~200,000 iter 권장.'
)
pdf.table_row(
    '인코더 동결',
    'component_encoder와 content_encoder를 얼리면 붓글씨의 독특한 부품 형태를 '
    '새로 학습하지 못함. decoder만으로는 스타일 변환에 한계.'
)
pdf.table_row(
    '75% 붓글씨 비율',
    '나머지 25% 일반 폰트 학습이 기존 지식 유지 역할이지만, '
    '붓글씨 학습을 방해할 수도 있음. 비율 조정 검토 필요.'
)
pdf.ln(3)

pdf.set_font('NanumBarunpen', 'B', 11)
pdf.cell(0, 7, '2-3. 추론 파이프라인 문제', new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)

pdf.table_row('원인', '설명', bold_first=True)
pdf.table_row(
    '체크포인트 선택',
    'inference.py에서 generator_ema → generator 순으로 로드. '
    'EMA가 더 안정적이므로 generator_ema가 없는 체크포인트는 품질 하락.'
)
pdf.table_row(
    'kshot=3 참조 이미지',
    '추론 시 참조 이미지 3장 사용. reference_images_v2에 해당 글자의 '
    '구성요소 이미지가 3장 모두 있어야 함.'
)
pdf.table_row(
    "reduction='mean'",
    '여러 참조를 평균내는 방식. 붓글씨처럼 변동이 큰 스타일에서는 특성이 뭉개질 수 있음.'
)

# ═══ 3. 부족한 구성요소 목록 ═══
pdf.add_page()
pdf.section_title('3. 부족한 구성요소 글자 목록')
pdf.body_text(
    f'reference_images_v2에 없지만 cr_mapping에서 필요로 하는 구성요소 '
    f'글자 {len(missing_components)}개:'
)
# 한 줄에 20자씩 표시
line = ''
for i, ch in enumerate(missing_chars_display):
    line += ch + ' '
    if (i + 1) % 20 == 0:
        pdf.body_text(line.strip())
        line = ''
if line.strip():
    pdf.body_text(line.strip())

# ═══ 4. 권장 대응 순서 ═══
pdf.add_page()
pdf.section_title('4. 권장 대응 순서')

steps = [
    ('1단계: 레퍼런스 이미지 보강',
     '붓글씨 이미지를 최소 200~300자로 늘려 cr_mapping의 구성요소 커버율을 100%로 만들기. '
     f'현재 부족한 {len(missing_components)}개 구성요소에 해당하는 글자 이미지를 추가해야 함.'),
    ('2단계: 학습 반복 수 증가',
     'iter를 100,000 이상으로 설정. 붓글씨 스타일의 복잡한 획 변형을 충분히 학습하도록 함.'),
    ('3단계: 인코더 동결 해제 시도',
     'component_encoder에 매우 작은 LR(1e-6)로 미세 조정을 허용하여 '
     '붓글씨 부품 인식 능력을 개선.'),
    ('4단계: 중간 체크포인트 비교',
     'save_freq=5000마다 저장된 체크포인트 중 가장 좋은 결과를 생성하는 것을 선택. '
     '과적합(overfitting) 방지.'),
]

for title, desc in steps:
    pdf.set_font('NanumBarunpen', 'B', 11)
    pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
    pdf.body_text(desc)
    pdf.ln(1)

# ═══ 5. 파인튜닝 코드 검토 결과 ═══
pdf.section_title('5. 파인튜닝 코드 검토 결과 (정상 확인)')

checks = [
    ('trainer_utils.py', 'discriminator 부분 복사 로직 적용 완료. shape mismatch 시 기존 78개 임베딩 보존, 79번째(reference_images_v2) 랜덤 초기화.'),
    ('trainer_utils.py', 'generator strict=False 로드 정상. optimizer/scheduler 미로드 (파인튜닝에 적합).'),
    ('train.py', '인코더 동결 (component_encoder, content_encoder) 정상. optimizer에 requires_grad=True 파라미터만 포함.'),
    ('combined_trainer.py', '학습 루프, D/G 분리 학습, gradient clipping 정상.'),
    ('base_trainer.py', 'EMA (decay=0.999), save (generator_ema 포함), discriminator state_dict 저장 정상.'),
    ('custom_finetune.yaml', 'LR, batch_size, iter, save_freq 등 설정 정상. save_freq(5000)가 val_freq(1000)의 배수 확인.'),
]

pdf.table_row('파일', '검토 결과', widths=(55, 125), bold_first=True)
for fname, result in checks:
    pdf.table_row(fname, result)

# ═══ 저장 ═══
output_path = '/home/dev/Project/VQ-Font/vqfont_finetune_report.pdf'
pdf.output(output_path)
print(f'PDF 저장 완료: {output_path}')
