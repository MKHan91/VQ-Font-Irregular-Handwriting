#!/usr/bin/env bash
# =============================================================================
# scripts/build_pdf.sh
#
# 보고서 Markdown → PDF 빌드 헬퍼.
#
# 엔진 자동 탐지 우선순위:
#   1) pandoc + xelatex   (한글 폰트 권장, 가장 깔끔)
#   2) pandoc + lualatex  (xelatex 없을 때)
#   3) pandoc + wkhtmltopdf (latex 환경 없을 때)
#   4) weasyprint         (pandoc 자체가 없을 때)
#
# 옵션:
#   -i, --input  FILE     입력 .md (기본 README.md)
#   -o, --output FILE     출력 .pdf (기본 report/<input-stem>.pdf)
#   -e, --engine NAME     엔진 강제 지정 (xelatex|lualatex|wkhtmltopdf|weasyprint)
#   -f, --font   NAME     본문 폰트 이름 (기본: 자동 탐지 한글 폰트)
#   -t, --toc             목차 포함 (pandoc 모드 한정)
#       --no-toc          목차 제외
#       --install         필요한 패키지 자동 설치 (apt + pip)
#       --html-only       PDF 대신 HTML 만 생성 (디버그용)
#       --check           도구 가용성만 점검하고 종료
#   -h, --help            도움말
#
# 사용 예:
#   bash scripts/build_pdf.sh                      # README.md → report/README.pdf
#   bash scripts/build_pdf.sh -i report.md         # report.md → report/report.pdf
#   bash scripts/build_pdf.sh --install            # 의존 도구 설치 후 빌드
#   bash scripts/build_pdf.sh --check              # 가용 엔진 확인만
# =============================================================================

set -Eeuo pipefail

# ----- 색상 / 로깅 -----------------------------------------------------------
if [[ -t 1 ]]; then
  C_R=$'\033[31m'; C_G=$'\033[32m'; C_Y=$'\033[33m'
  C_B=$'\033[34m'; C_D=$'\033[2m';  C_N=$'\033[0m'
else
  C_R=""; C_G=""; C_Y=""; C_B=""; C_D=""; C_N=""
fi
log()  { printf "%s[INFO]%s %s\n" "$C_B" "$C_N" "$*"; }
ok()   { printf "%s[ OK ]%s %s\n" "$C_G" "$C_N" "$*"; }
warn() { printf "%s[WARN]%s %s\n" "$C_Y" "$C_N" "$*"; }
err()  { printf "%s[ERR ]%s %s\n" "$C_R" "$C_N" "$*" >&2; }

# ----- 경로 ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT="README.md"
OUTPUT=""
ENGINE=""
FONT=""
TOC="auto"
DO_INSTALL=0
HTML_ONLY=0
DO_CHECK=0

# ----- 옵션 파싱 -------------------------------------------------------------
usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)   INPUT="$2"; shift 2 ;;
    -o|--output)  OUTPUT="$2"; shift 2 ;;
    -e|--engine)  ENGINE="$2"; shift 2 ;;
    -f|--font)    FONT="$2"; shift 2 ;;
    -t|--toc)     TOC="yes"; shift ;;
    --no-toc)     TOC="no"; shift ;;
    --install)    DO_INSTALL=1; shift ;;
    --html-only)  HTML_ONLY=1; shift ;;
    --check)      DO_CHECK=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *)            err "알 수 없는 옵션: $1"; usage; exit 2 ;;
  esac
done

# ----- 한글 폰트 자동 탐지 ---------------------------------------------------
detect_korean_font() {
  # 1) fontconfig
  if command -v fc-list >/dev/null 2>&1; then
    local n
    for n in "Nanum Gothic" "NanumGothic" "NanumBarunGothic" \
             "Noto Sans CJK KR" "Noto Sans KR" "Malgun Gothic"; do
      if fc-list :lang=ko 2>/dev/null | grep -qi "$n"; then
        echo "$n"; return 0
      fi
    done
  fi
  # 2) 시스템 파일
  local candidates=(
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
  )
  local f
  for f in "${candidates[@]}"; do
    [[ -f "$f" ]] && { echo "$f"; return 0; }
  done
  # 3) 프로젝트 동봉 ttf
  local repo_font="$REPO_ROOT/datasets/train_font_ttf/NanumBarunpenR.ttf"
  [[ -f "$repo_font" ]] && { echo "$repo_font"; return 0; }
  return 1
}

# ----- 엔진 탐지 -------------------------------------------------------------
have() { command -v "$1" >/dev/null 2>&1; }

pick_engine() {
  if [[ -n "$ENGINE" ]]; then
    case "$ENGINE" in
      xelatex|lualatex)
        have pandoc && have "$ENGINE" && { echo "$ENGINE"; return 0; } ;;
      wkhtmltopdf)
        have pandoc && have wkhtmltopdf && { echo "wkhtmltopdf"; return 0; } ;;
      weasyprint)
        have weasyprint || python3 -c "import weasyprint" 2>/dev/null && \
          { echo "weasyprint"; return 0; } ;;
    esac
    err "지정한 엔진 '$ENGINE' 가 동작 가능한 상태가 아닙니다."
    return 1
  fi
  if have pandoc; then
    have xelatex     && { echo "xelatex";     return 0; }
    have lualatex    && { echo "lualatex";    return 0; }
    have wkhtmltopdf && { echo "wkhtmltopdf"; return 0; }
  fi
  if have weasyprint || python3 -c "import weasyprint" 2>/dev/null; then
    echo "weasyprint"; return 0
  fi
  return 1
}

# ----- --check ---------------------------------------------------------------
print_check() {
  log "도구 가용성 점검"
  for c in pandoc xelatex lualatex wkhtmltopdf weasyprint fc-list; do
    if have "$c"; then ok "$c 사용 가능"; else warn "$c 없음"; fi
  done
  if python3 -c "import weasyprint" 2>/dev/null; then
    ok "python weasyprint 모듈 사용 가능"
  else
    warn "python weasyprint 모듈 없음"
  fi
  if python3 -c "import markdown" 2>/dev/null; then
    ok "python markdown 모듈 사용 가능"
  else
    warn "python markdown 모듈 없음"
  fi
  local font
  font="$(detect_korean_font || true)"
  if [[ -n "$font" ]]; then
    ok "한글 폰트: $font"
  else
    warn "한글 폰트 미탐지 — 한글이 깨질 수 있음 (--install 권장)"
  fi
  local eng
  if eng="$(pick_engine)"; then
    ok "선택될 엔진: $eng"
  else
    warn "사용 가능한 PDF 엔진이 없음. '--install' 로 자동 설치 가능"
  fi
}

# ----- --install -------------------------------------------------------------
do_install() {
  log "PDF 빌드 의존 패키지 설치 시작"
  if ! have apt-get; then
    err "apt-get 이 없는 환경입니다. 수동 설치하세요:"
    err "  - pandoc, texlive-xetex, texlive-fonts-recommended"
    err "  - texlive-lang-cjk, fonts-nanum, fonts-noto-cjk"
    return 1
  fi
  local sudo=""
  [[ $EUID -ne 0 ]] && sudo="sudo"
  log "apt-get update"
  $sudo apt-get update -y
  log "apt-get install pandoc + texlive(xetex) + 한글 폰트"
  DEBIAN_FRONTEND=noninteractive $sudo apt-get install -y --no-install-recommends \
    pandoc \
    texlive-xetex texlive-fonts-recommended texlive-latex-recommended \
    texlive-lang-cjk \
    fonts-nanum fonts-noto-cjk \
    fontconfig
  if have fc-cache; then fc-cache -f || true; fi
  log "pip install (fallback 용) weasyprint, markdown"
  python3 -m pip install --quiet --upgrade weasyprint markdown || \
    warn "weasyprint/markdown pip 설치 실패 (fallback 일부 비활성)"
  ok "설치 완료"
}

# ----- 빌드 함수 -------------------------------------------------------------
build_pandoc_latex() {
  local engine="$1" input="$2" output="$3" font="$4"
  local font_arg=()
  if [[ -n "$font" ]]; then
    font_arg+=( -V "mainfont=$font" -V "CJKmainfont=$font" )
  fi
  local toc_arg=()
  [[ "$TOC" == "yes" || "$TOC" == "auto" ]] && toc_arg+=( --toc --toc-depth=3 )

  log "pandoc + $engine 로 빌드 중..."
  pandoc "$input" \
    --pdf-engine="$engine" \
    -V geometry:margin=1in \
    -V linkcolor:blue \
    -V documentclass=article \
    -V "papersize=a4" \
    "${font_arg[@]}" \
    "${toc_arg[@]}" \
    --highlight-style=tango \
    --resource-path="$(dirname "$input"):$(pwd)" \
    -o "$output"
}

build_pandoc_wk() {
  local input="$1" output="$2" font="$3"
  log "pandoc + wkhtmltopdf 로 빌드 중..."
  local css
  css="$(mktemp --suffix=.css)"
  cat > "$css" <<EOF
body { font-family: "${font:-NanumGothic}", sans-serif;
       font-size: 11pt; line-height: 1.5; }
code, pre { font-family: monospace; font-size: 9.5pt; }
img { max-width: 100%; height: auto; }
h1,h2,h3,h4 { page-break-after: avoid; }
table { border-collapse: collapse; }
table, th, td { border: 1px solid #ccc; padding: 4px 8px; }
EOF
  pandoc "$input" \
    --pdf-engine=wkhtmltopdf \
    --css "$css" \
    --resource-path="$(dirname "$input"):$(pwd)" \
    --pdf-engine-opt=--enable-local-file-access \
    -o "$output"
  rm -f "$css"
}

build_weasyprint() {
  local input="$1" output="$2" font="$3"
  log "weasyprint (markdown → html → pdf) 로 빌드 중..."
  python3 - "$input" "$output" "${font:-NanumGothic}" <<'PY'
import sys
from pathlib import Path
inp, out, font = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
try:
    import markdown
except ImportError:
    sys.exit("[ERR] python markdown 모듈 필요: pip install markdown")
try:
    from weasyprint import HTML, CSS
except ImportError:
    sys.exit("[ERR] python weasyprint 모듈 필요: pip install weasyprint")

text = inp.read_text(encoding="utf-8")
html_body = markdown.markdown(
    text,
    extensions=["extra", "tables", "fenced_code", "toc", "codehilite"],
)
css = f"""
@page {{ size: A4; margin: 18mm; }}
body {{ font-family: "{font}", "Noto Sans CJK KR", sans-serif;
        font-size: 11pt; line-height: 1.55; color: #222; }}
code, pre {{ font-family: monospace; font-size: 9.5pt;
             background: #f6f8fa; }}
pre {{ padding: 8px 10px; border-radius: 4px; overflow: auto; }}
img {{ max-width: 100%; height: auto; }}
h1, h2, h3 {{ page-break-after: avoid; }}
table {{ border-collapse: collapse; }}
table, th, td {{ border: 1px solid #ccc; padding: 4px 8px; }}
"""
html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>{inp.stem}</title></head><body>{html_body}</body></html>"""
HTML(string=html, base_url=str(inp.parent.resolve())).write_pdf(
    str(out), stylesheets=[CSS(string=css)],
)
print(f"[weasyprint] saved -> {out}")
PY
}

build_html_only() {
  local input="$1" output="$2" font="$3"
  log "HTML 단독 빌드 (디버그 모드)"
  if have pandoc; then
    pandoc "$input" -s --toc --toc-depth=3 \
      --resource-path="$(dirname "$input"):$(pwd)" \
      -o "$output"
  else
    python3 - "$input" "$output" "${font:-NanumGothic}" <<'PY'
import sys
from pathlib import Path
import markdown  # type: ignore
inp, out, font = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
body = markdown.markdown(inp.read_text(encoding="utf-8"),
    extensions=["extra","tables","fenced_code","toc","codehilite"])
out.write_text(f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<style>body{{font-family:'{font}',sans-serif;max-width:880px;margin:24px auto;
line-height:1.55;color:#222}} pre{{background:#f6f8fa;padding:8px;border-radius:4px}}
img{{max-width:100%}} table,th,td{{border:1px solid #ccc;border-collapse:collapse;padding:4px 8px}}
</style></head><body>{body}</body></html>""", encoding="utf-8")
print(f"[python-markdown] saved -> {out}")
PY
  fi
}

# ----- main ------------------------------------------------------------------
cd "$REPO_ROOT"

if [[ $DO_CHECK -eq 1 ]]; then
  print_check
  exit 0
fi

if [[ $DO_INSTALL -eq 1 ]]; then
  do_install || exit 1
fi

[[ -f "$INPUT" ]] || { err "입력 파일 없음: $INPUT"; exit 2; }

if [[ -z "$OUTPUT" ]]; then
  stem="$(basename "$INPUT")"; stem="${stem%.*}"
  if [[ $HTML_ONLY -eq 1 ]]; then
    OUTPUT="report/${stem}.html"
  else
    OUTPUT="report/${stem}.pdf"
  fi
fi
mkdir -p "$(dirname "$OUTPUT")"

if [[ -z "$FONT" ]]; then
  FONT="$(detect_korean_font || true)"
  [[ -n "$FONT" ]] && log "한글 폰트 자동 선택: $FONT" \
                   || warn "한글 폰트 미탐지 — 한글 표시가 깨질 수 있음"
fi

if [[ $HTML_ONLY -eq 1 ]]; then
  build_html_only "$INPUT" "$OUTPUT" "$FONT"
  ok "HTML 생성: $OUTPUT"
  exit 0
fi

if ! ENGINE_USED="$(pick_engine)"; then
  err "사용 가능한 PDF 엔진이 없습니다."
  err "다음 명령으로 자동 설치하세요:"
  err "  bash scripts/build_pdf.sh --install"
  err "또는 수동:"
  err "  apt-get install -y pandoc texlive-xetex texlive-lang-cjk fonts-nanum"
  exit 3
fi

log "선택된 엔진: $ENGINE_USED"
log "입력 : $INPUT"
log "출력 : $OUTPUT"

case "$ENGINE_USED" in
  xelatex|lualatex) build_pandoc_latex "$ENGINE_USED" "$INPUT" "$OUTPUT" "$FONT" ;;
  wkhtmltopdf)      build_pandoc_wk    "$INPUT" "$OUTPUT" "$FONT" ;;
  weasyprint)       build_weasyprint   "$INPUT" "$OUTPUT" "$FONT" ;;
  *) err "내부 오류: 알 수 없는 엔진 $ENGINE_USED"; exit 4 ;;
esac

if [[ -f "$OUTPUT" ]]; then
  sz="$(du -h "$OUTPUT" | awk '{print $1}')"
  ok "PDF 생성 완료: $OUTPUT ($sz)"
else
  err "PDF 파일이 생성되지 않았습니다."
  exit 5
fi
