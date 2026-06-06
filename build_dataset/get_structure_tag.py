
# 초성 19자
CHOSUNG = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

# 중성 21자
JUNGSUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']

# 종성 28자 (0번 = 없음)
JONGSUNG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
def decompose_hangul(syllable):
    """한글 음절을 초성·중성·종성으로 분해"""
    code = ord(syllable) - 0xAC00
    chosung = code // 588
    jungsung = (code % 588) // 28
    jongsung = code % 28
    return [CHOSUNG[chosung], JUNGSUNG[jungsung]] + ([JONGSUNG[jongsung]] if jongsung != 0 else [])



def assign_structure_tag(char):
    comps = decompose_hangul(char)
    if len(comps) == 2:      # 초성+중성
        return 0             # 단일 구조
    elif len(comps) == 3:    # 초성+중성+종성
        return 1             # 상하 구조
    else:
        return 2             # 기타


structure_tags = {}
for code in range(0xAC00, 0xD7A4):
    char = chr(code)
    structure_tags[hex(ord(char))[2:].upper()] = assign_structure_tag(char)

# 일부 확인
for k in list(structure_tags.keys())[:10]:
    print(k, structure_tags[k])

# JSON으로 저장
import json
with open("structure_tags.json", "w", encoding="utf-8") as f:
    json.dump(structure_tags, f, ensure_ascii=False, indent=2)

print("Structure tags 생성 완료!")
