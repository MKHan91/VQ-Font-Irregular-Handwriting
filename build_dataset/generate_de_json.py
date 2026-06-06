import json
import os
import os.path as osp


def decompose_hangul(code):
    """
    한글 유니코드를 초성, 중성, 종성으로 분해
    
    Args:
        code: 한글 유니코드 값 (예: 0xAC00)
    
    Returns:
        tuple: (초성_인덱스, 중성_인덱스, 종성_인덱스)
    """
    # 한글 유니코드 범위: 0xAC00 ~ 0xD7A3
    HANGUL_BASE = 0xAC00
    
    if code < HANGUL_BASE or code > 0xD7A3:
        return None
    
    # 한글 조합 공식
    code_offset = code - HANGUL_BASE
    
    # 초성 19개, 중성 21개, 종성 28개
    jong_idx = code_offset % 28
    jung_idx = ((code_offset - jong_idx) // 28) % 21
    cho_idx = ((code_offset - jong_idx) // 28) // 21
    
    return cho_idx, jung_idx, jong_idx


def generate_de_json(output_path='de.json'):
    """
    한글 de.json 파일 생성
    옵션 2 방식: [초성, 중성+19, 종성+47]
    """
    de_mapping = {}
    
    # 한글 완성형 범위: AC00 ~ D7A3
    HANGUL_START = 0xAC00
    HANGUL_END = 0xD7A3
    
    for code in range(HANGUL_START, HANGUL_END + 1):
    # for char in train_chars:
        # code = ord(char)
        decomposed = decompose_hangul(code)
        
        if decomposed:
            cho_idx, jung_idx, jong_idx = decomposed
            
            # 옵션 2: 초성(0-18), 중성+19(19-39), 종성+47(47-74)
            component = [
                cho_idx,           # 초성: 0-18
                jung_idx + 19,     # 중성: 19-39
                jong_idx + 47      # 종성: 47-74
                # jong_idx + 40   # 40~67 (28개)
            ]
            
            # 16진수 문자열로 키 생성 (대문자, 0x 제외)
            hex_key = format(code, 'X')
            de_mapping[hex_key] = component
    
    # JSON 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(de_mapping, f, ensure_ascii=False, indent=2)
    
    print(f"✓ de.json 생성 완료!")
    print(f"  - 파일 위치: {output_path}")
    print(f"  - 총 글자 수: {len(de_mapping)}")
    print(f"\n샘플 (처음 5개):")
    
    # 샘플 출력
    for i, (key, value) in enumerate(list(de_mapping.items())[:5]):
        char = chr(int(key, 16))
        print(f"  {key} ({char}): {value}")


def verify_de_json(json_path='de.json'):
    """
    생성된 de.json 검증
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        de_map = json.load(f)
    
    print("\n=== 검증 결과 ===")
    
    # 테스트 케이스
    test_cases = [
        ("AC00", "가"),  # ㄱ + ㅏ + (없음)
        ("AC01", "각"),  # ㄱ + ㅏ + ㄱ
        ("B098", "나"),  # ㄴ + ㅏ + (없음)
        ("D55C", "한"),  # ㅎ + ㅏ + ㄴ
        ("AE00", "가"),  # ㄱ + ㅏ + (없음)
    ]
    
    print("\n테스트 케이스:")
    for code, expected_char in test_cases:
        if code in de_map:
            char = chr(int(code, 16))
            components = de_map[code]
            cho_idx = components[0]
            jung_idx = components[1] - 19
            jong_idx = components[2] - 47
            
            # 자모 이름
            cho_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 
                       'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
            jung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
                        'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
            jong_list = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
                        'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                        'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
            
            cho = cho_list[cho_idx] if 0 <= cho_idx < len(cho_list) else '?'
            jung = jung_list[jung_idx] if 0 <= jung_idx < len(jung_list) else '?'
            jong = jong_list[jong_idx] if 0 <= jong_idx < len(jong_list) else '?'
            
            print(f"  {code} ({char}): {components} = {cho} + {jung} + {jong}")


if __name__ == "__main__":
    # train_data_dir = "/home/dev/Project/VQ-Font/datasets/train_font_image"
    
    # train_names = []
    # for folderName in os.listdir(train_data_dir):
    #     if folderName == "reference_images": continue
        
    #     for fileName in os.listdir(osp.join(train_data_dir, folderName)):
    #         train_names.append(fileName.split('.')[0])
    
    # train_chars = list(set(train_names))
    
    # de.json 생성
    generate_de_json('./build_dataset/de_v2.json')
    
    # 검증
    verify_de_json('./build_dataset/de_v2.json')