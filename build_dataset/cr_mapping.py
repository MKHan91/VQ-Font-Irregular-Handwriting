import json
import random
import os
import os.path as osp

from collections import deque
from collections import Counter


def searchComponents(xi, T, max_depth):
    """
    xi: content character
    T: single-level decomposition table, dict {char: [components]}
    max_depth: 최대 탐색 깊이
    """
    queue = deque([(xi, 0)])
    visited = set()
    components = set()
    
    while queue:
        node, depth = queue.popleft()
        if node in visited or depth > max_depth:
            continue
        visited.add(node)
        
        if node in T:
            comps = T[node]
            components.update(comps)
            for c in comps:
                queue.append((c, depth + 1))
    
    return components

def generate_reference_set(X, T, Nref, max_depth):
    """
    X: content character list
    T: decomposition table
    Nref: reference set 최대 크기
    max_depth: component tree 탐색 최대 깊이
    """
    U = []  # reference set
    C = set()  # 커버 가능한 component 집합
    
    for xi in X:
        ci = searchComponents(xi, T, max_depth)
        if not ci.issubset(C):
            C.update(ci)
            U.append(xi)
            if len(U) >= Nref:
                break
    
    return U, C


"""
Content 글자마다 reference 3개를 선택할 때 단순 랜덤이 아니라 component 커버리지 기반으로 선택합니다.

즉:
1. Content 글자를 구성하는 획(component)
2. Reference 글자들이 가지고 있는 component

이 정보를 바탕으로, content 글자의 component를 가장 많이 커버하는 reference 글자를 선택합니다.
"""


def select_references(content_char, ref_chars, T, top_k=3):
    """
    content_char: Content 글자
    ref_chars: Reference 글자 리스트
    T: decomposition table {char: [components]}
    top_k: 선택할 reference 수
    """
    content_comps = Counter(T[content_char])
    
    # reference별 겹치는 component 개수 계산
    scores = []
    for ref in ref_chars:
        ref_comps = Counter(T[ref])
        # 겹치는 component 수 계산 (min count 합)
        score = sum(min(content_comps[c], ref_comps.get(c,0)) for c in content_comps)
        scores.append((ref, score))
    
    # score 내림차순 정렬 후 상위 top_k 선택
    scores.sort(key=lambda x: x[1], reverse=True)
    selected_refs = [ref for ref, _ in scores[:top_k]]
    
    return selected_refs


def decompose_hangul(syllable):
    """한글 음절을 초성·중성·종성으로 분해"""
    code = ord(syllable) - 0xAC00
    chosung = code // 588
    jungsung = (code % 588) // 28
    jongsung = code % 28
    return [CHOSUNG[chosung], JUNGSUNG[jungsung]] + ([JONGSUNG[jongsung]] if jongsung != 0 else [])


if __name__ == "__main__":
    # 초성 19자
    CHOSUNG = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

    # 중성 21자
    JUNGSUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']

    # 종성 28자 (0번 = 없음)
    JONGSUNG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

    start = 0xAC00
    end = 0xD7A3
    T = {}
    for code in range(start, end + 1):
        char = chr(code)
        T[char] = decompose_hangul(char)

    # train_data_dir = "/home/dev/VQ-Font/datasets/train_font_image"

    # train_names = []
    # for folderName in os.listdir(train_data_dir):
    #     for fileName in os.listdir(osp.join(train_data_dir, folderName)):
    #         train_names.append(fileName.split('.')[0])
    # train_chars = list(set(train_names))

    # T = {}
    # for char in train_chars:
    #     T[char] = decompose_hangul(char)

    # --------------------------
    # 1️⃣ content character 리스트
    # 예: 한글 완성형 11,172자
    content_chars = [chr(u) for u in range(start, end + 1)]
    # content_chars = train_chars

    # --------------------------
    # 2️⃣ reference character 리스트 (사용자가 갖고 있는 18개)
    # 예: 18개의 reference 글자를 HEX로 제공했다고 가정
    ref_dir = "/home/dev/Project/VQ-Font/datasets/train_font_image/reference_images_v2"
    ref_chars = [char[:-4] for char in os.listdir(ref_dir) if not (('2' in char) or ('3' in char))]

    # --------------------------
    # 3️⃣ C-R mapping 생성
    cr_mapping = {}
    random.seed(42)  # 재현성
    for idx, ch in enumerate(content_chars):
        print(f'\r {ch} - {idx+1}/{len(content_chars):<100}', end='')
        # ref_chars에서 3개 랜덤 선택 (중복 없음)
        # selected_refs = random.sample(ref_chars, 3)
        selected_refs = select_references(ch, ref_chars, T, top_k=3)
        
        # HEX 코드로 변환
        cr_mapping[hex(ord(ch))[2:].upper()] = [hex(ord(r))[2:].upper() for r in selected_refs]

    # --------------------------
    # 4️⃣ JSON 저장
    print()
    with open("./build_dataset/cr_mapping_v2.json", "w", encoding="utf-8") as f:
        json.dump(cr_mapping, f, ensure_ascii=False, indent=2)

    print("C-R mapping 생성 완료!")
