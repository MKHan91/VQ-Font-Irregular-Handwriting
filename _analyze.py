import json, os

# Q1: cr_mapping_v2.json
with open('build_dataset/cr_mapping_v2.json') as f:
    cr = json.load(f)
print('=== Q1: cr_mapping_v2.json ===')
print(f'Total keys: {len(cr)}')
items = list(cr.items())[:3]
for k, v in items:
    print(f'  {k}: {v}')

# Q2: reference_images_v2 files
ref_dir = 'datasets/train_font_image/reference_images_v2/'
files = os.listdir(ref_dir)
print(f'\n=== Q2: reference_images_v2 ===')
print(f'Total files: {len(files)}')
# Get hex codes of the 78 characters
ref_hexes = set()
for f2 in files:
    if f2.endswith('.png'):
        char = f2.replace('.png','')
        if len(char) == 1:
            ref_hexes.add(format(ord(char), '04X'))
print(f'Unique hex codes from filenames: {len(ref_hexes)}')
print(f'First 10 hex codes: {sorted(ref_hexes)[:10]}')

# Check how many keys have ALL their values in ref_hexes
match_count = 0
for k, vals in cr.items():
    if all(v.upper() in ref_hexes for v in vals):
        match_count += 1
print(f'Keys with ALL referenced hex codes in reference_images_v2: {match_count}')

# Q3: handwrite_dataset_v3/meta/train.json
print('\n=== Q3: handwrite_dataset_v3/meta/train.json ===')
with open('datasets/handwrite_dataset_v3/meta/train.json') as f:
    train_data = json.load(f)
# Check structure
if isinstance(train_data, dict):
    print(f'Top-level keys: {list(train_data.keys())}')
    if 'train' in train_data:
        train_section = train_data['train']
        if isinstance(train_section, dict):
            print(f'Train section keys count: {len(train_section)}')
            # Find reference_images_v2 entries
            ref_v2_entries = {k: v for k, v in train_section.items() if 'reference_images_v2' in str(v)}
            print(f'Entries with reference_images_v2: {len(ref_v2_entries)}')
            if ref_v2_entries:
                first_key = list(ref_v2_entries.keys())[0]
                print(f'  Example: {first_key}: {ref_v2_entries[first_key][:2] if isinstance(ref_v2_entries[first_key], list) else str(ref_v2_entries[first_key])[:200]}')
elif isinstance(train_data, list):
    print(f'List with {len(train_data)} entries')
    # Search for reference_images_v2
    ref_entries = [e for e in train_data if 'reference_images_v2' in str(e)]
    print(f'Entries with reference_images_v2: {len(ref_entries)}')

# Q4: de_v2.json
print('\n=== Q4: de_v2.json ===')
with open('build_dataset/de_v2.json') as f:
    de = json.load(f)
if isinstance(de, dict):
    print(f'Type: dict, Total keys: {len(de)}')
    items = list(de.items())[:3]
    for k, v in items:
        print(f'  {k}: {v}')
elif isinstance(de, list):
    print(f'Type: list, Total entries: {len(de)}')
    print(f'  First 3: {de[:3]}')

# Q5: structure_tags.json
print('\n=== Q5: structure_tags.json ===')
with open('build_dataset/structure_tags.json') as f:
    stru = json.load(f)
print(f'Type: {type(stru).__name__}, Total entries: {len(stru)}')
if isinstance(stru, dict):
    # Check if ref_hexes are covered
    covered = sum(1 for h in ref_hexes if h in stru or h.lower() in stru or h.upper() in stru)
    print(f'Reference hexes covered in structure_tags: {covered}/{len(ref_hexes)}')
    items = list(stru.items())[:3]
    for k, v in items:
        print(f'  {k}: {v}')
