import json

with open('datasets/handwrite_dataset_v3/meta/train.json') as f:
    data = json.load(f)

with open('build_dataset/cr_mapping_v2.json') as f:
    cr = json.load(f)

brush_train = set(data['train'].get('reference_images_v2', []))
brush_avail = set(data['avail'].get('reference_images_v2', []))

print(f'brush train: {len(brush_train)}')
print(f'brush avail: {len(brush_avail)}')
print(f'cr_mapping keys: {len(cr)}')

ok = 0
fail = 0
fails = []
for uni in brush_train:
    if uni in cr:
        refs = cr[uni]
        if all(r in brush_avail for r in refs):
            ok += 1
        else:
            fail += 1
            missing = [r for r in refs if r not in brush_avail]
            if len(fails) < 5:
                fails.append((uni, refs, missing))
    else:
        fail += 1

print(f'brush-only ref OK: {ok}')
print(f'brush-only ref FAIL: {fail}')
for e in fails:
    print(f'  {e[0]} -> refs={e[1]}, missing={e[2]}')

# all fonts avail check
all_avail = set()
for font, unis in data['avail'].items():
    all_avail.update(unis)
print(f'all avail unis: {len(all_avail)}')

ok2 = 0
for uni in brush_train:
    if uni in cr:
        refs = cr[uni]
        if all(r in all_avail for r in refs):
            ok2 += 1
print(f'all-font ref OK: {ok2}')
