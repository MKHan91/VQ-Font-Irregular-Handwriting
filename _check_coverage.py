import json, os, glob

with open('build_dataset/cr_mapping_v2.json') as f:
    cr = json.load(f)
print('cr_mapping keys:', len(cr))

imgs = glob.glob('datasets/train_font_image/reference_images_v2/*.png')
chars = [os.path.basename(p).split('.')[0] for p in imgs]
hex_set = set(hex(ord(c))[2:].upper() for c in chars)
print('ref v2 chars:', len(chars))

infer_count = sum(1 for uni, deps in cr.items() if set(deps).issubset(hex_set))
print('inferable:', infer_count, '/ 11172')



