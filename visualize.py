color_map = {
    'unlabeled': [0, [0, 0, 0]],
    'ego vehicle': [1, [0, 0, 0]],
    'rectification border': [2, [0, 0, 0]],
    'out of roi': [3, [0, 0, 0]],
    'static': [4, [0, 0, 0]],
    'dynamic': [5, [111, 74, 0]],
    'ground': [6, [81, 0, 81]],
    'road': [7, [128, 64, 128]],
    'sidewalk': [8, [244, 35, 232]],
    'parking': [9, [250, 170, 160]],
    'rail track': [10, [230, 150, 140]],
    'building': [11, [70, 70, 70]],
    'wall': [12, [102, 102, 156]],
    'fence': [13, [190, 153, 153]],
    'guard rail': [14, [180, 165, 180]],
    'bridge': [15, [150, 100, 100]],
    'tunnel': [16, [150, 120, 90]],
    'pole': [17, [153, 153, 153]],
    'polegroup': [18, [153, 153, 153]],
    'traffic light': [19, [250, 170, 30]],
    'traffic sign': [20, [220, 220, 0]],
    'vegetation': [21, [107, 142, 35]],
    'terrain': [22, [152, 251, 152]],
    'sky': [23, [70, 130, 180]],
    'person': [24, [220, 20, 60]],
    'rider': [25, [255, 0, 0]],
    'car': [26, [0, 0, 142]],
    'truck': [27, [0, 0, 70]],
    'bus': [28, [0, 60, 100]],
    'caravan': [29, [0, 0, 90]],
    'trailer': [30, [0, 0, 110]],
    'train': [31, [0, 80, 100]],
    'motorcycle': [32, [0, 0, 230]],
    'bicycle': [33, [119, 11, 32]],
    'license plate': [-1, [0, 0, 142]]
}

def get_palette(color_map):
    palette = []
    for k in color_map:
        if k != 'license plate':
            palette.append(color_map[k][1])
    palette = [b for a in palette for b in a]   # flatten
    return palette


if __name__ == '__main__':
    print(get_palette(color_map))
