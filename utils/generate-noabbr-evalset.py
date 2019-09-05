import pathlib
from shutil import copyfile
from tqdm import tqdm

if __name__ == '__main__':
    src_dir = 'word_imgs/best_segmentation_riccardo'
    dst_dir = 'word_imgs/riccardo_onlyabbr'

    pathlib.Path(dst_dir).mkdir(parents=True, exist_ok=True)

    root_path = pathlib.Path(src_dir)
    dst_path = pathlib.Path(dst_dir)

    for tsc_src_path in tqdm(root_path.glob('*/transcriptions/*')):
        with tsc_src_path.open('r') as f:
            tsc = f.readline().strip()

        if ('(' in tsc or ')' in tsc or "'" in tsc):
            _, _, page, tsc_dir, tsc_fnm = tsc_src_path.parts
            img_dir, img_fnm = 'images', tsc_fnm.split('.')[0]+'.png'

            img_src_path = root_path /page / img_dir
            tsc_dst_path = dst_path / page / tsc_dir
            img_dst_path = dst_path / page / img_dir

            tsc_dst_path.mkdir(parents=True, exist_ok=True)
            img_dst_path.mkdir(parents=True, exist_ok=True)

            copyfile(str(tsc_src_path), str(tsc_dst_path / tsc_fnm))
            copyfile(str(img_src_path / img_fnm), str(img_dst_path / img_fnm))

