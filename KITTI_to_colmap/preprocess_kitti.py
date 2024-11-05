import os
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='kitti process')
    parser.add_argument('--root_dir', type=str, default='./../KITTI/KITTI-360')
    parser.add_argument('--save_dir', type=str, default='./KITTI_to_colmap/KITTI-colmap')
    parser.add_argument('--drive_seq', type=str, default='3')
    parser.add_argument('--sequence_length', type=int, default=300)
    parser.add_argument('--discard_first', type=int, default=5)
    parser.add_argument('--discard_last', type=int, default=25)
    args = parser.parse_args()

    DRIVE = f'2013_05_28_drive_{args.drive_seq.zfill(4)}_sync'
    root_dir = args.root_dir
    save_dir = args.save_dir

    shutil.copytree(os.path.join(root_dir, 'calibration'), os.path.join(save_dir, 'calibration'))
    shutil.copytree(os.path.join(root_dir, 'data_poses'), os.path.join(save_dir, 'data_poses'))

    data_dir = f'{root_dir}/data_2d_raw/{DRIVE}'

    image_00_dir = os.path.join(data_dir, 'image_00/data_rect')
    image_01_dir = os.path.join(data_dir, 'image_01/data_rect')
    img_00_fns = sorted(os.listdir(image_00_dir))
    img_01_fns = sorted(os.listdir(image_01_dir))
    
    os.makedirs(os.path.join(save_dir, DRIVE), exist_ok=True)
    
    with open(os.path.join(root_dir, "data_poses", DRIVE, "poses.txt")) as fp:
        poses = fp.read().splitlines()

    # Make sequences
    frames = [int(p.split(" ")[0]) for p in poses]
    seq_num = (len(frames) - args.discard_first - args.discard_last) // (args.sequence_length - args.discard_first - args.discard_last)
    seq_len = len(frames) // seq_num + args.discard_first + args.discard_last
    sequences = [[0, seq_len]]
    for seq in range(seq_num - 1) :
        last_seq = sequences[-1]
        sequences.append([last_seq[0] + seq_len - args.discard_first - args.discard_last, 
                          last_seq[1] + seq_len - args.discard_first - args.discard_last])
    sequences[-1][1] = len(frames)
    print(len(frames), seq_num, seq_len, sequences)
    
    # Reorganize files
    for i in range(seq_num) :
        os.makedirs(os.path.join(save_dir, DRIVE, f"seq_{str(i+1).zfill(3)}"), exist_ok=True)
        for f in range(sequences[i][0], sequences[i][1]) :
            f_name = "%010d.png" % frames[f]
            if f_name not in img_00_fns or f_name not in img_01_fns :
                continue
            
            shutil.copy(
                os.path.join(data_dir, "image_00", "data_rect", f_name),
                os.path.join(save_dir, DRIVE, f"seq_{str(i+1).zfill(3)}", f"00_{f_name}"),
            )
            shutil.copy(
                os.path.join(data_dir, "image_01", "data_rect", f_name),
                os.path.join(save_dir, DRIVE, f"seq_{str(i+1).zfill(3)}", f"01_{f_name}"),
            )
        print(f"Drive: {DRIVE}, sequences: {i+1} done!")