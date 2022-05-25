import os
import pathlib


def shell(commands, warn=True):
    """Executes the string `commands` as a sequence of shell commands.

       Prints the result to stdout and returns the exit status.
       Provides a printed warning on non-zero exit status unless `warn`
       flag is unset.
    """
    file = os.popen(commands)
    print(file.read().rstrip('\n'))
    exit_status = file.close()
    if warn and exit_status != None:
        print(f"Completed with errors. Exit status: {exit_status}\n")
    return exit_status


def bulk_create_softlinks(abs_source, dest):
    cur_command = 'cp -rs ' +  abs_source + ' ' + dest
    shell(cur_command)

def select_clo_far_heatmaps(heatmap_home_dir, input_path_heatmap, log_name):
    input_path_heatmap_pos = input_path_heatmap + "/Pos/"
    input_path_heatmap_neg = input_path_heatmap + "/Neg/"
    heatmap_home_dir = heatmap_home_dir + "heatmap_output/" + log_name + "/"
    output_path_heatmap_pos_cl = heatmap_home_dir + "/Pos_Fake_0/" + "/50_closest/"
    output_path_heatmap_pos_fa = heatmap_home_dir + "/Pos_Fake_0/" + "/50_farthest/"
    output_path_heatmap_neg_cl = heatmap_home_dir + "/Neg_Real_1/" + "/50_closest/"
    output_path_heatmap_neg_fa = heatmap_home_dir + "/Neg_Real_1/" + "/50_farthest/"
    pathlib.Path(output_path_heatmap_pos_cl).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path_heatmap_pos_fa).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path_heatmap_neg_cl).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path_heatmap_neg_fa).mkdir(parents=True, exist_ok=True)

    pos_heatmaps = os.listdir(input_path_heatmap_pos)
    neg_heatmaps = os.listdir(input_path_heatmap_neg)
    pos_heatmaps.sort()
    neg_heatmaps.sort()

    all_pos_files = os.listdir("/home/shuoli.../")
    mask_images = [file for file in all_pos_files if 'm' in file]
    for file in mask_images:
        command = 'mv ' + input_path_heatmap_pos + file + ' ' + input_path_heatmap_pos + file[:-6] + 'm.png'
        os.system(command)

# list_commands = ['cp -rf /server_data/image-research/20220505_ffhq_11K/images1024x1024/00000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Pos/',
#                  'cp -rf /server_data/image-research/20220505_ffhq_11K/images1024x1024/01000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Pos/',
#                  'cp -rf /server_data/image-research/20220505_ffhq_11K/images1024x1024/02000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Pos/',
#                  'cp -rf /server_data/image-research/20220505_ffhq_11K/images1024x1024/03000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Pos/',
#                  'cp -rf /server_data/image-research/20220505_ffhq_11K/images1024x1024/04000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Pos/',
#                  'cp -rf /server_data/image-research/20220505_ffhq_11K/images1024x1024/05000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Pos/',
#                  'cp -rf /server_data/image-research/20220505_ffhq_11K/images1024x1024/06000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Pos/',
#                  'cp -rf /server_data/image-research/20220505_ffhq_11K/images1024x1024/07000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Pos/',
#                  'cp -rf /server_data/image-research/20220505_ffhq_11K/images1024x1024/08000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Pos/',
#                  'cp -rf /server_data/image-research/20220505_ffhq_11K/images1024x1024/09000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Pos/',
#                  'cp -rf /server_data/image-research/20220505_ffhq_11K/images1024x1024/10000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/validation/Pos/'
#                  ]
# 'gdown https://drive.google.com/drive/folders/1-5oQoEdAecNTFr8zLk5sUUvrEUN4WHXa -O /home/shuoli/attention_env/GAIN_GAN/deepfake_data/stylegan2/ --folder',
list_commands = [
                 'mv -f /home/shuoli/attention_env/drive-download-20220506T181844Z/000000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'mv -f /home/shuoli/attention_env/drive-download-20220506T181844Z/001000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'mv -f /home/shuoli/attention_env/drive-download-20220506T181844Z/002000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'mv -f /home/shuoli/attention_env/drive-download-20220506T181844Z/003000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'mv -f /home/shuoli/attention_env/drive-download-20220506T181844Z/004000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'mv -f /home/shuoli/attention_env/drive-download-20220506T181844Z/005000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'mv -f /home/shuoli/attention_env/drive-download-20220506T181844Z/006000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'mv -f /home/shuoli/attention_env/drive-download-20220506T181844Z/007000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'mv -f /home/shuoli/attention_env/drive-download-20220506T181844Z/008000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'mv -f /home/shuoli/attention_env/drive-download-20220506T181844Z/009000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'mv -f /home/shuoli/attention_env/drive-download-20220506T181844Z/010000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/validation/Neg/'
                 ]



for com in list_commands:

    shell(com)
