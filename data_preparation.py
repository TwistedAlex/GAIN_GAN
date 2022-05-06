import os


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
list_commands = ['gdown https://drive.google.com/drive/folders/1-5oQoEdAecNTFr8zLk5sUUvrEUN4WHXa -O /home/shuoli/attention_env/GAIN_GAN/deepfake_data/stylegan2/ --folder',
                 'cp -rf /home/shuoli/attention_env/drive-download-20220506T181844Z/000000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'cp -rf /home/shuoli/attention_env/drive-download-20220506T181844Z/001000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'cp -rf /home/shuoli/attention_env/drive-download-20220506T181844Z/002000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'cp -rf /home/shuoli/attention_env/drive-download-20220506T181844Z/003000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'cp -rf /home/shuoli/attention_env/drive-download-20220506T181844Z/004000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'cp -rf /home/shuoli/attention_env/drive-download-20220506T181844Z/005000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'cp -rf /home/shuoli/attention_env/drive-download-20220506T181844Z/006000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'cp -rf /home/shuoli/attention_env/drive-download-20220506T181844Z/007000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'cp -rf /home/shuoli/attention_env/drive-download-20220506T181844Z/008000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'cp -rf /home/shuoli/attention_env/drive-download-20220506T181844Z/009000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/training/Neg/',
                 'cp -rf /home/shuoli/attention_env/drive-download-20220506T181844Z/010000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT/validation/Neg/'
                 ]

for com in list_commands:

    shell(com)
