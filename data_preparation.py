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
list_commands = ['cp -rf /server_data/image-research/20220505_ffhq_11K/images_1024x1024/00000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT_1kT/training/Pos/',
                 'cp -rf /server_data/image-research/20220505_ffhq_11K/images_1024x1024/01000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT_1kT/training/Pos/',
                 'cp -rf /server_data/image-research/20220505_ffhq_11K/images_1024x1024/02000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT_1kT/training/Pos/',
                 'cp -rf /server_data/image-research/20220505_ffhq_11K/images_1024x1024/03000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT_1kT/training/Pos/',
                 'cp -rf /server_data/image-research/20220505_ffhq_11K/images_1024x1024/04000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT_1kT/training/Pos/',
                 'cp -rf /server_data/image-research/20220505_ffhq_11K/images_1024x1024/05000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT_1kT/training/Pos/',
                 'cp -rf /server_data/image-research/20220505_ffhq_11K/images_1024x1024/06000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT_1kT/training/Pos/',
                 'cp -rf /server_data/image-research/20220505_ffhq_11K/images_1024x1024/07000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT_1kT/training/Pos/',
                 'cp -rf /server_data/image-research/20220505_ffhq_11K/images_1024x1024/08000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT_1kT/training/Pos/',
                 'cp -rf /server_data/image-research/20220505_ffhq_11K/images_1024x1024/09000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT_1kT/training/Pos/',
                 'cp -rf /server_data/image-research/20220505_ffhq_11K/images_1024x1024/10000/* /home/shuoli/attention_env/GAIN_GAN/deepfake_data/data_s2_20kT_1kT/validation/Pos/'
                 ]
for com in list_commands:

    shell(com)
