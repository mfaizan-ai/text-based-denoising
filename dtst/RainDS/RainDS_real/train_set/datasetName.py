import os

folder = r"H:\aai\RainDS\RainDS_real\train_set\gt"
prefix = "raindrop_removal_"

for filename in os.listdir(folder):
    old_path = os.path.join(folder, filename)

    if os.path.isfile(old_path):
        new_name = prefix + filename
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)

print("Done")