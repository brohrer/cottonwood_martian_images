import os


image_dirs = ["training", "tuning", "evaluation"]

for image_dir in image_dirs:
    filenames = os.listdir(image_dir)
    for filename in filenames:
        if filename[:8] == "download":
            i_left = filename.find("(")
            i_right = filename.find(")")
            if i_left > 0:
                file_no = int(filename[i_left + 1: i_right])
            else:
                file_no = 0
            new_filename = f"mars_{file_no:03}.jpg"
            os.rename(
                os.path.join(image_dir, filename),
                os.path.join(image_dir, new_filename)
            )
            print("moving ", filename, "to",  new_filename)
