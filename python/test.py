import os
import shutil
# load files

files_dir = "/home/lqq/Downloads/uploads/"
files_new_dir = "/home/lqq/Downloads/"

filesNames = os.listdir(files_dir)
for i in range(0,len(filesNames)):
    tmp_file_name = files_dir + filesNames[i]
    tmp_child_names = os.listdir(tmp_file_name)
    for j in range(0,len(tmp_child_names)):
        tmp_child_name = tmp_child_names[j]
        child_name_array = tmp_child_name.split(".")
        if os.path.exists(files_new_dir + "faceImages/"+ child_name_array[0]):
            tmp_sec_child_file = os.listdir(files_new_dir + "faceImages/"+ child_name_array[0])
            # 重命名
            os.rename(files_new_dir + "faceImages/"+ child_name_array[0] + "/" + tmp_child_name
                      ,files_new_dir + "faceImages/"+ child_name_array[0] + "/" + child_name_array[0] + "_" + str(len(tmp_sec_child_file) +1)+ "." +child_name_array[1])
        else:
            os.mkdir(files_new_dir + "faceImages/"+ child_name_array[0])
        # 移动
        print(tmp_file_name + "/"+tmp_child_name)
        print(files_new_dir + "faceImages/" + child_name_array[0] + "/" +tmp_child_name)
        shutil.move(tmp_file_name + "/"+tmp_child_name,
                    files_new_dir + "faceImages/" + child_name_array[0] + "/" +tmp_child_name)



