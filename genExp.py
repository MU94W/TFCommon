import time, os, codecs

now = time.localtime()

exp_nm = "Y%d_M%d_D%d_h%d_m%d_s%d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

exp_description = ""
exp_description_tmp = input("实验描述:\n")
while exp_description_tmp != "EOF":
    exp_description += exp_description_tmp + "\n"
    exp_description_tmp = input()
if not exp_description:
    print("[*]无实验描述，不能生成新实验!")
    exit(1)
exp_description += "EOF"

os.mkdir(exp_nm)
with codecs.open(os.path.join(exp_nm, "README.md"), "w", "utf-8") as f:
    f.write(exp_description)

split_line = "*" * 10 + "\n"
exp_nm_des = split_line + exp_nm + "\n" + exp_description + "\n" + split_line

with codecs.open("ExpList.md", "a", "utf-8") as f:
    f.write(exp_nm_des)

