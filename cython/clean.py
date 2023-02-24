import shutil
import os

try:
    shutil.rmtree("build")
except:
  print("Build does not exist")

dir_name = "./"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".pyd"):
        os.remove(os.path.join(dir_name, item))
    if item.endswith(".cpp"):
        os.remove(os.path.join(dir_name, item))
    if item.endswith(".c"):
        os.remove(os.path.join(dir_name, item))
    if item.endswith(".h"):
        os.remove(os.path.join(dir_name, item))