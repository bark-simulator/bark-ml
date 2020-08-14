import os
from notebook import notebookapp

cwd = os.getcwd()
runfiles_dir = cwd.split("run.runfiles")[0]
bark_root = os.path.join(runfiles_dir, "run.runfiles/bark_ml/docs/gnn_practical_course_2020")

print("Changing to bark root {}".format(bark_root))
os.chdir(bark_root)

notebookapp.main()