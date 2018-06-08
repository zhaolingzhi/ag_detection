import os

result_file="whole_result.txt"
running_file_commands=[]
print "start"
running_file_commands.append("python gender/gender_train_"+str(0)+".py"+" >> "+result_file)
# for i in range(5):
#     running_file_commands.append("python age/age_train_"+str(i)+".py"+" >> "+result_file)
for command in running_file_commands:
    os.system(command)
print "end"
