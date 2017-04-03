import os, fnmatch, os.path
 
array = []
filename = "/home/sarah/Documents/AsianClassifierImages/Positives" #Can be changed it to test1
counter = 0

def Dirsearch(filename, counter):
    if os.path.isdir(filename):
        for file in os.listdir(filename):
            #print file + "\n"
            sub_path = filename + os.sep + file
            if os.path.isdir(sub_path):
                #counter +=100 #once it's all renamed, we need to move to all one directory & comment this line out
                Dirsearch(sub_path, counter)
            else:
                if fnmatch.fnmatch(file, '*.jpg'):
                    file_name, file_extension = os.path.splitext(file)
                    #print file
                    #readfile = open(sub_path, 'r')
                    counter +=1
                    newFileName = 'pos{}.jpg'.format(counter)
                    os.rename(filename + os.sep + file, filename + os.sep + newFileName)
                    #print file
 
Dirsearch(filename, counter)

# import os
# import fnmatch
# # Lets change working directory to the pictures folder
# #os.chdir("/home/rbruce/caffeV2/adbloq/images/")
# os.chdir("/home/rbruce/caffeV2/adbloq/training/NegativeImages/combined")
# # confirm working directory by printing it out
# print os.getcwd()

# # loop over the files in the working directory and printing them out
# # for file in os.listdir('C:'):
#  # print file
# i = 0
# for file in os.listdir('/home/rbruce/caffeV2/adbloq/training/NegativeImages/combined'):
#  file_name, file_extension = os.path.splitext(file)
#  i += 1
#  new_file_name = 'neg{}.jpg'.format(i)
#  os.rename(file, new_file_name)

# # print file_name, file_extension

# # #/home/rbruce/caffeV2/adbloq/images