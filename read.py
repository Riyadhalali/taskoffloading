worker_files = open("workes.txt", "r")
print(worker_files.readline())

for worker in worker_files.readlines():
    print(worker+" is cool")
worker_files.close()
