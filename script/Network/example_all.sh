# file name: job.condor
Executable = /home/amedina/DNN_Project/script/Network/all.py
output = all.out
error = all.err
log = all.log
notification = never
request_cpus = 4
request_memory = 8000

#arguments = $(Item)
Requirements = has_avx =?= true

# use the current metaproject environment
getenv = True

arguments = $(Item)

queue 1 Item from all.txt

