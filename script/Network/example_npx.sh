# file name: job.condor
Executable = /home/amedina/DNN_Project/script/Network/DNN_model.py
output = model1.out
error = model1.err
log = job.log
notification = never
request_cpus = 4
request_memory = 15000

#arguments = $(Item)
Requirements = has_avx =?= true

# use the current metaproject environment
getenv = True

arguments = $(Item)

queue 1 Item from arguments.txt

