# file name: job.condor
Executable = /home/amedina/DNN_Project/script/Network/Energy_model.py
output = energy.out
error = energy.err
log = job_3.log
notification = never
request_cpus = 3
request_memory = 8000

#arguments = $(Item)
Requirements = has_avx =?= true

# use the current metaproject environment
getenv = True

#queue 1 Item from arguments.txt
queue 1
