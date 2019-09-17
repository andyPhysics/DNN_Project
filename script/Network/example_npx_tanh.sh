# file name: job.condor
Executable = /home/amedina/DNN_Project/script/Network/Fit_generator_example_2.py
output = simple_all_2.out
error = simple_all_2.err
log = job_2.log
notification = never
request_cpus = 3
request_memory = 8000

#arguments = $(Item)
Requirements = has_avx =?= true

# use the current metaproject environment
getenv = True

#queue 1 Item from arguments.txt
queue 1
