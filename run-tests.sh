#!/bin/bash
#mpiexec -np 25 /home/jspringer/OpenPV/build/tests/BasicSystemTest/Release/BasicSystemTest -p benign.params -t 2 -batchwidth 25 -rows 1 -columns 1
mpiexec -np 25 /home/jspringer/OpenPV/build/tests/BasicSystemTest/Release/BasicSystemTest -p benign-downscale.params -t 2 -batchwidth 25 -rows 1 -columns 1
#mpiexec -np 25 /home/jspringer/OpenPV/build/tests/BasicSystemTest/Release/BasicSystemTest -p adversarial.params -t 2 -batchwidth 25 -rows 1 -columns 1
#mpiexec -np 25 /home/jspringer/OpenPV/build/tests/BasicSystemTest/Release/BasicSystemTest -p adversarial-downscale.params -t 2 -batchwidth 25 -rows 1 -columns 1
#mpiexec -np 25 /home/jspringer/OpenPV/build/tests/BasicSystemTest/Release/BasicSystemTest -p noisy.params -t 2 -batchwidth 25 -rows 1 -columns 1
#mpiexec -np 25 /home/jspringer/OpenPV/build/tests/BasicSystemTest/Release/BasicSystemTest -p noisy-downscale.params -t 2 -batchwidth 25 -rows 1 -columns 1
