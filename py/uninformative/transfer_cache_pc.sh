#!/bin/bash
remote_root=/home/marnix/formant-prior/py/uninformative
local_root=$WRK/proj/formant-prior/research/py/uninformative
scp -r marnix@ai30.vub.ac.be:$remote_root/cache $local_root
