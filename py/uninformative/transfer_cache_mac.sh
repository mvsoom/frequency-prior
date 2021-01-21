#!/bin/bash
remote_root=/Users/marnix/formant-prior/py/uninformative
local_root=$WRK/proj/formant-prior/research/py/uninformative
scp -r marnix@ai31.vub.ac.be:$remote_root/cache $local_root
