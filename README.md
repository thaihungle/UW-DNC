# UW-DNC
source code for paper Learning to Remember More with Less Memorization
https://openreview.net/forum?id=r1xlvi0qYm  
reference: https://github.com/Mostafa-Samir/DNC-tensorflow  

# Synthetic tasks
Training:  
1. regular DNC:  
python synthetic_task.py --task=copy --mode=train
2. UW DNC:  
python synthetic_task.py --task=copy --mode=train --hold_mem_mode=2
3. CUW DNC:
python synthetic_task.py --task=copy --mode=train --hold_mem_mode=2 --cache_attend_dim=16 --cache_size=10 --hidden_dim=100  
Testing:  
--mode=test


