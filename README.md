simple example for cartpole imitation learning


---
data collection
---

python3 collect.py 


you can change number of steps and episodes by changing parameter (num_episodes, max_steps)

--
training(500 steps x 10 episodes)
--

python3 train_10x500.py


--
training(50 steps x 10 episodes)
--

python3 train_10x50.py

--
training(50 steps x 10 episodes, optimization)
--

python3 train_symmetric.py --augment --imp_weight



--
evaluation(500 steps x 10 episodes)
--

python3 eval_10x500.py



--
evaluation(50 steps x 10 episodes)
--

python3 eval_10x50.py

--
evaluation(50 steps x 10 episodes, optimization)
--

python3 eval_10x50_symmetric.py

