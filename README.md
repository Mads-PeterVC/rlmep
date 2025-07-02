# Exploring reinforcement learning in materials science

| Subject | File | Colab |
| --------  | ---- | ------ |
| Test     | [test_notebook.ipynb](https://github.com/Mads-PeterVC/rlmep/blob/main/exercises/test_notebook.ipynb) | [ ![Open in Google Colab] ](https://colab.research.google.com/github/Mads-PeterVC/rlmep/blob/main/exercises/test_notebook.ipynb#) |
| Exercise 1  | [exercise_1_qlearning_frozenlake.ipynb](https://github.com/Mads-PeterVC/rlmep/blob/main/exercises/exercise_1_qlearning_frozenlake.ipynb) | [ ![Open in Google Colab] ](https://colab.research.google.com/github/Mads-PeterVC/rlmep/blob/main/exercises/exercise_1_qlearning_frozenlake.ipynb#) |
| Exercise 2  | [exercise_2_mep_env.ipynb](https://github.com/Mads-PeterVC/rlmep/blob/main/exercises/exercise_2_mep_env.ipynb) | [ ![Open in Google Colab] ](https://colab.research.google.com/github/Mads-PeterVC/rlmep/blob/main/exercises/exercise_2_mep_env.ipynb#) |

[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg

## Local setup

If you prefer to work elsewhere than Colab you can download the notebooks, create an environment and run 

```
pip install "rlmep[local] @ git+https://github.com/Mads-PeterVC/rlmep.git"
```

To install all dependencies and the supporting code.

# Additional resources: 

These are not required or necessary for the tutorials, but if you'd like to reinforce your learning they are good next steps

- The bible of reinforcement learning is [Sutton & Barto](http://incompleteideas.net/book/the-book-2nd.html)
- An excellent youtube series on RL by [Mutual Information](https://www.youtube.com/watch?v=NFo9v_yKQXA&list=PLzvYlJMoZ02Dxtwe-MmH4nOB5jYlMGBjr&ab_channel=MutualInformation)
- Concise write-ups on Lil'Log by Lilian Weng covering an [overview on RL](https://lilianweng.github.io/posts/2018-02-19-rl-overview/), a long post on [policy gradient methods](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) and even some [implementations](https://lilianweng.github.io/posts/2018-05-05-drl-implementation/#actor-critic) off some algorithms. 
- There exists a variety of Towards Data Science, Medium etc articles on RL - but they often skip the most difficult steps in their explanations so don't be surprised if you dont learn much from them. 
- [Spinning Up](https://spinningup.openai.com/en/latest/index.html) contains a nice but mathy introduction to policy gradient methods. 
- Original papers for algorithms can be helpful, but honestly they often express themselves in the most complicated way possible for their given algorithm - perhaps this increases the chance of being accepted for NIPS. You can look them up if you want to. 
- The internet, with the correct search terms it is often possible to find good questions an answers from e.g. stack exchange on RL topics. 

# Code libraries
There are several rather good code libraries containing implementations of many of the best RL algorithms, these are usually written to interface with `gynamsium` environments. These are important to keep in mind  for several reasons, in some cases they might provide an implementation that can be taken off the shelf. In other cases it might be nice to benchmark your own implementation against a known good implementation, and have access to the code of their implementations is also nice for debugging to see if some 
concept has been misunderstood. 

- [StableBaselines](https://github.com/DLR-RM/stable-baselines3) has implementations of many algorithms. 
- [CleanRL](https://github.com/vwxyzjn/cleanrl) have particularly clean and self-contained codes for many algorithms, that makes it easier to read and perhaps use as a starting point for a tweaked version for your use case. 
- [Spinning up](https://spinningup.openai.com/en/latest/index.html) has implementations of several Policy gradient methods. 
- [Gymnasium](https://gymnasium.farama.org/) The standard for implementing environments. 
- [RLlib](https://docs.ray.io/en/latest/rllib/index.html) This is from the AnyScale team, that also are the developers of Ray, and advertise 'Industrial grade RL', this library offers MANY algorithms but it is not too easy to get started with but probably 
one of the highest performing libraries. 
- [Acme](https://github.com/deepmind/acme) developed by Deepmind provides a toolbox for building agents (Though I have not really tried it)
- [TorchRL](https://docs.pytorch.org/rl/stable/index.html) Official Pytorch implementations of various pieces of RL algorithms, and some good looking [tutorials](https://docs.pytorch.org/rl/stable/tutorials/coding_ppo.html). 
