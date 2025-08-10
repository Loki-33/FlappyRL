# FlappyRL

To better understand the working of an RL training Pipeline, I implemented from scratch the FlappyBird game, where the env is hard coded along with the trianing algo
1. flappy.py -> contains the env, it returns a grayscale image, with stacked frames of 4, for better training, kinda does the same thing as AtariPreprocessing
2. main.py -> the main training algo used PPO
3. test.py ->testing the trained model, by wayching it play the game live


NOTE:
1. Tried my best at implementing but sadly it could barely passed the first pipe, tried various reward method that would make it learn better positioning but didnt help much
2. Also tried various hyperparameters tuning to help the triaing faster but that was in vain too
3. My inuition is that maybe the reward needs to be more nicely coded to ensure positioning and maybe the env needs some changing(future)
