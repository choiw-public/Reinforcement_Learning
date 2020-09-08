# Reinforcement Learning 

This repository includes python codes of reinforcement learning (RL) models.
RL can be used in robotics, traffic light control, stock trading etc.
 
**Demo of trained models**
* Adventurer (vegetarian version)

![demo](demo/vegetarian/demo1.gif) ![demo](demo/vegetarian/demo2.gif) ![demo](demo/vegetarian/demo3.gif)\
The adventurer is a vegetarian, he needs to evade meats. 
 I set the surface slippery to make it hard to control. 
 Use ```Left``` and ```right``` to move. 

* Bubble Shooter

![demo](demo/bubble/demo1.gif) ![demo](demo/bubble/demo2.gif)\
Shoot the bubbles and match the color to pop. 

**Game Assets**
* Adventurer
  - Adventurer images: https://rvros.itch.io/animated-pixel-hero
  - Meat images: https://vectorpixelstar.itch.io/food?download
 * Bubble Shooter
   - Base code: https://github.com/PranavB6/Bubbles_2.0

**Dependancies**
* Python v3.7
* TensorFlow v1.15
* NumPy
* OpenCV
* PyGame

**To play a game manually**
```
run.py --game='vegetarian' --phase='manual_play'
```

**To train a RL model (more RL models are coming)**
```
run.py --game='vegetarian' --phase='train'
```

**To test a trained model**
```
run.py --game='vegetarian' --phase='test'
``` 