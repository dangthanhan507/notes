---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title:  ML (Git Gud) Resources
math: katex
has_children: true
---

This is a collection of resources I've used to try and git gud in ML. Hopefully, I can come back to these or someone will find use in it. I'm not going to list papers I've read since I think it's much better to go through compiled information first. This isn't a sufficient condition to get started in ML Research, but I think it's necessary to have gathered skills from these resources to be able to do research. 


<!-- # Display a link -->
## 1. Get started (Basics)
1. Andrej Karpathy Zero to Hero - [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rO2c7g2i0a3d8b6e1a5c3d4)
    - Indirectly learn how pytorch works under the hood.
    - ML Concepts: Backprop, Gradient Descent, Weight Init, Train/Val, BatchNorm/LayerNorm, Attention ...etc.
    - Training Concepts (gradient accumulation, mixed precision, distributed pytorch, )
2. UvA DL - [Notebook-Website](https://uvadlc-notebooks.readthedocs.io/en/latest/)
    - Deep Learning 1 (PyTorch): exposes you to popular architectures
    - Training Models at Scale
        - takes stuff you learned from (1.) and levels it up.
3. CS294 - [Website](https://sites.google.com/view/berkeley-cs294-158-sp24/home)
    - Taught by Pieter Abbeel @ UC Berkeley. 
    - Homeworks are from scratch mini-paper implementations per question. (most valuable)
    - Makes you more solid in your PyTorch and implementation intuition!

## 3. RL Stuff
1. CS285 - [Website](https://sites.google.com/berkeley.edu/cs285/)
    - Taught by Sergey Levine @ UC Berkeley.
    - This is a course on RL and covers the basics of RL, policy gradients, actor-critic, PPO, DDPG, SAC, and more.
    - The homeworks are very well designed and will help you understand the concepts better.
2. UCL David Silver DeepMind - [Youtube](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
3. Pulkit 6.8200 - [Website](https://pulkitag.github.io/6.8200/lectures)
    - This is a course on RL+IL but for robotics.
    - Well designed to help you setup code in this domain.
4. 

## 2. Becoming a good engineer
1. Full-stack DL - [Website](https://fullstackdeeplearning.com/)
    - This is a course that tries to make you become a better ML Engineer.
    - These are my takeaways
        - PyTorch Lightning intro.
        - W&B (Weights and Biases) for experiment tracking.
        - Gradio for model deployment.
2. Missing Semester of CS Education - [Website](https://missing.csail.mit.edu/)
    - If you don't know stuff from this course, you're missing out!

## 3. Learning more!
1. UvA DL - [Notebook-Website](https://uvadlc-notebooks.readthedocs.io/en/latest/)
    - Deep Learning 2: Learn some super niche stuff that will make you overall better.
2. Diffusion - [Website](https://diffusion.csail.mit.edu/)
    - Gives mathematical details of diffusion models and flow matching.
    - Labs help you train a diffusion model from the ground up.
3. Machine Learning & Simulation - [Youtube](https://www.youtube.com/@MachineLearningSimulation)
    - This guy does lots of ML Physics stuff and goes into classical techniques on dynamical systems (FEM from scratch).
    - He also is a huge fan of JAX and has a lot of videos on it.


## 4. Theoretical Stuff
1. CS274B (UCI) - [Canvas](https://canvas.eee.uci.edu/courses/64508/assignments/syllabus)
    - This is a course taught by Erik Sudderth @ UC Irvine (I think better than Stanford)
    - It goes into probabilistic graphical models (just more bayesian probability)
    - Going through material + homework will get you very comfortable with probability (useful for research)
2. Matrix Calculus - [Youtube](https://www.youtube.com/playlist?list=PLUl4u3cNGP62EaLLH92E_VCN4izBKK6OE)
3. Transformer Circuits! - [Website](https://transformer-circuits.pub)
    - Listed are insightful posts I think are worth reading.
    - [A mathematical framework for Transformer Circuits.](https://transformer-circuits.pub/2021/framework/index.html)
        - Comes with exercises
    - [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
    - [Circuit Tracking: Revealing Computational Graphs on LMs](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
    - [Biology of LLMs](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
4. Interpreting GPT - [Website](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
5. Vision Circuits! - [Website](https://distill.pub/2020/circuits/)

## 5. Learning more about ML Research
1. Yannic Kilcher - [YouTube](https://www.youtube.com/@YannicKilcher)
    - Better understanding of ML PhD life - [Video](https://www.youtube.com/watch?v=rHQPBqMULXo)
    - He has a lot of videos on ML research papers. He does a good job of explaining the concepts and the math behind them.
    - I read the same papers he does a video on and compare my analysis to his to see what I'm missing.
2. Aleksa Gordic - [Youtube](https://www.youtube.com/c/TheAIEpiphany)
    - This guy does the same sort of content as Yannic Kilcher for papers. I like having another datapoint to compare. Also multiple sources help you keep up to date with current hype ML stuff.
    - Exclusive stuff from his channel:
        - Deep dives into codebases from papers he also reviews! (DINO, DETR, Diffusion,...etc.)
        - Livestreams himself implementing super large models.
