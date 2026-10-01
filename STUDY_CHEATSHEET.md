# Optimizer Cheat Sheet

## Main progression

SGD → Momentum → NAG → AdaGrad → RMSProp → Adam → AdamW

## One-line ideas

- SGD: current gradient
- Momentum: gradient history
- NAG: look-ahead gradient
- AdaGrad: cumulative squared-gradient scaling
- RMSProp: EWMA of squared gradients
- Adam: EWMA of gradients + squared gradients + bias correction
- AdamW: Adam + decoupled weight decay

## High-yield interview questions

1. Why does Momentum help?
2. Why does NAG look ahead?
3. What problem does AdaGrad have?
4. How does RMSProp address AdaGrad's learning-rate decay?
5. What do Adam's first and second moments represent?
6. Why does Adam use bias correction?
7. What is the difference between Adam and AdamW?
