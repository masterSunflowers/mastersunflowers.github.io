---
title: Transformers
author: Thieu Luu
date: 2025-21-01
category: architecture
layout: post
---
# Transformers

# Introduction

Transformers is a deep learning architecture introduced in the paper "Attention Is All You Need", which is written by Google Research team. This architecture is the core of various improvements in both Natural Language Processing and Computer Vision.

# Overview

In the original paper, Transformers include two main components. The first component consists of 6 encoders stacked together. And the second includes 6 decoders.

![overview](../images/Transformers/overview.jpg)
# Embedding layer

![embedding_layer](../images/Transformers/overview.jpg)

Input will go through an embedding layer first in both encoder and decoder

# Position encoding

Assume input includes 7 tokens like above image, we have position encoding will be a matrix with size 7 * `d_embedding`  in which, each row will be encoding of token's position.

$$
PE(pos, 2i) = sin(pos / 10000^{2i/d_{embedding}}) 
$$

$$
PE(pos, 2i+1) = cos(pos/10000^{2i/d_{embedding}})
$$

 In which, `i` takes a integer value from `0` to `d_embedding / 2` 

For example, with `pos = 0` , `d_embedding = 512` , we have:

$$
PE(0, 0) = sin(0 / 10000^0) = 0 \\
PE(0, 1) = cos (0 / 10000^0) = 1 \\
PE(0, 2) = sin(0 / 10000^{2/512}) = 0 \\
...
$$

The encoded vector for row 0 will be `[0, 1, 0, 1, …, 0, 1]`
