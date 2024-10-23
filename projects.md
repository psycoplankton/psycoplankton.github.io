---
layout: page
title: Projects

---

# Projects

I have worked on various projects since I started machine lerning in 2022. Initially when i was learning the basics, I made a CNN classifier using a resnet50, a review classification model using Bidirectional LSTM network and a time series model to predict the price of bitcoin. But that is in the past. After continuosly learning about different topics in AI, i have built some cool stuff that you might like.

## From Scratch (projects I built for fun).
- Built a large language model from scratch, just for the sake of it, which can generate without any prompt or auto-regressively complete a prompt in shakespeare style. [link](https://github.com/psycoplankton/GPT-Decoded). This helped me understand the intricacies of transoformer architecture, helping me get a deep understanding of how it works. This was necessary because you can read blogs after blogs, articles after articles or even paper after paper, as a Oppenhiemer said "theory will only take you so far :wink: 

- Built a RAG pipeline from sratch using gemma as the base LLM. this takes in any document you want and provides answers based on the queries and the context it can retrieve. [link](https://github.com/psycoplankton/RAG-from-scratch). 

- Built a VAE from scratch, although it was meant for my understanding so i did not use any special datasets, just the MNIST was enough. My main motive behind this was to understand variational inference ad how it is implemented. Just reading about it did not really do good for me.[link](https://github.com/psycoplankton/VAE-from-scratch)

- Currently I am building Diffusion model from scratch. I find the math of difusion models is a bit harder than other models we have, and it is very intuitive as well. Once I am done with it, I might write an article on it as well. 

I almost forgot to mention, I write blogs on [medium](https://medium.com/@_psycoplankton). Although I have not been very active, but i try to write whenever I get time.

## Internship Projects (projects I did for more fun 😜)

- I worked at Visual Computing and Analytics Lab, IIT BHU, for 9 months on GraphGANs and Fuzzy Neural Networks. 
    - *Problem Statememt*: GraphGANs use node embeddings as input to generate similar embeddings. This directly implies the better
     the quality of the embeddigs, the better the generated graphs will be. I worked on this problem and tried to improve the 
     information content of the node embeddings by incorporating fuzzy logic. The data we have is usually messy and we need to 
     find some hidden semantics between the data. Fuzzy logic is known to incorporate the uncertainities of an entity and thus 
     could be a potential candidate for solving this problem.

    - *Solution*: I modelled a fuzzy pre-processing layer which could be used with the existing node embedding generating 
    algorithms. This layer was based on TSK Fuzzy Logic Systems and took as input the whole embedding, assigned membrship values 
    to each embedding dimension, and then performed defuzzification to output crisp embeddings. It took 15 min to perform the 
    whole operation on a graph with 60k edges. The experiments were done using [Node2Vec](https://arxiv.org/abs/1607.00653) by A.Grover et. al., [DeepWalk](https://arxiv.org/abs/1403.6652) and [Struc2Vec](https://arxiv.org/abs/1704.03165) on Ca-GrQc and Biogrid-human datasets 


    - *Results*: I was somewhat successful in increasing the accuracy of the model as well as th F-1 score. however it only increased by 0.1%. With some improvements in the structure and a good loss function, this pre-processing layer could be turned into an algorithm which will be able to learn much better and generate good results.

- Currently (as of Oct 2024), I am working in a startup as an intern.
    - *Problem* The aim of the project is to optimize the inference pipeline and consequently the inference process for [CodeFormer](https://shangchenzhou.com/projects/CodeFormer/)
    architecure. To do this, I had to study some optimization methods like [Knowledge distillation](https://neptune.ai/blog/knowledge-distillation), [Quantization](https://www.youtube.com/watch?v=0VdNflU08yA), [Mixed Precision Training](https://arxiv.org/abs/1710.03740), [Open Neural Network Exchange](https://onnx.ai/) (ONNX) frameworks. I also grew familiar with Pytorch's API for Quantization and Mixed Precision Training. These are a little new and upgrades are always coming. I grew familiar with the code-former architecure, a very good approach for image restoration,
    I must say. 

    - *Solutions*: I devised three solutions for this optimization problem.
        -  first of all, very simple approach is to use batch inference. The original pipeline is written in one image at a time format which will introduce redundant loops and will not use the full power of GPUs in the backend, which can parallelize the computation and hence increase time.   


