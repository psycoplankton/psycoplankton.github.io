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
    whole operation on a graph with 60k edges. The experiments were done with [Node2Vec](https://arxiv.org/abs/1607.00653) by A.Grover et. al., [DeepWalk](https://arxiv.org/abs/1403.6652) and [Struc2Vec](https://arxiv.org/abs/1704.03165) on Ca-GrQc and Biogrid-human datasets. 
    
    - *Results*: I was somewhat successful in increasing the accuracy of the model as well as th F-1 score. however it only increased by 0.1%. With some improvements in the structure and a good loss function, this pre-processing layer could be turned into an algorithm which will be able to learn much better and generate good results.


- Currently (as of Oct 2024), I am working at [BingeClip.Ai](https://www.bingeclip.ai/) as a Machine Learning Engineer Intern.
    - *Problem*: The aim of the project is to optimize the inference pipeline and consequently the inference process for [CodeFormer](https://shangchenzhou.com/projects/CodeFormer/) architecure. To do this, I had to study some optimization methods like [Knowledge distillation](https://neptune.ai/blog/knowledge-distillation), [Quantization](https://www.youtube.com/watch?v=0VdNflU08yA), [Mixed Precision Training](https://arxiv.org/abs/1710.03740), [Open Neural Network Exchange](https://onnx.ai/) (ONNX) frameworks. I also grew familiar with Pytorch's API for Quantization and Mixed Precision Training. These are a little new and upgrades are always coming. I grew familiar with the code-former architecure. The authors have very appropriately used transformer module to capture facial relationships. This would not be possible with a multi layer perceptron.

    - *Solutions*: I devised three solutions for this optimization problem.
        -  first of all, very simple approach is to use batch inference. The original pipeline is written in 'one image at a time' format, which will introduce redundant loops and will not use the full power of GPUs in the backend, which can parallelize the computation and hence increase time. To avoid redundant loops, I first introduced batches of test inputs and performed inference in order to save time.
        - Second step was to introduce qunatization. Quantization is known to increase inference speeds at minimal accuracy losses. So i experimented with Post Training static quantization using pytorch's fx graph api. This is the latest API right now and runs without any errors. You can check out my code [here](https://github.com/psycoplankton/CodeFormer_optimization).
        - After quantization, i also experimented with ONNX framework. And it definitely works fast. I 100% recommend it. I did not have to do much work, just a couple of API calls and sorted. 
        - One more framework, I am thinking of experimenting with is tensorrt. From what I have heard and read, this helps in efficient resource allocation on GPUs. This could be a good approach, given that the scope of reducing time with modifications is not much.
        - If the above approaches do not work out, I would experimet with Knowledge distillation and pruning techniques. 


- I am also working as a research intern at Speech and Language Lab, NTU Singapore.

    - *Problem*: working on synthetic interview dataset generation using LLMs for depression detection. I am using DAIC-WOZ dataset which is a record of 187 interviews taken by an agent Ellie(AI) and human users. The problem is this dataset is too small for the LLM to be able to learn anything from it, therefore some other sources of relevant information has to be included here in order to be able to learn from the interviews. 
    - *Solution*: The approach I am currently working on includes knowledge graphs and Emotion Dynamics.
        - I reviewed some LLM parameter efficient fine-tuning techniques like LoRA, LLaMA Adapter and knowledge graphs based augmentation techniques. Also reviwed KG-BERT architecture. This could potentially be used to augment the relevant embeddings with the input data embeddings. 
        - Currently, I am working on a approach which includes ECoK Knowledge Graph, which is a knowledge graph which captures text based on emotional semantics in text. This could be used with KG-BERT to generate embeddings, which could then be augmented with input embeddings from the DAIC-WOZ dataset. Also, I am considering using using [Utterence Emotion Dynamics](https://arxiv.org/abs/2310.17369)(Daniela et. al 2023) to anotate the text with emotional metrics which could also help in giving additional information for fine-tuning.  



