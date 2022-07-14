<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Jiahui-Zhang-20/POStagging">
    <img src="https://d33wubrfki0l68.cloudfront.net/d5cbc4b0e14c20f877366b69b9171649afe11fda/d96a8/assets/images/bigram-hmm/pos-title.jpg" alt="Logo" width="600" height="200">
  </a>

  <h1 align="center">Part of Speech Tagging using Hidden Markov Models</h1>

<!-- ABOUT THE PROJECT -->
## About The Project

This program labels the words of a sentence with the corresponding "parts of speech" tags. According to the Merriam-Webster dictionary, part of speech is "a traditional class of words (such as adjectives, advers, nouns, and verbs) distinguished according to the kind of idea denoted and the function performed in a sentence.

For example, consider the sentence "Ocaml is an amazing functional language". The corresponding POS sequence would be "noun verb determiner adjective adjective noun".

Part of speech tagging is a fundamental concept in natural language processing. In this project, the problem is modeled as a Hidden Markov Model (HMM) and the Viterbi algorithm is implemented to tag the words.

## Project Motivation

I implemented this program as a project for a computer science course at Dartmouth College. This project was very interesting because it illustrated the power and efficiency of dynamic programming and stochastic processes and natural language processing. 

In an attempt to improve the accuracy of tagging, I later implemented cross-validation and interpolated the original bigram model with a trigram model.

<p align="right">(<a href="#top">back to top</a>)</p>

## Background

This program uses a **hidden Markov Model** to model the words and their corresponding tags. In this case, the words are called observable states and the tags are called hidden states (this is further explained below).

### Markov Chains
Before diving into the definition of a hidden Markov Model, I will briefly provide the definition of a Markov chain. A Markov chain is a stochastic process in which a rule named the **Markov assumption** is obeyed. Specifically, consider a sequence of random variables $q_1, q_2, q_3, \ldots, q_i$, the Markov assumption states

$$
\mathbb{P}(q_t=a|q_{t_1}, q_{t_2}, \ldots, q_3, q_2, q_1) = \mathbb{P}(q_t=a|q_{t-1})
$$

In words, in a Markov process, the state $q_t$ at time $t$ may depend on the previous state $q_{t-1}$ but no states before that.

A Markov chain is a fundamental concept in probability and stochastic processes. It is used in numerous applications such as modeling chemical reactions, predicting population dynamics, and many other phenomena.

I recommend one to refer to the text *Essentials of Stochastic Processes* by Richard Durrett for more details.

### Hidden Markov Chains

Now, a hidden Markov chain also involves hidden (not observable) states. In our case, the words in a sentence are observable but the corresponding parts of the speech tags are hidden.

Suppose again that we have a set of states (these will be known as hidden states) $q_1, q_2, q_3, \ldots, q_N$ and we define some $N \times N$ transition matrix M such that the $(i,j)^\text{th}$ entry is the transition probability from state $i$ to state $j$.


$$
[M]_{ij} = \mathbb{P}(q_t=j|q_{t-1}=i) \quad 1 \le i,j \le N
$$

Moreover, we define the $\pi_i$ to be the the probability that the initial state is $i$ for $1 \le i \le N$.

Now, what is new here the a sequence of **observed states** $\mathcal{V}=\{ v_i | 1 \le i \le m \}$ and an **observation sequence** 


$$
\mathcal{O} = o_1, o_2, o_3, \ldots, o_T
$$

where $o_i$ is the observed state at time step $i$.

Finally, we define an **observation likelihood** $$b_i(o_t)$$ to be the probability of seeing the observable state $o_t$ given the hidden state $q_t=i$.

In addition to the Markov assumption, we also need another assumption about the observed states named the **output independence assumption** where

$$
\mathbb{P}(o_i|q_1, \ldots, q_i, \ldots q_T, o_1, o_2, \ldots, o_T) = \mathbb{P}(o_i|q_i)
$$

In words, the probability of the $i^\text{th}$ observed state may only depend on the $i^\text{th}$ hidden state and not on any other observed or hidden states.

To summarize, we now draw a correspondence between the terminology of a hidden Markov model and our part of speech tagging problem through a table

| Hidden Markov Model             | Part of Speech Tagging                                              |
| -----------                     | -----------                                                         |
| transition probability          | probability of a tag immediately following another tag in a sentence|
| observed states                 | words in a dictionary                                               |
| observation likelihood          | likelihood of a word given a part of speech                         |
|initial probability distribution | probability of the first part of speech in a sentence               |

### Viterbi Algorithm

The Viterbi algorithm is a dynamic programming approach to computing the maximum likelihood of hidden states given the a sequence of observed states.

The algorithm is a supervised learning algorithm in which the transition probabilities among tags, the observation likelihoods of words given a tag, and the initial probability distributions are computed using a large labeled data set (part of the Brown Corpus in our case).

Next, we wish to find a sequence of hidden states (part of speech tags) that maximizes the likelihood of the observed states (sentences). Suppose we observe a sentence (e.g. "Ocaml is an amazing functional language). Let's denote this sentence by $\mathcal{O}$. Then we wish to find a tag sequence (e.g. ""noun verb determiner adjective adjective noun") that maximizes the likelihood probability. To this end, see that by the laws of total probability and conditional probability

$$
\mathbb{P}(\mathcal{O}) = \sum_{Q} \mathbb{P}(\mathbb{\mathcal{O},Q})=\sum_{Q} \mathbb{P}(\mathcal{O}|Q)\mathbb{P}(Q) 
$$

Since there are $N^T$ possible states where $N$ is the number of possible tags and $T$ is the number of words in the sentence, we see that the computational complexity is exponential. However, the Viterbi algorithm uses dynamic programming and backtracking to solve this optimization problem in polynomial time.

This program implements the Viterbi algorithm to compute the maximum likelihoods and uses backtracking to recover the sequence of tags that gives the maximum likelihood.

The details of the Viterbi algorithm is brilliant and extremely well explained in the text *Speech and Language Processing* by Dan Jurafsky and James H. Martin. I followed the explanation in the text to implement my part of speech tagging algorithm.

## alternative implementations

### Trigram Model

In addition to a **bigram model** which strictly obeys the Markov property, I have also experimented with a trigram model in which each tag depends on the previous *two* tags in a sentence.

Suppose we observe a line of words $o_0, o_1, o_2, \ldots, o_n$ and some sequence of tags $q_0, q_1, q_2, \ldots, q_n$. Then in a trigram model, we have that

$$
    \mathbb{P}(q_{i+1} | q_{i}, q_{i-1}, \ldots, q_{0}) = \mathbb{P}(q_{i+1} | q_{i}, q_{i-1})
$$

It is apparent that a trigram model is computationally more expensive to train and test but may provide improve performance depending on the data set.

One issue that arises in moving to the trigram model is that it may be less robust than the bigram model. Because the state space increased quadratically, when testing, the instances that we run into a prior tag pair not in the training data set may drastically increase. As a result, the true tag path of a line may not be obtained if  we do not data on certain trigrams. One particular issue with the Brown Corpus is that permutations of punctuation may not be the training set. This would decrease performance for the trigram model.

To mitigate this problem, we use an interpolation technique in which the weighted average of the bigram and trigram models is used as the **transition score**.

To calculate the interpolated transition probability we have,

$$
    \text{transition score} = o_t \cdot \mathbb{P}(q_{i+1} | q_{i}, q_{i-1}) + o_b \cdot \mathbb{P}(q_{i+1} | q_{i})
$$

where $w_t + w_b = 1$.

To actually compute the weights, we use a deleted interpolation technique called **leave-one-out cross-validation**. This technique calculates the maximum likelihood of the weights by removing each trigram/bigram in an iterative manner.

Using the Brown Corpus, the weights computed are $w_b = 0.4423$ and $w_t = 0.5577$. The results are below

<div align="left">

```
Beginning test 1...
Training with the simple sentences
bigram weight: 0.576271186440678
trigram weight: 0.423728813559322
Testing on simple test sentences
The POS tagger identified 27 out of 37 tags correctly.
The accuracy is: 72.97297297297297%

Beginning test 2...
Training with the Brown corpus
bigram weight: 0.4423146592324493
trigram weight: 0.5576853407675507
Testing on Brown test sentences
The POS tagger identified 34086 out of 36394 tags correctly.
The accuracy is: 93.65829532340496%
```

</div>

It is surprising to note that the performance did not actually improve from  using purely the bigram model. There are multiple factors that resulted in this. First, although the trigram model is a stronger condition, it is actually less robust than the bigram model given small data set we have. we often do not have a trigram prior (especially if a state pair contain punctuation).

## Test Cases

Both the bigram and the interpolated trigram implementations can be tested with the standard Brown corpus, which is a very large collection of American English text (https://en.wikipedia.org/wiki/Brown_Corpus). The program computes the accuracy of the tagging is also interactive. It takes sentences as prompt and outputs the corresponding tags.

An example is shown below 

<div align="left">

```
Beginning test 1...
Training with the simple sentences
Testing on simple test sentences
The POS tagger identified 32 out of 37 tags correctly.
The accuracy is: 86.48648648648648%

Beginning test 2...
Training with the Brown corpus
Testing on Brown test sentences
The POS tagger identified 35109 out of 36394 tags correctly.
The accuracy is: 96.46919821948673%

Beginning console-based tagging...
Please enter a sentence to get tags (enter "q" to quit game)
> Ocaml is an amazing functional language, and everyone should learn it!                                
PRO V DET ADJ ADJ N CNJ PRO MOD V DET 

Please enter a sentence to get tags (enter "q" to quit game)
```

</div>


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Jiahui Zhang - jiahui.zhang.20@gmail.com

Nour Hayek - https://www.linkedin.com/in/nourhayek

Project Link: [https://github.com/Jiahui-Zhang-20/POStagging](https://github.com/Jiahui-Zhang-20/POStagging)

<p align="right">(<a href="#top">back to top</a>)</p>

## References

"Speech and Language Processing" by Dan Jurafsky and James H. Martin

https://stathwang.github.io/part-of-speech-tagging-with-trigram-hidden-markov-models-and-the-viterbi-algorithm.html

https://www.freecodecamp.org/news/a-deep-dive-into-part-of-speech-tagging-using-viterbi-algorithm-17c8de32e8bc/