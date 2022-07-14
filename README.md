<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Jiahui-Zhang-20/POStagging">
    <img src="https://d33wubrfki0l68.cloudfront.net/d5cbc4b0e14c20f877366b69b9171649afe11fda/d96a8/assets/images/bigram-hmm/pos-title.jpg" alt="Logo" width="600" height="200">
  </a>

  <h1 align="center">Part of Speeh Tagging using the Hidden Markov Models</h1>

<!-- ABOUT THE PROJECT -->
## About The Project

This program labels the words of a sentence with the corresponding "parts of speech" or POS. According to the Merriam-Webster dictionary, part of speech is "a traditional class of words (such as adjectives, advers, nouns, and verbs) distinguished according to the kind of idea denoted and the function performed in a sentence.

For example, consider the sentence "Ocaml is an amazing functional language". The corresponding POS sequence would be "noun verb determiner adjective adjective noun".

Part of speech tagging is a fundamental concept in natural language processing. In this project, the problem is modeled as a Hidden Markov Model (HMM) and the Viterbi algorithm is implemented to tag the words.

### Project Motivation

I implemented this program as a project for a computer science course at Dartmouth College. This project was very interesting because it illustrated the power and efficiency of dynamic programming and stochastic processes and natural language processing. 

In an attempt to improve the accuracy of tagging, I later implemented cross-validation and interpolated the original bigram model with a trigram model.

<p align="right">(<a href="#top">back to top</a>)</p>

### Background

### Test Cases

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

https://web.stanford.edu/~jurafsky/slp3/A.pdf

https://stathwang.github.io/part-of-speech-tagging-with-trigram-hidden-markov-models-and-the-viterbi-algorithm.html

https://www.freecodecamp.org/news/a-deep-dive-into-part-of-speech-tagging-using-viterbi-algorithm-17c8de32e8bc/