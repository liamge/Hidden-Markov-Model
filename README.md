# Hidden-Markov-Model
An instance of a Hidden Markov Model object made from scratch

Utilizes bigrams and a simplistic out of vocabulary solution of having that probability be 1/10000
Can improve accuracy by transitioning to a trigram model and having a more sophisticated OOV solution (i.e. morphological)

Included in this model is an instance of the Viterbi algorithm used to decode a sequence
of likely states given a string of observations

main.py is an example of an implementation of the HMM trained on a corpus of the format:
WORD\tPOS\n
It then writes to the outfile the most likely states for each sequece of observations (in this case sentences)

Command line access looks like: python3 main.py trainfile.txt testfile.txt output.txt

When trained on the WSJ corpus it achieved an accuracy of 92.85%
