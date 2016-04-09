import nltk, re, sys, itertools
from hmm import HMM

trainFile = open(sys.argv[1]).read()
readFile = open(sys.argv[2]).read()
outFile = open(sys.argv[3],'w')
pattern = re.compile(r'(.*?)\t(.*?)\n')
sent_pattern = re.compile(r'\n{2}')
sents = re.split(sent_pattern, trainFile)
sents.pop()
sent_tags = []
for sent in sents:
    sent_tags.append(re.findall(pattern, sent))
for i in range(len(sent_tags)):
    sent_tags[i].insert(0,('START','START'))
    sent_tags[i].append(('END','END'))
concat_sents = list(itertools.chain(*sent_tags))
# concat_sents = [(w.lower(),t) for (w,t) in concat_sents]

def viterbi(obs, hmm, outfile):
    V = [{}]
    # Initialize the start states
    for s in hmm.states:
        if obs[0] in hmm.observations:
            V[0][s] = hmm.transition_p('START',s)*hmm.emission_p(obs[0],s)
        else:
            V[0][s] = hmm.transition_p('START',s)*(1/1000)
    # Run viterbi while t > 0
    for t in range(1,len(obs)):
        # For each observation add a column
        V.append({})
        # Cycles through previous cells in V and takes the maximum probability of the combination of transition,
        # emission, and previous probability
        for s in hmm.states:
            # Deal with OOV items by replacing emission probability with 1/1000
            if obs[t] in hmm.observations:
                # V[t-1][y0] refers to the all the cells in the last column of the Viterbi matrix V
                prob = max(V[t - 1][y0]*hmm.transition_p(y0,s)*hmm.emission_p(obs[t],s) for y0 in hmm.states)
                V[t][s] = prob
            else:
                prob = max(V[t - 1][y0]*hmm.transition_p(y0,s)*(1/1000) for y0 in hmm.states)
                V[t][s] = prob
    opt = []
    # For each row in V
    for j in V:
        # For the tuple of key:value for each column in V
        for x, y in j.items():
            # If the cell x is the most probable append that state to opt
            if j[x] == max(j.values()):
                opt.append(x)
    # The highest probability value
    h = max(V[-1].values())
    for i in range(len(obs)):
        outfile.write(str(obs[i])+"\t"+str(opt[i])+"\n")
    # print('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s'%h)

hmm = HMM(concat_sents)
hmm.initEM()
hmm.initTM()
read_sents = re.split(r'\n{2}', readFile)
read_sents.pop()
obsSents = []
for sent in read_sents:
    obsSents.append(sent.split())
for sent in obsSents:
    viterbi(sent, hmm, outFile)
    outFile.write('\n')
