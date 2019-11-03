#!/usr/python

from gensim.models import KeyedVectors
from gensim.models.keyedvectors import FastTextKeyedVectors
import re,sys,os,pickle,random,codecs,nltk
import numpy as np
import tensorflow as tf
from skip_thoughts import SkipThoughts 
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from sklearn.decomposition import TruncatedSVD
import tqdm
import unicodedata
from nltk.tokenize.regexp import RegexpTokenizer
import progressbar

import numpy as np



#####################################################################
#                                                                   #
#            Basic classes and var for data manipulation            #
#                                                                   #
#####################################################################

# unkown word label in the WE
UNK = u'unk'


# UD Portuguese Tokenizer
class PtgTokenizer(RegexpTokenizer):
    """
    Tokenize the given sentence in Portuguese.
    :param text: text to be tokenized, as a string
    """
    def __init__(self):
       tokenizer_regexp = r'''(?ux)
              # the order of the patterns is important!!
              # more structured patterns come first
              [a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+|    # emails
              (?:https?://)?\w{2,}(?:\.\w{2,})+(?:/\w+)*|                  # URLs
              (?:[\#@]\w+)|                     # Hashtags and twitter user names
              (?:[^\W\d_]\.)+|                  # one letter abbreviations, e.g. E.U.A.
              (?:[DSds][Rr][Aa]?)\.|            # common abbreviations such as dr., sr., sra., dra.
              (?:\B-)?\d+(?:[:.,]\d+)*(?:-?\w)*|
              # numbers in format 999.999.999,999, possibly followed by hyphen and alphanumerics
              # \B- avoids picks as F-14 as a negative number
              \.{3,}|                           # ellipsis or sequences of dots
              \w+|                              # alphanumerics
              -+|                               # any sequence of dashes
              \S                                # any non-space character
       '''
       RegexpTokenizer.__init__(self,tokenizer_regexp)

# Basic Tokenizer (used in (SOUZA and SANCHES, 2018)) 
class BasicTokenizer():
   def tokenize(self,s):
      return [re.sub("[.,!?;:()#%]","",word) for word in s.lower().split()]


def grava_dados(data,filename):
      outputFile = open(filename,'wb')
      pickle.dump(data,outputFile)
      outputFile.close()


#####################################################################
#            Functions for computing Representations                #
#                                                                   #
# 
#####################################################################


#get_embeddings:
#   params:
#       sentences: list of raw text sentences
#       tokenizer: a tokenizer object (any object that implements tokenize(self, sentence) and returns a list/iterable of tokens
#       model: a gensim word embedding model
#  output:
#        word_vectors: a np.array(#sentences, len(sentence), #WE_dimensions) of word vectors for each sentence
#        inds: a np.array(#sentences, len(sentence), #WE_dimensions) with the index of each word in the WE model vocabulary
def get_embeddings(sentences,tokenizer, model):
   word_vectors=[]
   inds = []
   vocab = list(model.vocab.keys())
   for sentence in sentences:
      words = tokenizer.tokenize(sentence)   
      indexes = []
      vectors = []
      for word in words:
         if word in model.vocab:
            vectors.append(model.get_vector(word))
            indexes.append(vocab.index(word))
         else:
            vectors.append(model.get_vector(UNK))
            indexes.append(vocab.index(UNK))
      word_vectors.append(np.array(vectors))
      inds.append(np.array(indexes))
   return np.array(word_vectors),np.array(inds)



#################
# name:    AVG
# description: averaged sum of words representations
#################

def avgVec(sentences,tokenizer, model):
   sentList,_ = get_embeddings(sentences,tokenizer, model)
   avgs = []
   for sent in sentList:
      if len(sent)>0:
         avgs.append(sum(sent)/len(sent))
   return np.array(avgs)

#################
# name: SIF
# description: removes principal components of the word vectors on IDF weighted average
# Ref: Arora, Sanjeev, Yingyu Liang, and Tengyu Ma. "A simple but tough-to-beat baseline for sentence embeddings." (2016).
#################
def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX

def get_weighted_average(We, x, w):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
#print n_samples
    emb = np.zeros((n_samples, We.shape[1]))
    for i in range(n_samples):
        for j in range(len(x[i])):
         emb[i,:] += We[x[i][j],:]*w[i][j]
#        emb[i,:] = w[i,:].dot(We[x[i,:],:]) / np.count_nonzero(w[i,:])
    return emb



def SIF_embedding(We, x, w):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    emb = get_weighted_average(We, x, w)
    emb = remove_pc(emb)
    return emb

def get_sif(pair, tokenizer, word2vec_model,IDF):
    We = word2vec_model.vectors
    vocab = list(word2vec_model.vocab)
    _,x = get_embeddings(pair,tokenizer, word2vec_model)
    w = np.array([np.array([IDF[vocab[i]] for i in wordsi]) for wordsi in x])
    n=[]
    for i in w:
       if sum(i) != 0:
          n.append(1.0/sum(i))
       else:
          n.append(0)
    for i in range(len(w)):
       w[i] = w[i]*n[i]
    return SIF_embedding(We,x,w) 
    




#################
# name: agg
# description: computes IDF-weighted sum/aggregation of word vectors
# Ref: Mihalcea, Rada, Courtney Corley, and Carlo Strapparava. "Corpus-based and knowledge-based measures of text semantic similarity." Aaai. Vol. 6. No. 2006. 2006.
#################

def get_aggregated(pair, tokenizer, word2vec_model, IDF):
    We = word2vec_model.vectors
    _,x = get_embeddings(pair,tokenizer, word2vec_model)
    vocab = list(word2vec_model.vocab)
    w = np.array([np.array([IDF[vocab[i]] for i in wordsi]) for wordsi in x])
    n=[]
    for i in w:
       if sum(i) != 0:
          n.append(1.0/sum(i))
       else:
          n.append(0)
    for i in range(len(w)):
       w[i] = w[i]*n[i]
    return get_weighted_average(We,np.array(x),w)
   


#####################################################################
#                                                                   #
#            Functions for computing Similarities                   #
#                                                                   #
#####################################################################

def simMax(s1,s2,tokenizer,model):
   t1 = [t for t in tokenizer.tokenize(s1)]
   t2 = [t for t in tokenizer.tokenize(s2)]
       
   similarity = -1
   sim1 = 0.0
   sim2 = 0.0
   for token1 in t1:
      for token2 in t2:
         try:
            sim1 = max(sim1, model.similarity(token1, token2))
         except:
            continue
#   similarity = 0.5*(sim1+sim2)
   return sim1


def simMean(s1,s2,tokenizer,model):
   try:
      t1 = [t for t in tokenizer.tokenize(s1)]
      t2 = [t for t in tokenizer.tokenize(s2)]
      return model.n_similarity(t1,t2)
   except:
      t1 = [t for t in tokenizer.tokenize(s1) if t in model.vocab]
      t2 = [t for t in tokenizer.tokenize(s2) if t in model.vocab]
      return model.n_similarity(t1,t2)


def distWordMover(s1,s2,tokenizer,model):
   try:
      t1 = [t for t in tokenizer.tokenize(s1)]
      t2 = [t for t in tokenizer.tokenize(s2)]
      return model.wmdistance(t1,t2)
   except:
      t1 = [t for t in tokenizer.tokenize(s1) if t in model.vocab]
      t2 = [t for t in tokenizer.tokenize(s2) if t in model.vocab]
      return model.wmdistance(t1,t2)

def cos_sim(a, b):
	"""Takes 2 vectors a, b and returns the cosine similarity according 
	to the definition of the dot product
	"""
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)


#####################################################################
#                                                                   #
#                             Main code                             #
#                                                                   #
#####################################################################

def main():


#Load parameters
   if len(sys.argv)>1:
      path_to_config = sys.argv[1]
      config = dict([line.split(":") for line in open(path_to_config).read().split('\n') if line!='' and not re.match("#.*",line)])
      for parameter in config:
         if config[parameter]=="True":
            config[parameter]=True
         elif config[parameter]=="False":
            config[parameter]=False
   else:
      config={'path_to_word_embedding_model':'/home/marlo/pesquisa/modelos/word embeddings/fast_s300'}
      config.update({'collection_of_we':False})
      config.update({'path_to_skip_thought_model':'/home/marlo/pesquisa/modelos/skipthought/102018/nilc300'})
      config.update({'path_to_idf_dict':'/home/marlo/pesquisa/exp/parafrase/data/idf/idf.csv'})
      config.update({'path_to_sentences':'dev.full.txt', 'output_file':'dev_ft'})
      config.update({'output_path':'data/vectors'})

#Load data
   print("Loading data...")
   xmlText = codecs.open(config['path_to_sentences'],encoding='utf-8').read()
   pairsText = re.findall("(?s)<pair entailment=\"([^\"]+)\".+?similarity=\"([^\"]+)\">(.+?)</pair>",xmlText)
   pairs = [(re.findall("<t>(.+?)</t>",t[2])[0].lower(),re.findall("<h>(.+?)</h>",t[2])[0].lower(),t[1],1) for t in pairsText if t[0]=="Paraphrase"]
   pairs.extend([(re.findall("<t>(.+?)</t>",t[2])[0].lower(),re.findall("<h>(.+?)</h>",t[2])[0].lower(),t[1],0) for t in pairsText if t[0]!="Paraphrase"])
   sentences=[pair[0:2] for pair in pairs]
   sim = [pair[2] for pair in pairs]
   classes = [pair[3] for pair in pairs]
   tokenizer = BasicTokenizer()
   print("Loading Word Embedding Model...")
   if config['fasttext']:
      word2vec_model = FastTextKeyedVectors.load(config['path_to_word_embedding_model'], mmap='r')
   else:
      word2vec_model = KeyedVectors.load(config['path_to_word_embedding_model'], mmap='r')
   print("Loading IDF dictionary...")
   IDF = pickle.load(open(config['path_to_idf_dict'],"rb"))
   
   graph = tf.Graph()

#  Elmo only works on Python3 with allennlp
   if config['elmo']:
       from allennlp.commands.elmo import ElmoEmbedder
       print("Loading ELMO Model...")      
       elmo = ElmoEmbedder(
          options_file=os.path.join(config['path_to_elmo_model'],"elmo_pt_options.json"),
          weight_file=os.path.join(config['path_to_elmo_model'],"elmo_pt_weights.hdf5"),
          cuda_device=0
       )
   if config['st']:
      with graph.as_default():
        # Refer to the constructor docstring for more information on the arguments.
        print("Loading Skip-Thought Model...")
        model = SkipThoughts(word2vec_model)

   with tf.compat.v1.Session(graph=graph):
      # Restore the model only once.
      # Here, `save_dir` is the directory where the .ckpt files live. Typically
      # this would be "output/mymodel" where --model_name=mymodel in train.py. 
      if config['st']:
         model.restore(config['path_to_skip_thought_model'])
      # Run the model like this as many times as desired.   
      i=0
      X={'sim':[],'avg':[],'st':[],'sif':[],'agg':[],'elmo':[]}
      Xdif = {'avg':[],'st':[],'sif':[],'agg':[],'elmo':[]}
      Xdir = {'avg':[],'st':[],'sif':[],'agg':[],'elmo':[]}


      y=[]
      y2=[]
      print("Computing Representations...")
      bar = progressbar.ProgressBar(maxval=len(sentences), \
          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
      bar.start()  
      #Computes sentence representations
      for pair in sentences:
         bar.update(i)
         #saves (s1,s2,class) and (s2,s1,class)
         y.append(int(classes[i]))
         y.append(int(classes[i]))
         y2.append(float(sim[i]))
         y2.append(float(sim[i]))  

         # Computes Average vector
         if config['avg']:
            avg = avgVec(pair,tokenizer,word2vec_model)
            dotuv = (avg[0]*avg[1])
            minuv = avg[0]-avg[1]

            WEsimMean = float(simMean(pair[0],pair[1],tokenizer,word2vec_model))
            X['avg'].append(np.concatenate([avg[0],avg[1],dotuv,minuv,[np.linalg.norm(minuv),WEsimMean]]))
            Xdif['avg'].append(np.concatenate([dotuv,minuv,[np.linalg.norm(dotuv),np.linalg.norm(minuv),WEsimMean]]))
            Xdir['avg'].append(np.concatenate([avg[0],avg[1],[WEsimMean]]))
            X['avg'].append(np.concatenate([avg[1],avg[0],dotuv,-minuv,[np.linalg.norm(minuv),WEsimMean]]))
            Xdif['avg'].append(np.concatenate([dotuv,-minuv,[np.linalg.norm(dotuv),np.linalg.norm(minuv),WEsimMean]]))
            Xdir['avg'].append(np.concatenate([avg[1],avg[0],[WEsimMean]]))


         # Computes IDF-weighted aggregated vector
         if config['agg']:
            agg = get_aggregated(pair, tokenizer, word2vec_model,IDF)
            dotuv = (agg[0]*agg[1])
            minuv = agg[0]-agg[1]
            WEsimAgg = float(cosine_similarity(agg[0].reshape(1,-1),agg[1].reshape(1,-1)))
            X['agg'].append(np.concatenate([agg[0],agg[1],dotuv,minuv,[np.linalg.norm(minuv),WEsimAgg]]))
            Xdif['agg'].append(np.concatenate([dotuv,minuv,[np.linalg.norm(dotuv),np.linalg.norm(minuv),WEsimAgg]]))
            Xdir['agg'].append(np.concatenate([agg[0],agg[1],[WEsimAgg]]))
            X['agg'].append(np.concatenate([agg[1],agg[0],dotuv,-minuv,[np.linalg.norm(minuv),WEsimAgg]]))
            Xdif['agg'].append(np.concatenate([dotuv,-minuv,[np.linalg.norm(dotuv),np.linalg.norm(minuv),WEsimAgg]]))
            Xdir['agg'].append(np.concatenate([agg[1],agg[0],[WEsimAgg]]))


         # Computes SIF representation
         if config['sif']:
            sif = get_sif(pair, tokenizer, word2vec_model,IDF)
            dotuv = (sif[0]*sif[1])
            minuv = sif[0]-sif[1]
            WEsimSIF = float(cosine_similarity(sif[0].reshape(1,-1),sif[1].reshape(1,-1)))
            X['sif'].append(np.concatenate([sif[0],sif[1],dotuv,minuv,[np.linalg.norm(minuv),WEsimSIF]]))
            Xdif['sif'].append(np.concatenate([dotuv,minuv,[np.linalg.norm(dotuv),np.linalg.norm(minuv),WEsimSIF]]))
            Xdir['sif'].append(np.concatenate([sif[0],sif[1],[WEsimSIF]]))
            X['sif'].append(np.concatenate([sif[1],sif[0],dotuv,-minuv,[np.linalg.norm(minuv),WEsimSIF]]))
            Xdif['sif'].append(np.concatenate([dotuv,-minuv,[np.linalg.norm(dotuv),np.linalg.norm(minuv),WEsimSIF]]))
            Xdir['sif'].append(np.concatenate([sif[1],sif[0],[WEsimSIF]]))


         # Computes Skip-Thought representation
         if config['st']:
            vecSent= model.encode(pair)
            dotuv = vecSent[0]*vecSent[1]
            dotuv = dotuv / np.linalg.norm(dotuv)
            minuv = vecSent[0]-vecSent[1]
            STSim = cosine_similarity([vecSent[0]],[vecSent[1]])[0][0]
            X['st'].append(np.concatenate([vecSent[0],vecSent[1],dotuv,minuv,[np.linalg.norm(dotuv),np.linalg.norm(minuv),STSim]])) 
            Xdif['st'].append(np.concatenate([dotuv,minuv,[np.linalg.norm(dotuv),np.linalg.norm(minuv),STSim]]))
            Xdir['st'].append(np.concatenate([vecSent[0],vecSent[1],[STSim]]))
            X['st'].append(np.concatenate([vecSent[1],vecSent[0],dotuv,-minuv,[np.linalg.norm(dotuv),np.linalg.norm(minuv),STSim]])) 
            Xdif['st'].append(np.concatenate([dotuv,-minuv,[np.linalg.norm(dotuv),np.linalg.norm(minuv),STSim]]))
            Xdir['st'].append(np.concatenate([vecSent[1],vecSent[0],[STSim]]))


         # Computes average of Elmo 3rd layer
         if config['elmo']:
            embeddings= [elmo.embed_sentence(pair[0])[2], elmo.embed_sentence(pair[1])[2]]
            vecSent = [sum(embeddings[0])/len(embeddings[0]),sum(embeddings[1])/len([1])]
            dotuv = vecSent[0]*vecSent[1]
            dotuv = dotuv / np.linalg.norm(dotuv)
            minuv = vecSent[0]-vecSent[1]
            ElmoSim = cosine_similarity([vecSent[0]],[vecSent[1]])[0][0]
            X['elmo'].append(np.concatenate([vecSent[0],vecSent[1],dotuv,minuv,[np.linalg.norm(dotuv),np.linalg.norm(minuv),ElmoSim]])) 
            Xdif['elmo'].append(np.concatenate([dotuv,minuv,[np.linalg.norm(dotuv),np.linalg.norm(minuv),ElmoSim]]))
            Xdir['elmo'].append(np.concatenate([vecSent[0],vecSent[1],[ElmoSim]]))
            X['elmo'].append(np.concatenate([vecSent[1],vecSent[0],dotuv,-minuv,[np.linalg.norm(dotuv),np.linalg.norm(minuv),ElmoSim]])) 
            Xdif['elmo'].append(np.concatenate([dotuv,-minuv,[np.linalg.norm(dotuv),np.linalg.norm(minuv),ElmoSim]]))
            Xdir['elmo'].append(np.concatenate([vecSent[1],vecSent[0],[ElmoSim]]))


         # Computes Similarities
         if config['sim']:
            rep = []
            rep.append(float(simMax(pair[0],pair[1],tokenizer,word2vec_model)))
            rep.append(float(simMean(pair[0],pair[1],tokenizer,word2vec_model)))
            rep.append(float(distWordMover(pair[0],pair[1],tokenizer,word2vec_model)))
            if config['agg']:
               rep.append(WEsimAgg)
            if config['sif']:
               rep.append(WEsimSIF)
            if config['st']:
               rep.append(STSim)
            if config['elmo']:
               rep.append(ElmoSim)
            
            X['sim'].append(rep)
         i=i+1

      bar.finish()   
      print("Saving data...")

      # Persists Paraphrase truth file
      grava_dados(y,os.path.join(config['output_path'],config['output_file']+".y"))      
      # Persists Similarity truth file
      grava_dados(y2,os.path.join(config['output_path'],config['output_file']+".y2"))      
      
      # Persists computed representations
      for rep in ['avg','agg','st','sif','elmo']:
         if config[rep]:
            for data_form in ['X','Xdif','Xdir']: 
               if config[data_form]:
                  grava_dados(X[rep],os.path.join(config['output_path'],config['output_file']+".{}.{}".format(rep,data_form)))
      if config['sim']:
         grava_dados(X['sim'],os.path.join("data/vectors",config['output_file']+".sim.X"))

 



      
if __name__ == "__main__":
    main()    


