# Name: Lejia Duan (ldu917)
# Date: 5/19
# Description:
#
#

import math, os, pickle, re

class Bayes_Classifier:

    def __init__(self, eval = False, training_data = []):
        """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
        cache of a trained classifier has been stored, it loads this cache.  Otherwise,
        the system will proceed through training.  After running this method, the classifier
        is ready to classify input text."""

        # initialize dictionaries
        self.positive_dict = dict()
        self.negative_dict = dict()
        self.two_gram_n = dict()
        self.two_gram_p = dict()

        # check if saved dictionaries already exist
        if os.path.isfile("pickled_pos.dat") and os.path.isfile("pickled_neg.dat"):
            self.positive_dict = self.load("pickled_pos.dat")
            self.negative_dict = self.load("pickled_neg.dat")
            self.two_gram_n = self.load("two_neg.dat")
            self.two_gram_p = self.load("two_pos.dat")
        else:
            self.train(eval, training_data)

    def train(self, eval = False, training_data = []):
        """Trains the Naive Bayes Sentiment Classifier."""

        # Get names of all files in directory
        lFileList = []
        for fFileObj in os.walk("movie_reviews/movies_reviews/"):
            lFileList = fFileObj[2]
            break

        if eval:
            lFileList = training_data

        # parse file name, determine if positive (5 stars) or negative (1 star)
        for training_file in lFileList:
            sentiment = "---"
            if training_file.startswith("movies-1"):
                sentiment = "neg"
            elif training_file.startswith("movies-5"):
                sentiment = "pos"

            # update word frequencies
            if sentiment == "---":
                continue

            training_text = self.loadFile("movie_reviews/movies_reviews/" + str(training_file))
            training_tokens = self.tokenize(training_text)

            if sentiment == "pos":
                self.updateDict(training_tokens, self.positive_dict, 1)
                self.updateDict(training_tokens, self.two_gram_p, 2)
            else:
                self.updateDict(training_tokens, self.negative_dict, 1)
                self.updateDict(training_tokens, self.two_gram_n, 2)

        self.save(self.positive_dict, "pickled_pos.dat")
        self.save(self.negative_dict, "pickled_neg.dat")
        self.save(self.two_gram_p, "two_pos.dat")
        self.save(self.two_gram_n, "two_neg.dat")


    def classify(self, sText):
        """Given a target string sText, this function returns the most likely document
        class to which the target string belongs (i.e., positive, negative or neutral).
        """

        target_tokens = self.tokenize(sText)

        # initialize probabilities
        prob_pos = 0.0
        prob_neg = 0.0

        # calculate probabilities for each word
        for token in target_tokens:
            word = token.lower().strip()
            # add one smoothing
            freq_pos = 1
            freq_neg = 1

            if self.positive_dict.has_key(word):
                freq_pos += float(self.positive_dict[word])
            if self.negative_dict.has_key(word):
                freq_neg += float(self.negative_dict[word])

            prob_pos += math.log(freq_pos/float(sum(self.positive_dict.values())))
            prob_neg += math.log(freq_neg/float(sum(self.negative_dict.values())))

        # calculate probabilities of bigrams
        for i in range(len(target_tokens)-2):
            phrase = ''.join(target_tokens[i:i+2]).lower().strip()
            freq_pos = 1
            freq_neg = 1
            if self.two_gram_p.has_key(phrase):
                freq_pos += float(self.two_gram_p[phrase])
            if self.two_gram_n.has_key(phrase):
                freq_neg += float(self.two_gram_n[phrase])

            prob_pos += math.log(freq_pos/float(sum(self.two_gram_p.values())))
            prob_neg += math.log(freq_neg/float(sum(self.two_gram_n.values())))

        # determine positive/negative/neutral by comparing probabilities
        if prob_pos > prob_neg:
            result = "positive"
        else:
            result = "negative"
        # else:
        #     result = "neutral"

        return result

    def loadFile(self, sFilename):
        """Given a file name, return the contents of the file as a string."""

        f = open(sFilename, "r")
        sTxt = f.read()
        f.close()
        return sTxt

    def save(self, dObj, sFilename):
        """Given an object and a file name, write the object to the file using pickle."""

        f = open(sFilename, "w")
        p = pickle.Pickler(f)
        p.dump(dObj)
        f.close()

    def load(self, sFilename):
        """Given a file name, load and return the object stored in the file."""

        f = open(sFilename, "r")
        u = pickle.Unpickler(f)
        dObj = u.load()
        f.close()
        return dObj

    def tokenize(self, sText):
        """Given a string of text sText, returns a list of the individual tokens that
        occur in that string (in order)."""

        lTokens = []
        sToken = ""
        for c in sText:
            if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_" or c == "-":
                sToken += c
            else:
                if sToken != "":
                    lTokens.append(sToken)
                    sToken = ""
                if c.strip() != "":
                    lTokens.append(str(c.strip()))

        if sToken != "":
            lTokens.append(sToken)

        return lTokens

    def updateDict(self, add_words, orig_dict, n):
        """Updates frequencies in orig_dict with add_words; n-gram """

        for i in range(len(add_words)-n):
            phrase = ''.join(add_words[i:i+n]).lower().strip()
            if orig_dict.has_key(phrase):
                orig_dict[phrase] += 1
            else:
                orig_dict[phrase] = 1

        return orig_dict

def evaluate_bae(n):
    """ Calculates precision, recall, and f-measure for classifier; n-fold validation """

    if os.path.isfile("pickled_pos.dat") and os.path.isfile("pickled_neg.dat"):
        os.remove("pickled_neg.dat")
        os.remove("pickled_pos.dat")
        os.remove("two_pos.dat")
        os.remove("two_neg.dat")

    # separate into positive and negative
    lFileList = []
    for fFileObj in os.walk("movie_reviews/movies_reviews/"):
        lFileList = fFileObj[2]
        break

    negatives = []
    positives = []

    for file in lFileList:
        if file.startswith("movies-1"):
            negatives.append(file)
        elif file.startswith("movies-5"):
            positives.append(file)

    len_slice = min(len(negatives), len(positives)) * 1/n

    for i in range(n):
        p_actual = 0
        p_false = 0
        n_actual = 0
        n_false = 0
        print "slice: " + str(i)
        # train
        training_data = negatives[i*(len(negatives)/n):i*(len(negatives)/n)+len_slice] + positives[i*len(positives)/n:i*len(positives)/n+len_slice]
        bc = Bayes_Classifier(True, training_data)
        # test
        testing_data = [x for x in lFileList if x not in training_data]
        for review in testing_data:
            tText = bc.loadFile("movie_reviews/movies_reviews/" + str(review))
            result = bc.classify(tText)
            if result == "positive":
                if review.startswith("movies-5"):
                    p_actual += 1
                elif review.startswith("movies-1"):
                    p_false += 1
            else:
                if review.startswith("moview-1"):
                    n_actual += 1
                else:
                    n_false += 1
        p_recall = float(p_actual) / len([x for x in testing_data and x in positives])
        n_recall = float(n_actual) / len([x for x in testing_data and x in negatives])
        p_precision = float(p_actual) / (p_actual + p_false)
        n_precision = float(n_actual) / (n_actual + n_false)
        p_fm = 2*(p_recall * p_precision) / (p_recall + p_precision)
        n_fm = 2*(n_recall * n_precision) / (n_recall + n_precision)

        print "recall (pos / neg): " + str(p_recall) + " / " + str(n_recall)
        print "precision (pos / neg): " + str(p_precision) +  " / " + str(n_precision)
        print "f-measure (pos / neg): " + str(p_fm) + " / " + str(n_fm)

        # remove saved dictionaries
        os.remove("pickled_neg.dat")
        os.remove("pickled_neg.dat")
        os.remove("two_pos.dat")
        os.remove("two_neg.dat")
