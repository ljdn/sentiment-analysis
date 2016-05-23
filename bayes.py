# Name: Lejia Duan (ldu917)
# Date: 5/19
# Description:
#
#

import math, os, pickle, re

class Bayes_Classifier:

    def __init__(self):
        """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
        cache of a trained classifier has been stored, it loads this cache.  Otherwise,
        the system will proceed through training.  After running this method, the classifier
        is ready to classify input text."""

        # initialize dictionaries
        self.positive_dict = dict()
        self.negative_dict = dict()

        # check if saved dictionaries already exist
        if os.path.isfile("pickled_pos.dat") and os.path.isfile("pickled_neg.dat"):
            self.positive_dict = self.load("pickled_pos.dat")
            self.negative_dict = self.load("pickled_neg.dat")
        else:
            self.train()

    def train(self):
        """Trains the Naive Bayes Sentiment Classifier."""

        print "Training..."
        # Get names of all files in directory
        lFileList = []
        for fFileObj in os.walk("movie_reviews/movies_reviews/"):
            lFileList = fFileObj[2]
            break

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
                self.updateDict(training_tokens, self.positive_dict)
            else:
                self.updateDict(training_tokens, self.negative_dict)

        self.save(self.positive_dict, "pickled_pos.dat")
        self.save(self.negative_dict, "pickled_neg.dat")


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

        # determine positive/negative/neutral by comparing probabilities
        if prob_pos > prob_neg:
            result = "positive"
        elif prob_neg < prob_pos:
            result = "negative"
        else:
            result = "neutral"

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

    def updateDict(self, add_words, orig_dict):
        """Updates frequencies in orig_dict with add_words """

        for word in add_words:
            word = word.lower().strip()
            if orig_dict.has_key(word):
                orig_dict[word] += 1
            else:
                orig_dict[word] = 1

        return orig_dict
