import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

if __name__ == '__main__':
    # 2)
    # Count the number of instances (make into method)
    bCount = len(glob.glob1("BBC\\business", "*.txt"))
    eCount = len(glob.glob1("BBC\\entertainment", "*.txt"))
    pCount = len(glob.glob1("BBC\\politics", "*.txt"))
    sCount = len(glob.glob1("BBC\\sport", "*.txt"))
    tCount = len(glob.glob1("BBC\\tech", "*.txt"))

    # Arrange positions and data
    posi = [1, 2, 3, 4, 5]
    dist = [bCount, eCount, pCount, sCount, tCount]

    # Create the plot
    plt.bar(posi, dist)

    # Fix labels
    labels = ['Business', 'Entertainment', 'Politics', 'Sports', 'Technology']
    plt.xticks(posi, labels)

    # Save plot
    plt.savefig('BBC-distribution.pdf', dpi=100)

    # 3)
    dataset = load_files("BBC", encoding='latin-1')

    # 4)
    vect = CountVectorizer()
    dataset_counts = vect.fit_transform(dataset.data)

    # Display Term Doc and make a pandas dataframe
    term_doc = pd.DataFrame(dataset_counts.todense())
    term_doc.columns = vect.get_feature_names_out()
    term_document_matrix = term_doc.T
    print(term_document_matrix)

    # 5)
    train_x, test_x, train_y, test_y = \
        train_test_split(dataset_counts, dataset.target, test_size=0.2, random_state=None)
    print(dataset.target)

    # 6)
    # Use frequencies rather than counts to train NB
    tfidf_transformer = TfidfTransformer()
    train_x_tfidf = tfidf_transformer.fit_transform(train_x)
    clf = MultinomialNB().fit(train_x_tfidf, train_y)

    # Predict using frequencies of test set
    test_x_tfidf = tfidf_transformer.transform(test_x)
    predicted = clf.predict(test_x_tfidf)

    # Comparison
    #print("\nComparison between predicted and true")
    #print(np.mean(predicted == test_y))

    # 7)
    with open('bbc-performance.txt', 'a') as f:
        # a)
        f.write('a)\n***********************************\nMultinomialNB default values, try 1\n\n')

        # b)
        f.write('b) Confusion Matrix\n')
        np.savetxt(f, confusion_matrix(test_y, predicted), fmt='%d')
        f.write('\n')

        # c) & d)
        f.write('c) precision, recall, and F1 measure\n')
        f.write('d) accuracy, macro average F1, and weight average F1 of model\n')
        f.write(classification_report(test_y, predicted,
                                      target_names=["Business", "Entertainment", "Politics", "Sports", "Tech"]))
        f.write('\n')

        # e)
        f.write('e) prior probability of each class\n')
        np.savetxt(f, clf.class_log_prior_, fmt='%e')
        f.write('\n')

        # f)
        f.write('f) size of the vocabulary\n')
        vocab = len(term_document_matrix[(term_document_matrix.select_dtypes(include=["number"]) != 0).any(1)].index)
        f.write(str(vocab))
        f.write('\n\n')

        # g)
        f.write('g) the number of word tokens in each class\n')
        f.write('Business: ')
        f.write(str(term_document_matrix.T.iloc[0:(bCount - 1)].sum().sum()))
        f.write('\nEntertainment: ')
        f.write(str(term_document_matrix.T.iloc[bCount:bCount + eCount - 1].sum().sum()))
        f.write('\nPolitics: ')
        f.write(str(term_document_matrix.T.iloc[bCount + eCount:bCount + eCount + pCount - 1].sum().sum()))
        f.write('\nSports: ')
        f.write(str(term_document_matrix.T.iloc[
                    bCount + eCount + pCount:bCount + eCount + pCount + sCount - 1].sum().sum()))
        f.write('\nTechnology: ')
        f.write(str(term_document_matrix.T.iloc[
              bCount + eCount + pCount + sCount:bCount + eCount + pCount + sCount + tCount - 1].sum().sum()))

        f.write('\n\n')

        # h)
        f.write('h) the number of words in the entire corpus\n')
        f.write(str((term_document_matrix.sum()).sum()))
        f.write('\n\n')

        # i)

        f.write('i) the number and percentage of words with a frequency of zero in each class\n')

        # Business
        f.write('Business: ')
        BFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[:, 0:bCount]).iloc[i, :]).sum() == 0:
                BFreq = BFreq + 1

        f.write(str(BFreq))
        f.write(', ')
        f.write(str(BFreq / vocab))

        # Entertainment
        f.write('\nEntertainment: ')
        EFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[:, bCount:bCount + eCount]).iloc[i, :]).sum() == 0:
                EFreq = EFreq + 1

        f.write(str(EFreq))
        f.write(', ')
        f.write(str(EFreq / vocab))

        # Politics
        f.write('\nPolitics: ')
        PFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[:, bCount + eCount:bCount + eCount + pCount]).iloc[i, :]).sum() == 0:
                PFreq = PFreq + 1

        f.write(str(PFreq))
        f.write(', ')
        f.write(str(PFreq / vocab))

        # Sports
        f.write('\nSports: ')
        SFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[
                 :, bCount + eCount + pCount:bCount + eCount + pCount + sCount]).iloc[i, :]).sum() \
                    == 0:
                SFreq = SFreq + 1

        f.write(str(SFreq))
        f.write(', ')
        f.write(str(SFreq / vocab))

        # Technology
        f.write('\nTechnology: ')
        TFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[
                 :, bCount + eCount + pCount + sCount:bCount + eCount + pCount + sCount + tCount]).iloc[i, :]).sum() \
                    == 0:
                TFreq = TFreq + 1

        f.write(str(TFreq))
        f.write(', ')
        f.write(str(TFreq / vocab))
        f.write('\n\n')

        # j)
        f.write('j) the number and percentage of words with a frequency of one in the entire corpus\n')
        CFreq = 0
        for i in range(vocab):
            if (term_document_matrix.iloc[i, :]).sum() == 1:
                CFreq = CFreq + 1

        f.write(str(CFreq))
        f.write(', ')
        f.write(str(CFreq / vocab))
        f.write('\n\n')

        # k)
        f.write('k) your two favorite words (that are present in the vocabulary) and their log-prob\n')
        f.write('The log-probability of the word \'the\' is: ')
        #print(term_document_matrix.loc['the'].sum())
        f.write(str(math.log(term_document_matrix.loc['the'].sum() / (term_document_matrix.sum()).sum(), 10)))
        f.write('\nThe log-probability of the word \'in\' is: ')
        #print(term_document_matrix.loc['in'].sum())
        f.write(str(math.log(term_document_matrix.loc['in'].sum() / (term_document_matrix.sum()).sum(), 10)))
        f.write("\n\n")

    # 8) ********************************************************************************************************

        # Use frequencies rather than counts to train NB
        tfidf_transformer = TfidfTransformer()
        train_x_tfidf = tfidf_transformer.fit_transform(train_x)
        clf = MultinomialNB().fit(train_x_tfidf, train_y)

        # Predict using frequencies of test set
        test_x_tfidf = tfidf_transformer.transform(test_x)
        predicted = clf.predict(test_x_tfidf)

        # Comparison
        #print("\nComparison between predicted and true")
        #print(np.mean(predicted == test_y))

        # a)
        f.write('a)\n***********************************\nMultinomialNB default values, try 2\n\n')

        # b)
        f.write('b) Confusion Matrix\n')
        np.savetxt(f, confusion_matrix(test_y, predicted), fmt='%d')
        f.write('\n')

        # c) & d)
        f.write('c) precision, recall, and F1 measure\n')
        f.write('d) accuracy, macro average F1, and weight average F1 of model\n')
        f.write(classification_report(test_y, predicted,
                                        target_names=["Business", "Entertainment", "Politics", "Sports", "Tech"]))
        f.write('\n')

        # e)
        f.write('e) prior probability of each class\n')
        np.savetxt(f, clf.class_log_prior_, fmt='%e')
        f.write('\n')

        # f)
        f.write('f) size of the vocabulary\n')
        vocab = len(
        term_document_matrix[(term_document_matrix.select_dtypes(include=["number"]) != 0).any(1)].index)
        f.write(str(vocab))
        f.write('\n\n')

        # g)
        f.write('g) the number of word tokens in each class\n')
        f.write('Business: ')
        f.write(str(term_document_matrix.T.iloc[0:(bCount - 1)].sum().sum()))
        f.write('\nEntertainment: ')
        f.write(str(term_document_matrix.T.iloc[bCount:bCount + eCount - 1].sum().sum()))
        f.write('\nPolitics: ')
        f.write(str(term_document_matrix.T.iloc[bCount + eCount:bCount + eCount + pCount - 1].sum().sum()))
        f.write('\nSports: ')
        f.write(str(term_document_matrix.T.iloc[
                    bCount + eCount + pCount:bCount + eCount + pCount + sCount - 1].sum().sum()))
        f.write('\nTechnology: ')
        f.write(str(term_document_matrix.T.iloc[
                    bCount + eCount + pCount + sCount:bCount + eCount + pCount + sCount + tCount - 1].sum().sum()))

        f.write('\n\n')

        # h)
        f.write('h) the number of words in the entire corpus\n')
        f.write(str((term_document_matrix.sum()).sum()))
        f.write('\n\n')

        # i)

        f.write('i) the number and percentage of words with a frequency of zero in each class\n')

        # Business
        f.write('Business: ')
        BFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[:, 0:bCount]).iloc[i, :]).sum() == 0:
                BFreq = BFreq + 1

        f.write(str(BFreq))
        f.write(', ')
        f.write(str(BFreq / vocab))

        # Entertainment
        f.write('\nEntertainment: ')
        EFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[:, bCount:bCount + eCount]).iloc[i, :]).sum() == 0:
                EFreq = EFreq + 1

        f.write(str(EFreq))
        f.write(', ')
        f.write(str(EFreq / vocab))

        # Politics
        f.write('\nPolitics: ')
        PFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[:, bCount + eCount:bCount + eCount + pCount]).iloc[i, :]).sum() == 0:
                PFreq = PFreq + 1

        f.write(str(PFreq))
        f.write(', ')
        f.write(str(PFreq / vocab))

        # Sports
        f.write('\nSports: ')
        SFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[
                    :, bCount + eCount + pCount:bCount + eCount + pCount + sCount]).iloc[i, :]).sum() \
                        == 0:
                SFreq = SFreq + 1

        f.write(str(SFreq))
        f.write(', ')
        f.write(str(SFreq / vocab))

        # Technology
        f.write('\nTechnology: ')
        TFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[
                     :, bCount + eCount + pCount + sCount:bCount + eCount + pCount + sCount + tCount]).iloc[i, :]).sum() \
                        == 0:
                TFreq = TFreq + 1

        f.write(str(TFreq))
        f.write(', ')
        f.write(str(TFreq / vocab))
        f.write('\n\n')

        # j)
        f.write('j) the number and percentage of words with a frequency of one in the entire corpus\n')
        CFreq = 0
        for i in range(vocab):
            if (term_document_matrix.iloc[i, :]).sum() == 1:
                CFreq = CFreq + 1

        f.write(str(CFreq))
        f.write(', ')
        f.write(str(CFreq / vocab))
        f.write('\n\n')

        # k)
        f.write('k) your two favorite words (that are present in the vocabulary) and their log-prob\n')
        f.write('The log-probability of the word \'the\' is: ')
        # print(term_document_matrix.loc['the'].sum())
        f.write(str(math.log(term_document_matrix.loc['the'].sum() / (term_document_matrix.sum()).sum(), 10)))
        f.write('\nThe log-probability of the word \'in\' is: ')
        # print(term_document_matrix.loc['in'].sum())
        f.write(str(math.log(term_document_matrix.loc['in'].sum() / (term_document_matrix.sum()).sum(), 10)))
        f.write("\n\n")

    # 9) **********************************************************************************************************

        # Use frequencies rather than counts to train NB
        tfidf_transformer = TfidfTransformer()
        train_x_tfidf = tfidf_transformer.fit_transform(train_x)
        clf = MultinomialNB(alpha=.0001).fit(train_x_tfidf, train_y)

        # Predict using frequencies of test set
        test_x_tfidf = tfidf_transformer.transform(test_x)
        predicted = clf.predict(test_x_tfidf)

        # Comparison
        # print("\nComparison between predicted and true")
        # print(np.mean(predicted == test_y))

        # a)
        f.write('a)\n***********************************\nMultinomialNB default values, try 3\n\n')

        # b)
        f.write('b) Confusion Matrix\n')
        np.savetxt(f, confusion_matrix(test_y, predicted), fmt='%d')
        f.write('\n')

        # c) & d)
        f.write('c) precision, recall, and F1 measure\n')
        f.write('d) accuracy, macro average F1, and weight average F1 of model\n')
        f.write(classification_report(test_y, predicted,
                                      target_names=["Business", "Entertainment", "Politics", "Sports", "Tech"]))
        f.write('\n')

        # e)
        f.write('e) prior probability of each class\n')
        np.savetxt(f, clf.class_log_prior_, fmt='%e')
        f.write('\n')

        # f)
        f.write('f) size of the vocabulary\n')
        vocab = len(
            term_document_matrix[(term_document_matrix.select_dtypes(include=["number"]) != 0).any(1)].index)
        f.write(str(vocab))
        f.write('\n\n')

        # g)
        f.write('g) the number of word tokens in each class\n')
        f.write('Business: ')
        f.write(str(term_document_matrix.T.iloc[0:(bCount - 1)].sum().sum()))
        f.write('\nEntertainment: ')
        f.write(str(term_document_matrix.T.iloc[bCount:bCount + eCount - 1].sum().sum()))
        f.write('\nPolitics: ')
        f.write(str(term_document_matrix.T.iloc[bCount + eCount:bCount + eCount + pCount - 1].sum().sum()))
        f.write('\nSports: ')
        f.write(str(term_document_matrix.T.iloc[
                    bCount + eCount + pCount:bCount + eCount + pCount + sCount - 1].sum().sum()))
        f.write('\nTechnology: ')
        f.write(str(term_document_matrix.T.iloc[
                    bCount + eCount + pCount + sCount:bCount + eCount + pCount + sCount + tCount - 1].sum().sum()))

        f.write('\n\n')

        # h)
        f.write('h) the number of words in the entire corpus\n')
        f.write(str((term_document_matrix.sum()).sum()))
        f.write('\n\n')

        # i)

        f.write('i) the number and percentage of words with a frequency of zero in each class\n')

        # Business
        f.write('Business: ')
        BFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[:, 0:bCount]).iloc[i, :]).sum() == 0:
                BFreq = BFreq + 1

        f.write(str(BFreq))
        f.write(', ')
        f.write(str(BFreq / vocab))

        # Entertainment
        f.write('\nEntertainment: ')
        EFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[:, bCount:bCount + eCount]).iloc[i, :]).sum() == 0:
                EFreq = EFreq + 1

        f.write(str(EFreq))
        f.write(', ')
        f.write(str(EFreq / vocab))

        # Politics
        f.write('\nPolitics: ')
        PFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[:, bCount + eCount:bCount + eCount + pCount]).iloc[i, :]).sum() == 0:
                PFreq = PFreq + 1

        f.write(str(PFreq))
        f.write(', ')
        f.write(str(PFreq / vocab))

        # Sports
        f.write('\nSports: ')
        SFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[
                 :, bCount + eCount + pCount:bCount + eCount + pCount + sCount]).iloc[i, :]).sum() \
                    == 0:
                SFreq = SFreq + 1

        f.write(str(SFreq))
        f.write(', ')
        f.write(str(SFreq / vocab))

        # Technology
        f.write('\nTechnology: ')
        TFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[
                 :, bCount + eCount + pCount + sCount:bCount + eCount + pCount + sCount + tCount]).iloc[i, :]).sum() \
                    == 0:
                TFreq = TFreq + 1

        f.write(str(TFreq))
        f.write(', ')
        f.write(str(TFreq / vocab))
        f.write('\n\n')

        # j)
        f.write('j) the number and percentage of words with a frequency of one in the entire corpus\n')
        CFreq = 0
        for i in range(vocab):
            if (term_document_matrix.iloc[i, :]).sum() == 1:
                CFreq = CFreq + 1

        f.write(str(CFreq))
        f.write(', ')
        f.write(str(CFreq / vocab))
        f.write('\n\n')

        # k)
        f.write('k) your two favorite words (that are present in the vocabulary) and their log-prob\n')
        f.write('The log-probability of the word \'the\' is: ')
        # print(term_document_matrix.loc['the'].sum())
        f.write(str(math.log(term_document_matrix.loc['the'].sum() / (term_document_matrix.sum()).sum(), 10)))
        f.write('\nThe log-probability of the word \'in\' is: ')
        # print(term_document_matrix.loc['in'].sum())
        f.write(str(math.log(term_document_matrix.loc['in'].sum() / (term_document_matrix.sum()).sum(), 10)))
        f.write("\n\n")

    # 10) *********************************************************************************************************

        # Use frequencies rather than counts to train NB
        tfidf_transformer = TfidfTransformer()
        train_x_tfidf = tfidf_transformer.fit_transform(train_x)
        clf = MultinomialNB(alpha=.9).fit(train_x_tfidf, train_y)

        # Predict using frequencies of test set
        test_x_tfidf = tfidf_transformer.transform(test_x)
        predicted = clf.predict(test_x_tfidf)

        # Comparison
        # print("\nComparison between predicted and true")
        # print(np.mean(predicted == test_y))

        # a)
        f.write('a)\n***********************************\nMultinomialNB default values, try 4\n\n')

        # b)
        f.write('b) Confusion Matrix\n')
        np.savetxt(f, confusion_matrix(test_y, predicted), fmt='%d')
        f.write('\n')

        # c) & d)
        f.write('c) precision, recall, and F1 measure\n')
        f.write('d) accuracy, macro average F1, and weight average F1 of model\n')
        f.write(classification_report(test_y, predicted,
                                      target_names=["Business", "Entertainment", "Politics", "Sports", "Tech"]))
        f.write('\n')

        # e)
        f.write('e) prior probability of each class\n')
        np.savetxt(f, clf.class_log_prior_, fmt='%e')
        f.write('\n')

        # f)
        f.write('f) size of the vocabulary\n')
        vocab = len(
            term_document_matrix[(term_document_matrix.select_dtypes(include=["number"]) != 0).any(1)].index)
        f.write(str(vocab))
        f.write('\n\n')

        # g)
        f.write('g) the number of word tokens in each class\n')
        f.write('Business: ')
        f.write(str(term_document_matrix.T.iloc[0:(bCount - 1)].sum().sum()))
        f.write('\nEntertainment: ')
        f.write(str(term_document_matrix.T.iloc[bCount:bCount + eCount - 1].sum().sum()))
        f.write('\nPolitics: ')
        f.write(str(term_document_matrix.T.iloc[bCount + eCount:bCount + eCount + pCount - 1].sum().sum()))
        f.write('\nSports: ')
        f.write(str(term_document_matrix.T.iloc[
                    bCount + eCount + pCount:bCount + eCount + pCount + sCount - 1].sum().sum()))
        f.write('\nTechnology: ')
        f.write(str(term_document_matrix.T.iloc[
                    bCount + eCount + pCount + sCount:bCount + eCount + pCount + sCount + tCount - 1].sum().sum()))

        f.write('\n\n')

        # h)
        f.write('h) the number of words in the entire corpus\n')
        f.write(str((term_document_matrix.sum()).sum()))
        f.write('\n\n')

        # i)

        f.write('i) the number and percentage of words with a frequency of zero in each class\n')

        # Business
        f.write('Business: ')
        BFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[:, 0:bCount]).iloc[i, :]).sum() == 0:
                BFreq = BFreq + 1

        f.write(str(BFreq))
        f.write(', ')
        f.write(str(BFreq / vocab))

        # Entertainment
        f.write('\nEntertainment: ')
        EFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[:, bCount:bCount + eCount]).iloc[i, :]).sum() == 0:
                EFreq = EFreq + 1

        f.write(str(EFreq))
        f.write(', ')
        f.write(str(EFreq / vocab))

        # Politics
        f.write('\nPolitics: ')
        PFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[:, bCount + eCount:bCount + eCount + pCount]).iloc[i, :]).sum() == 0:
                PFreq = PFreq + 1

        f.write(str(PFreq))
        f.write(', ')
        f.write(str(PFreq / vocab))

        # Sports
        f.write('\nSports: ')
        SFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[
                 :, bCount + eCount + pCount:bCount + eCount + pCount + sCount]).iloc[i, :]).sum() \
                    == 0:
                SFreq = SFreq + 1

        f.write(str(SFreq))
        f.write(', ')
        f.write(str(SFreq / vocab))

        # Technology
        f.write('\nTechnology: ')
        TFreq = 0
        for i in range(vocab):
            if ((term_document_matrix.iloc[
                 :, bCount + eCount + pCount + sCount:bCount + eCount + pCount + sCount + tCount]).iloc[i, :]).sum() \
                    == 0:
                TFreq = TFreq + 1

        f.write(str(TFreq))
        f.write(', ')
        f.write(str(TFreq / vocab))
        f.write('\n\n')

        # j)
        f.write('j) the number and percentage of words with a frequency of one in the entire corpus\n')
        CFreq = 0
        for i in range(vocab):
            if (term_document_matrix.iloc[i, :]).sum() == 1:
                CFreq = CFreq + 1

        f.write(str(CFreq))
        f.write(', ')
        f.write(str(CFreq / vocab))
        f.write('\n\n')

        # k)
        f.write('k) your two favorite words (that are present in the vocabulary) and their log-prob\n')
        f.write('The log-probability of the word \'the\' is: ')
        # print(term_document_matrix.loc['the'].sum())
        f.write(str(math.log(term_document_matrix.loc['the'].sum() / (term_document_matrix.sum()).sum(), 10)))
        f.write('\nThe log-probability of the word \'in\' is: ')
        # print(term_document_matrix.loc['in'].sum())
        f.write(str(math.log(term_document_matrix.loc['in'].sum() / (term_document_matrix.sum()).sum(), 10)))

        # Close text file
        f.close()

