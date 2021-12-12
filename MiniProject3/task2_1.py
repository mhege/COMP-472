from random import seed
from random import randint
import csv
import pandas as pd
import gensim.downloader as api

if __name__ == '__main__':
    # load question-answer-words dataset
    questionWord = pd.read_csv('synonyms.csv')

    # load word embedding model
    modelName = 'glove-wiki-gigaword-100'
    wiki_100 = api.load(modelName)

    # seed random number generator
    seed(1)

    # correct label number
    cLabel = 0

    # answered without guessing
    nGuess = 0

    with open('glove-wiki-gigaword-100-details.csv', 'w', newline='') as csvfile:

        for i in range(len(questionWord)):

            # for appending in one iteration
            tempString = ""

            # question
            question = [(questionWord.T)[i][0]]

            # append question word
            tempString += question[0] + ','

            # answer
            answer = [(questionWord.T)[i][1]]

            # append answer word
            tempString += answer[0] + ','

            # words
            words = [(questionWord.T)[i][2], (questionWord.T)[i][3], (questionWord.T)[i][4], (questionWord.T)[i][5]]

            if (question[0] not in wiki_100.index_to_key) \
                    or (words[0] not in wiki_100.index_to_key and words[1] not in wiki_100.index_to_key
                        and words[2] not in wiki_100.index_to_key and words[3] not in wiki_100.index_to_key):

                # random guess
                guess = words[randint(0, 3)]
                nGuess += 1

                # append guess word
                tempString += guess + ',' + 'guess'

                # Correct label?
                if answer[0] == guess:
                    cLabel += 1

                # Check
                print(question)
                print(words)
                print(guess)
                print('')

            else:

                # Similarities
                sims = []
                for j in range(len(words)):
                    if words[j] not in wiki_100.index_to_key:
                        sims.append(0.0)
                    else:
                        sims.append(wiki_100.similarity(question, words[j])[0])

                # Find most similar word
                maxVal = max(sims)
                maxIndex = sims.index(maxVal)
                guess = words[maxIndex]

                # Correct label?
                if answer[0] == guess:
                    cLabel += 1

                    # append correct word
                    tempString += guess + ',' + 'correct'

                else:

                    # append wrong word
                    tempString += guess + ',' + 'wrong'

                # Check
                print(question)
                print(words)
                print(sims)
                print(guess)
                print('')

            csvfile.write(tempString + '\n')

        csvfile.close()

    with open('analysis.csv', 'a') as csvfile:

        csvfile.write(modelName + ',' + str(len(wiki_100)) + ',' + str(cLabel) + ','
                      + str(len(questionWord) - nGuess) + ','
                      + str(cLabel / (len(questionWord) - nGuess)) + '\n')

        csvfile.close()