import cPickle
import pandas as pd 
import numpy as np
# from scrape_allSides import retrieve 

def parse_data(lib, con, neutral):
    #len(lib) = 2025, len(con) = 1701, len(600) = 600
    # how to access sentence text
    print 'Liberal examples (out of ', len(lib), ' sentences): '
    for tree in lib[0:5]:
        print tree.get_words()

    print '\nConservative examples (out of ', len(con), ' sentences): '
    for tree in con[0:5]:
        print tree.get_words()

    print '\nNeutral examples (out of ', len(neutral), ' sentences): '
    for tree in neutral[0:5]:
        print tree.get_words()

    # how to access phrase labels for a particular tree
    ex_tree = lib[0]

    print '\nPhrase labels for one tree: '

    # see treeUtil.py for the tree class definition
    for node in ex_tree:

        # remember, only certain nodes have labels (see paper for details)
        if hasattr(node, 'label'):
            print node.label, ': ', node.get_words()

        # elif hasattr(node, 'print_leaf'):
        #     print node.print_leaf()

        # elif hasattr(node, 'get_words'):
        #     print node.get_words()

def write_combined(lib, con):
    data = []
    labels = []
    combined = lib + con
    for each in combined:
        for node in each:
            phrase = node.get_words()
            if hasattr(node, 'label'):
                # print 'phrase', node.get_words(), node.label 
                if node.label == "Conservative":
                    data.append(phrase)
                    labels.append(0)
                elif node.label == "Liberal":
                    data.append(phrase)
                    labels.append(1)
                elif node.label == "Neutral":
                    j = 0
                else:
                    print node.label
    return data, labels

def output(data, labels, datafile, labelfile):
    df = pd.DataFrame(np.array(data))
    lf = pd.DataFrame(np.array(labels))
    df.to_csv(datafile, index=False)
    lf.to_csv(labelfile, index=False)
    return len(df), len(lf)


def main():
    [lib, con, neutral] = cPickle.load(open('ibcData.pkl', 'rb'))
    # data1, labels1 = retrieve()
    data, labels = write_combined(lib, con)
    np.array(data)
    np.array(labels)
    # newData = np.append(data1, data, axis=0)
    # newLabels = np.append(labels1, labels, axis=0)
    df, lf = output(data, labels, 'IBCData_withPhrases.csv', 'IBCLabels_withPhrases.csv')
    print df, lf

if __name__ == "__main__":
    main()




