import cPickle
import pandas as pd 
import numpy as np

if __name__ == '__main__':
    [lib, con, neutral] = cPickle.load(open('ibcData.pkl', 'rb'))
    data = []
    labels = []
    combined = lib + con
    for each in combined:
        for node in each:
            if hasattr(node, 'label'):
                if node.label == "Conservative":
                    data.append(each.get_words())
                    labels.append(0)
                elif node.label == "Liberal":
                    data.append(each.get_words())
                    labels.append(1)
                elif node.label == "Neutral":
                    data.append(each.get_words())
                    labels.append(2)
                else:
                    print "else??"
                   

    df = pd.DataFrame(np.array(data))
    lf = pd.DataFrame(np.array(labels))
    df.to_csv('processedData2.csv', index=False)
    lf.to_csv('processedLabels2.csv', index=False)
    print len(df)
    print len(lf)

    # # how to access sentence text
    # print 'Liberal examples (out of ', len(lib), ' sentences): '
    # for tree in lib[0:5]:
    #     print tree.get_words()

    # print '\nConservative examples (out of ', len(con), ' sentences): '
    # for tree in con[0:5]:
    #     print tree.get_words()

    # print '\nNeutral examples (out of ', len(neutral), ' sentences): '
    # for tree in neutral[0:5]:
    #     print tree.get_words()

    # # how to access phrase labels for a particular tree
    # ex_tree = lib[0]

    # print '\nPhrase labels for one tree: '

    # # see treeUtil.py for the tree class definition
    # for node in ex_tree:

    #     # remember, only certain nodes have labels (see paper for details)
    #     if hasattr(node, 'label'):
    #         print node.label, ': ', node.get_words()

    #     elif hasattr(node, 'print_leaf'):
    #         print node.print_leaf()

    #     elif hasattr(node, 'get_words'):
    #         print node.get_words()


#ReadMe 
    # len(lib) = 2025
    # len(con) = 1701
    # len(600) = 600

    # Each sub-array is an array of nodes of type nodeObj 
    # Each tree element is a nodeObj(iterable tree) containing: 
        # some nodeObjs containing a phrase with a label 
        # some nodeObjs containing a phrase with no label 
        # some leafObjs that break the phrases down into indivdiual words with some annotation