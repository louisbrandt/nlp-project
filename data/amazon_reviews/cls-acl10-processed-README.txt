Cross-Lingual Sentiment Dataset (v1.0)
======================================

Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
Date: 11.5.2010

Introduction
------------

This is the README of the processed Cross-Lingual Sentiment Dataset v1.0 first
used in Prettenhofer and Stein (2010). The dataset comprises about 
800.000 Amazon product reviews for three product categories---books,
dvds and music---written in four different languages: English, German,
French, and Japanese. The German, French, and Japanese reviews were
crawled from Amazon in November, 2009. The English reviews were
sampled from the Multi-Domain Sentiment Dataset (Blitzer et. al.,
2007). For more information about the construction of the dataset
see Prettenhofer and Stein (2010). 

For each language-category pair there are three sets of training
documents, test documents, and unlabeled documents. The training and
test sets comprise 2.000 documents each, whereas the number of
unlabeled documents varies from 9.000 - 170.000. 

The reviews are grouped by language and product category. The
directory structure looks as follows: 
.
|-- de
|   |-- books
|   |   |-- test.processed
|   |   |-- train.processed
|   |   |-- trans
|   |   |   `-- en
|   |   |       `-- books
|   |   |           `-- test.processed
|   |   `-- unlabeled.processed
|   |-- dvd
|   |-- ...
|-- fr
|-- ...
|-- dict

The *.processed files are plain text files containing one example per
line. The format is as follows:  

<term>:<term_count> <term>:<term_count> ... #label#:[positive|negative]

Where <term> is a unigram; and <term_count> is the frequency of the
unigram in the review (we consider text and headline of the review).   

test.processed... 1000 positive and 1000 negative test reviews. 
train.processed... 1000 positive and 1000 negative training reviews. 
unlabeled.processed... a balanced set of reviews which
are neither in the test nor in the trainings set (varies from 9.000 to
170.000 reviews).

For each of the three target languages: German, French, and Japanese
there exists a directory 'trans' which contains the translations of
the *test reviews* in the source language (i.e., English). We translated
the summary (i.e., the headline) and the review text using Google
Translate. The train.processed and unlabeled.processed files in the
'trans' directories are empty.  

The directory 'dict' contains the cached single word queries to Google
Translate, which was used as the word translation oracle.   

NOTE: The training and test sets are exactly the same as those used in
Prettenhofer and Stein (2010), however, for the experiments conducted
for our paper we restricted ourselves to the first 50.000 documents in
unlabeled.processed. 

NOTE: The order of the documents in unlabeled.processed is not
necessary the order of the documents in the unprocessed dataset. 

Preprocessing
-------------

We applied the following preprocessing and text normalization operations: 

0. We mapped smileys -- e.g., :-), :-(, :-D -- to unique identifiers
such as "<happy>" or "<sad>".  

1. We tokenized the text (for de, en, and fr we used a language
independent regex tokenizer provided by nltk.org; for Japanese we used
the morphological analyzer MECAB). 

2. We lower cased the tokens. 

3. We mapped tokens containing soley digits to the unique identifier "<num>".

4. (en only) We normalized contractions, e.g., don't -> do not

Acknowledgments
---------------

We kindly thank Mark Dredze and John Blitzer for the permission to
include a sample of the Multi-Domain Sentiment Dataset (Blitzer et. al.,
2007) in our dataset.

References
----------

Peter Prettenhofer, Benno Stein. Cross-Language Text Classification
using Structural Correspondence 
Learning. Association of Computational Linguistics (ACL), 2010. 

John Blitzer, Mark Dredze, Fernando Pereira. Biographies, Bollywood,
Boom-boxes and Blenders: Domain Adaptation for Sentiment
Classification. Association of Computational Linguistics (ACL), 2007. 

Citation Information
--------------------

This data was first used in Peter Prettenhofer and Benno Stein, 
``Cross-Language Text Classification using Structural Correspondence
Learning.'', Proceedings of the ACL, 2010.  

@InProceedings{Prettenhofer2010,
  author =       {Peter Prettenhofer and Benno Stein},
  title =        {Cross-Language Text Classification 
                  using Structural Correspondence Learning},
  booktitle =    {Proceedings of the ACL},
  year =         2010
}
