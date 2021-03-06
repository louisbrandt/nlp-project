Cross-Lingual Sentiment Dataset (v1.0)
======================================

Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
Date: 11.5.2010

Introduction
------------

This is the README of the unprocessed Cross-Lingual Sentiment Dataset
v1.0 first used in Prettenhofer and Stein (2010). The dataset
comprises about 800.000 Amazon product reviews for three product
categories---books, dvds and music---written in four different
languages: English, German, French, and Japanese. The German, French,
and Japanese reviews were crawled from Amazon in November, 2009. The
English reviews were sampled from the Multi-Domain Sentiment Dataset
(Blitzer et. al., 2007). For more information about the construction
of the dataset see Prettenhofer and Stein (2010).  

For each language-category pair there are three sets of training
documents, test documents, and unlabeled documents. The training and
test sets comprise 2.000 documents each, whereas the number of
unlabeled documents varies from 9.000 - 170.000. 

The reviews are grouped by language and product category. The
directory structure looks as follows: 
.
|-- de
|   |-- books
|   |   |-- test.review
|   |   |-- train.review
|   |   |-- trans
|   |   |   `-- en
|   |   |       `-- books
|   |   |           `-- test.review
|   |   `-- unlabeled.review
|   |-- dvd
|   |-- ...
|-- fr
|-- ...
|-- dict

The *.review files are XML files (data format see below): 

test.review... 1000 positive and 1000 negative test reviews. 
train.review... 1000 positive and 1000 negative training reviews. 
unlabeled.reviews... a balanced set of reviews which
are neither in the test nor in the trainings set (varies from 9.000 to
170.000 reviews). 

For each of the three target languages: German, French, and Japanese
there exists a directory 'trans' which contains the translations of
the test reviews in the source language (i.e., English). We translated
the summary and the text fields using Google Translate. 

The directory 'dict' contains the cached single word queries to Google
Translate, which was used as the word translation oracle.  

NOTE: The training and test sets are exactly the same as those used in
Prettenhofer and Stein (2010), however, for the unlabeled documents we
restricted ourselves to the first 50.000 documents in
unlabeled.review.

NOTE: The order of the documents in unlabeled.review must not
correspond to the order in the processed format.

Data Format
-----------

Each review is represented by an <item> element. Each <item> element has
the following child elements: 
 
category: The product category.

rating: The number of stars (1-5).

badges: A list of <value> tags. Possible values are 'TOP 500 REVIEWER'
or 'REAL NAME' (only for French and Japanese).

asin: The article id. 

url: The url of the webpage from which the review was fetched. 

text: The review text. 

summary: The summary/headline of the review. 

title: The title of the article. 

reviewer: The name author

location: The location of the author (may be empty). 

date: The date of submission. 

helpfulness_votes: A pair of <value> elements; The first is the number
of positive votes and the second is the number of total votes.  

The German XML files have a slightly different schema than the French
and Japanese:
there is no <badges> element, however, there is a <realname> tag that
indicates whether or not the author of the review has a Real Name
badge.  

For the English XML files there exists no meta-data. 

Additionally, there is a template parsing pyton script 'parse.py' that
implements a parsing routine which you can modify to fit your needs.  

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
