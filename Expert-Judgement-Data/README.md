# Expert Judgemnet 

These six excel files contain Type-5 mismatches in the output og BioBERT-based model trained with st21pv dataset. 
Type-5 mismatch is where the entity extracted by NER model has the same tag as the annotated entity and overlaps in span.
We considered an information extraction task and asked a medical doctor to assess these mismatches and either confirm or
reject the extracted entity with granular scores.

The expert is given the original sentence in the test set, the annotated (gold-standard) entity,
and the entity predicted by the NER model for all Type-5 mismatches. 


Our goal is to: 

1) investigate the proportion of Type-5 extracted entities that are acceptable, 

2) set a benchmark of human experience from Type-5 errors. 



### Description of the columns of excel files:

column 1: The annotated entity in the test set of NER model.

column 2: The annotated tag in the test set of NER model. 

column 3: The entity extrcated by the NER model.

column 4: The score assigned by expert. 

column 5: The full sentence. 


### Scoring scheme: 

SCORE = 1: The predicted entity is wrong and gets rejected.  

SCORE = 2: The predicted entity is correct but an important piece of information is missing when seen in the full sentence. 

SCORE = 3: The predicted entity is correct but could be more complete. 

SCORE = 4: The predicted entity is equally correct and is accepted by the expert. 

SCORE = 5: The predicted entity is more complete than the annotated entity and is accepted by the expert. 
