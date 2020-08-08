# TransformerForTopic-to-essay

# transformer for topic-to-essay 
 input : no more than 5 tokens, 
 output : a sequence
 
# main reqirements
python 3.5.3
tensorflow 1.12.0
cuda 9.0
cudnn 7.4

The data is put in the 'data/', you can train dircetly. 
Attention! because the input is unorderd, I drop the position embedding.
Bleu score and dist-n score is emplyed to test.
