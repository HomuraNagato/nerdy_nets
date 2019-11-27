# nerdy_nets

## Model

Draft 1 
Based heavily on Seq2Seq language translation assignment. A transformer with an encoder encodes the input
paragraph, then is run through a decoder. Loss is measured by cross-entropy loss comparing the probabilities
output from the decoder and the reference summary.

Expected Final Model
..

## Metadata

 - unfilted vocab size: 3053570 
 - filtered vocab size: 717515 (0.765 reduction)
 - longest paragraph:   1015
 - longest summary:     397
 - num examples:        3,084,500 (3084410 lines)

## Preprocess

The data is gathered from https://tldr.webis.de. [about the data]

 - train: 80% (2450000 lines) test 20% (634410 lines)

Data is first batched using pandas then preprocessed in the training module. Preprocess involves removing
punctuation, lowercasing, padding by window size (retain text from 0:window_size should it be longer), and
converted to an ID by a vocab lookup. The vocabulary is pre-computed and stored on disk to reduce overhead.
Vocab preprocessed as in training, retain only words that occur more than once.

## TODO

 - ROUGE test
 - filter dataset on length of summary
 - query selector to reduce paragraph size
 - decoder_transformer shape error. Should be able to have different window sizes, but unable to currently.
   Work around is to have the same window sizes.
 - GPT-2 pre-trained model to initialize weights?
 - memory shortage! GCP tesla K80 has 11GB of memory. Vocab size of ~70,000 makes allocating memory for
   dense layer memory intensive. Work around, reduce batch size from 100 to 10.
    - Look into sparse memory computations to reduce computation
    - Look into reducing vocab size, would allow large window sizes
    - Look into tensorflow paradigms that may allow reducing memory requirements
 - incorporate checkpointing (similar to hw7)

As noted above, our greatest challenge is memory. Our initial plan was to have a maximum window size for
paragraph and summary (1024 and 400 respectively) and as large a vocab as possible, but that's just not
feasible in any memory setting. Can initialize model with 32x32x717515 ~= 230 million parameters. Use this
as a soft cap on output tensor shapes.
 

## Draft 1

Tested model on GCP with small amount of training and restricted window_size = 32.

Train 1000 examples
 - total_time: 1087 seconds
 - average_step_time: 109 seconds
 - model loss: 1.19
 - perplexity: 3.28

Test 1000 examples
 - total_time: 398 seconds
 - perplexity: 4.42
 - accuracy: 0.756

Examples
original paragraph  
i just finished my first year astronomy they enforce a rule that you neee 4560 credits otherwise they kick you out first semester i missed 6 credits one subject the first semester was the easy part i had to spend 60 hoursweek on stuff and projects of course i had a bit of time here and there never enough to take off a whole eveningnight to get wasted a really stressful year then some luck happened i barely passed optics and after classical mechanics and after that magnetic fields and calc ii flux integrals green stokes volume integrals jacobians etc i passed the year and now i am onto my second year luckely after the first year you cant be kicked i am planning for the second year to finally enjoy my student life and go to a party etc tldr extremely stressful due to 4560 credits and courses that make exams really hard and not just a redo of the previous exam

summary sentence  
extremely stressful due to 45 60 credits and courses that make exams really hard and not just a redo of the previous exam *STOP* *PAD* *PAD* *PAD* *PAD* *PAD* *PAD* *PAD* *PAD*  
decoded sentence  
extremely ruin due to garbage sweet app and shoes that make assume really hard and not just a loses of the previous legitimate *STOP* loses liberal sites liberal sites liberal ruin chaos

original paragraph  
my son was diagnosed with hypothyroidism when he was a month old so food has always been a big deal for me im always afraid when he doesnt eat worried that hes not getting enough and have literally resorted to feeding him ice cream and cookies just to make sure hes getting some type of nutrition the long and short of it he hasnt died yet in fact hasnt had any adverse effects at all when he skips meals except for being cranky when he realizes hes hungry ive stopped giving in to the bellymustbefull mentality shes still in many ways learning how to eat one thing i would recommend is remembering the size of her stomach and how much nutrition shes actually getting at her meals even a half piece of toast in a belly th$t tiny really fills it up disregarding any nutrition from the bread or her favorite topping i think i recall something about of an adult size serving kids only need something like 14 of that serving for the sam$ relative nutrition but that might just be an anecdote tldr dont worry if shes putting something other than water into her mouth and swallowing it there is some nutritional value going into her body shell be just fine hugs i know what youre going through and i have much commiseration that said this too will pass

summary sentence  
dont worry if shes putting something other than water into her mouth and swallowing it there is some nutritional value going into her body shell be just fine hugs i know *STOP*  
decoded sentence  
dont worry if shes speed something other than water into her mouth and assume it there is some loses value going into her body shoes be just fine mod i know *STOP*

original paragraph  
my best friend since high school is still my friend but i no longer consider him best now i call him my good friend even though ive known him longer then some of my best friends now this is because when he got caught smoking weed by his parents he threw me and my other friend j under the bus telling his parents it was ours and we introduced him to it then when he got caught again several months later he did the same thing he got caught a third time and guess what he did the same damn thing he tried lying about it to but im not dumb and i called him on his bullshit he got super dramatic and sort of teared up on why he did it but i told him bluntly that i had been put in the same situation and didnt rat my friends out plus his fiance is a total bitch when they were first engage he asked me to be a part of his wedding and i gladly said yes this was before he got caught smoking for a third time when he got caught by the way i was at my college living there about 2 hours away when this happened he told them it was me his parents and his fiance then pushed him to exclude me from the wedding and he did little to stop it so now im not in the wedding and his family doesnt think highly of me and that im a bad influence on him when in reality im one year away from graduating with a bs i have a internship and a job lined up after wards and im in the same relationship ive been in since high school while on the other hand he is working at a restaurant dropped out of the jc he was going to has cheated on his fiance and has a serious problem with marijuana sorry for the rant i guess im just fed up with him but ive known him so long i cant simply throw away a relationship because he is a pussy tldr friend got caught smoking threw me under the bus multiple times asked me to be in his wedding then a few months later kicked me out of it

summary sentence  
friend got caught smoking threw me under the bus multiple times asked me to be in his wedding then a few months later kicked me out of it *STOP* *PAD* *PAD* *PAD*  
decoded sentence  
friend got andor general failing me under the shoes multiple times asked me to be in his wedding then a few months later rent me out of it *STOP* liberal ruin chaos

