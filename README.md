# Tacotron with Emotion

Implementation of tacotron (TTS) with Tensorflow 2.0.0 heavily inspired by: </br>

General structure of algorithm: https://github.com/Kyubyong/tacotron </br>
Attention: https://www.tensorflow.org/tutorials/text/nmt_with_attention </br>

## Requirements

- Python=3.7
- tensorflow-gpu >= 2.0.0
- librosa
- tqdm
- matplotlib
- jamo
- unidecode
- inflect

## Data

For English, We have used ESD dataset (https://hltsingapore.github.io/ESD/index.html). </br>

## Training

First, set your parameters (including directory, language, etc) in hyperparams.py. For generating examples, I set "use_monotonic" and "normalize_attention" parameter as True. </br>
Then, you can just run training.py file as follows: </br>

<pre>
<code> 
python training.py 
</code>
</pre>

## Result

To show some example results, I trained with both English and Korean dataset applying Bahdanau monotonic attention with normalization.
Results of training English data (LJSpeech) are given below: </br>
![Alt Text](https://github.com/dabsdamoon/gif_save/blob/master/tacotron_English.gif) </br>
![Alt Text](https://github.com/dabsdamoon/gif_save/blob/master/tacotron_English_mel.png)
![Alt Text](https://github.com/dabsdamoon/gif_save/blob/master/tacotron_English_linear.png) </br>
</br>

algorithm have been trained for roughly 15 hours.

## Sample Synthesis

First, set your parameters in hyperparams.py. Note that you need to set "use_monotonic" and "normalize_attention" parameter as True if you have trained the algorithm in such way. Then, use the function "synthesizing" to generate the sentence you want. </br>

<pre>
<code>
synthesizing("The boy was there when the sun rose",1, hp)
</code>
</pre>

Finally, run synthesizing.py with console command:

<pre>
<code> 
python synthesizing.py 
</code>
</pre>

For audio samples, I uploaded synthesized English sentence of "The boy was there when the sun rose" and Korean sentence of "오늘 점심은 쌀국수 한그릇 먹고싶네요" in a folder "sample_synthesis". The algorithm has been trained 77000 steps for English (roughly 40 hours), and 67000 steps for Korean (roughly 15 hours). </br>

## Notes

- We have directly taken Dabin Moon implementation of tacotron and modified the encoder to accept one more input (emotion token)
- As I have mentioned earlier, training Korean dataset takes quite less time than training English dataset. Thus, if you can understand both languages, you may notice that Korean synthesizing result sounds better than English one. The English result will be better if you spend more time on training.
- Any comments on improving codes or questions are welcome, but it may take some time for me to respond.

March 2024, Dabin Moon
