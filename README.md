 IoT From Scratch Research and Development Theoretical REPO
===========================================================

# Informative Docs

### Important Links

- [Nginx Doc](https://github.com/nginx)
- [Build Nginx](https://nginx.org/en/docs/configure.html)
- [Free Nginx E-Book](https://www.nginx.com/resources/library/complete-nginx-cookbook/?utm_source=nginxorg&utm_medium=books&utm_campaign=ebooks&_ga=2.156844219.1469873918.1611598879-549172211.1611598879)
- [Mozilla TTS](https://github.com/mozilla/TTS)
- [WaveRNN vocoder](https://github.com/fatchord/WaveRNN/)
- [melgan vocoder](https://github.com/seungwonpark/melgan)
- [QUARTZNET-automatic speech recognition](https://github.com/isadrtdinov/quartznet) :- [paper](https://arxiv.org/pdf/1910.10261.pdf)
- [Microsoft MIMIC2](https://github.com/MycroftAI/mimic2)
- [Transformer TTS](https://github.com/as-ideas/TransformerTTS)
- [Google-Voice Builder](https://github.com/google/voice-builder)


<hr>


Text To Speech Concept ref- https://github.com/erogol/TTS-papers/
=====================================================================
(Feel free to suggest changes)
# Papers
- Merging Phoneme and Char representations: https://arxiv.org/pdf/1811.07240.pdf
- Tacotron transfer learning : https://arxiv.org/pdf/1904.06508.pdf
- phoneme timing from attention: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8683827
- SEMI-SUPERVISED TRAINING FOR IMPROVING DATA EFFICIENCY IN END-TO-ENDSPEECH SYNTHESI - https://arxiv.org/pdf/1808.10128.pdf
- Listening while Speaking: Speech Chain by Deep Learning - https://arxiv.org/pdf/1707.04879.pdf
- GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION: https://arxiv.org/pdf/1710.10467.pdf
- Es-Tacotron2: Multi-Task Tacotron 2 with Pre-Trained Estimated Network for Reducing the Over-Smoothness Problem: https://www.mdpi.com/2078-2489/10/4/131/pdf
	- Against Over-Smoothness
- FastSpeech: https://arxiv.org/pdf/1905.09263.pdf
- Learning singing from speech: https://arxiv.org/pdf/1912.10128.pdf
- TTS-GAN: https://arxiv.org/pdf/1909.11646.pdf
    - they use duration and linguistic features for en2en TTS.
    - Close to WaveNet performance.
- DurIAN: https://arxiv.org/pdf/1909.01700.pdf
    - Duration aware Tacotron
- MelNet: https://arxiv.org/abs/1906.01083
- AlignTTS: https://arxiv.org/pdf/2003.01950.pdf
- Unsupervised Speech Decomposition via Triple Information Bottleneck
    - https://arxiv.org/pdf/2004.11284.pdf
    - https://anonymous0818.github.io/
- FlowTron: https://arxiv.org/pdf/2005.05957.pdf
    - Inverse Autoregresive Flow on Tacotron like architecture
    - WaveGlow as vocoder.
    - Speech style embedding with Mixture of Gaussian model.
    - Model is large and havier than vanilla Tacotron
    - MOS values are slighly better than public Tacotron implementation.
- Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention : https://arxiv.org/pdf/1710.08969.pdf 
    
<details>
<summary> End-to-End Adversarial Text-to-Speech: http://arxiv.org/abs/2006.03575 (Click to Expand)</summary> 
	
  - end2end feed-forward TTS learning.
  - Character alignment has been done with a separate aligner module.	
  - The aligner predicts length of each character.
		- The center location of a char is found wrt the total length of the previous characters.
		- Char positions are interpolated with a Gaussian window wrt the real audio length.
	- audio output is computed in mu-law domain. (I don't have a reasoning for this)
	- use only 2 secs audio windows for traning.
	- GAN-TTS generator is used to produce audio signal.
	- RWD is used as a audio level discriminator.
	- MelD: They use BigGAN-deep architecture as spectrogram level discriminator regading the problem as image reconstruction.
	- Spectrogram loss
		- Using only adversarial feed-back is not enough to learn the char alignments. They use a spectrogram loss b/w predicted spectrograms and ground-truth specs.
		- Note that model predicts audio signals. Spectrograms above are computed from the generated audio.
		- Dynamic Time Wraping is used to compute a minimal-cost alignment b/w generated spectrograms and ground-truth.
		- It involves a dynamic programming approach to find a minimal-cost alignment.
	- Aligner length loss is used to penalize the aligner for predicting different than the real audio length.
	- They train the model with multi speaker dataset but report results on the best performing speaker.
	- Ablation Study importance of each component: (LengthLoss and SpectrogramLoss) > RWD > MelD > Phonemes > MultiSpeakerDataset.
	- My 2 cents: It is a feed forward model which provides end-2-end speech synthesis with no need to train a separate vocoder model. However, it is very complicated model with a lot of hyperparameters and implementation details. Also the final result is not close to the state of the art. I think we need to find specific algorithms for learning character alignments which would reduce the need of tunning a combination of different algorithms.
	  <img src="https://user-images.githubusercontent.com/1402048/84696449-d25e6b80-af4c-11ea-8b3a-66ede19124b0.png" width="50%">


</details>

<details>
<summary> Fast Speech2: http://arxiv.org/abs/2006.04558 (Click to Expand)</summary> 
	
  - Use phoneme durations generated by [MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/introduction.html) as labels to train a length regulator.
  - Thay use frame level F0 and L2 spectrogram norms (Variance Information) as additional features. 
  - Variance predictor module predicts the variance information at inference time.
  - Ablation study result improvements: model < model + L2_norm < model + L2_norm + F0
  ![image](https://user-images.githubusercontent.com/1402048/84696094-3c2a4580-af4c-11ea-8de3-4e918d651cd4.png)

</details>

<details>
<summary> Glow-TTS: https://arxiv.org/pdf/2005.11129.pdf (Click to Expand)</summary> 
	
  - Use Monotonic Alignment Search to learn the alignment b/w text and spectrogram
  - This alignment is used to train a Duration Predictor to be used at inference.
  - Encoder maps each character to a Gaussian Distribution.
  - Decoder maps each spectrogram frame to a latent vector using Normalizing Flow (Glow Layers)
  - Encoder and Decoder outputs are aligned with MAS.
  - At each iteration first the most probable alignment is found by MAS and this alignment is used to update mode parameters.
  - A duration predictor is trained to predict the number of spectrogram frames for each character.
  - At inference only the duration predictor is used instead of MAS
  - Encoder has the architecture of the TTS transformer with 2 updates
  - Instead of absolute positional encoding, they use realtive positional encoding.
  - They also use a residual connection for the Encoder Prenet.
  - Decoder has the same architecture as the Glow model.
  - They train both single and multi-speaker model.
  - It is showed experimentally, Glow-TTS is more robust against long sentences compared to original Tacotron2
  - 15x faster than Tacotron2 at inference
  - My 2 cents: Their samples sound not as natural as Tacotron. I believe normal attention models still generate more natural speech since the attention learns to map characters to model outputs directly. However, using Glow-TTS might be a good alternative for hard datasets.
  - Samples: https://github.com/jaywalnut310/glow-tts
  - Repository: https://github.com/jaywalnut310/glow-tts
  ![image](https://user-images.githubusercontent.com/1402048/85527284-06035a80-b60b-11ea-8165-b2f3e841f77f.png)


</details>

<details>
<summary> Non-Autoregressive Neural Text-to-Speech: http://arxiv.org/abs/1905.08459 (Click to Expand)</summary> 
	
   - A derivation of Deep Voice 3 model using non-causal convolutional layers.
   - Teacher-Student paradigm to train annon-autoregressive student with multiple attention blocks from an autoregressive teacher model.
   - The teacher is used to generate text-to-spectrogram alignments to be used by the student model.
   - The model is trained with two loss functions for attention alignment and spectrogram generation.
   - Multi attention blocks refine the attention alignment layer by layer.
   - The student uses dot-product attention with query, key and value vectors. The query is only positinal encoding vectors. The key and the value are the encoder outputs.
   - Proposed model is heavily tied to the positional encoding which also relies on different constant values.
  ![image](https://user-images.githubusercontent.com/1402048/87929772-3e218000-ca87-11ea-9f13-9869bee96b57.png)
</details>

<details>
<summary> Double Decoder Consistency: https://erogol.com/solving-attention-problems-of-tts-models-with-double-decoder-consistency (Click to Expand)</summary> 
	
   - The model uses a Tacotron like architecture but with 2 decoders and a postnet.
   - DDC uses two synchronous decoders using different reduction rates.
   - The decoders use different reduction rates thus they compute outputs in different granularities and learn different aspects of the input data.
   - The model uses the consistency between these two decoders to increase robustness of learned text-to-spectrogram alignment.
   - The model also applies a refinement to the final decoder output by applying the postnet iteratively multiple times. 
   - DDC uses Batch Normalization in the prenet module and drops Dropout layers.
   - DDC uses gradual training to reduce the total training time.
   - We use a Multi-Band Melgan Generator as a vocoder trained with Multiple Random Window Discriminators differently than the original work.
   - We are able to train a DDC model only in 2 days with a single GPU and the final model is able to generate faster than real-time speech on a CPU.
  Demo page: https://erogol.github.io/ddc-samples/
  Code: https://github.com/mozilla/TTS
  ![image](https://erogol.com/wp-content/uploads/2020/06/DDC_overview-1536x1220.png)
</details>

## Multi-Speaker Papers
- Training Multi-Speaker Neural Text-to-Speech Systems using Speaker-Imbalanced Speech Corpora - https://arxiv.org/abs/1904.00771
- Deep Voice 2 - https://papers.nips.cc/paper/6889-deep-voice-2-multi-speaker-neural-text-to-speech.pdf
- Sample Efficient Adaptive TTS - https://openreview.net/pdf?id=rkzjUoAcFX
	- WaveNet + Speaker Embedding approach
- Voice Loop - https://arxiv.org/abs/1707.06588
- MODELING MULTI-SPEAKER LATENT SPACE TO IMPROVE NEURAL TTS QUICK ENROLLING NEW SPEAKER AND ENHANCING PREMIUM VOICE - https://arxiv.org/pdf/1812.05253.pdf
- Transfer learning from speaker verification to multispeaker text-to-speech synthesis - https://arxiv.org/pdf/1806.04558.pdf
- Fitting new speakers based on a short untranscribed sample - https://arxiv.org/pdf/1802.06984.pdf
- Generalized end-to-end loss for speaker verification

<details>
<summary> Semi-supervised Learning for Multi-speaker Text-to-speech Synthesis Using Discrete Speech Representation: http://arxiv.org/abs/2005.08024 </summary> 
	
   - Train a multi-speaker TTS model with only an hour long paired data (text-to-voice alignment) and more unpaired (only voide) data.
   - It learns a code book with each code word corresponds to a single phoneme.
   - The code-book is aligned to phonemes using the paired data and CTC algorithm.
   - This code book functions like a proxy to implicitly estimate the phoneme sequence of the unpaired data.
   - They stack Tacotron2 model on top to perform TTS using the code word embeddings generated by the initial part of the model.
   - They beat the benchmark methods in 1hr long paired data setting.
   - They don't report full paired data results.
   - They don't have a good ablation study which could be interesting to see how different parts of the model contribute to the performance.
   - They use Griffin-Lim as a vocoder thus there is space for improvement.

  Demo page: https://ttaoretw.github.io/multispkr-semi-tts/demo.html <br>
  Code: https://github.com/ttaoREtw/semi-tts
  ![image](https://user-images.githubusercontent.com/1402048/93603135-de325180-f9c3-11ea-8081-7c1c3390b9f0.png)
</details>
<details>
<summary> Attentron: Few-shot Text-to-Speech Exploiting Attention-based Variable Length Embedding: https://arxiv.org/abs/2005.08484 </summary> 
	
   - Use two encoders to learn speaker depended features.
   - Coarse encoder learns a global speaker embedding vector based on provided reference spectrograms. 
   - Fine encoder learns a variable length embedding keeping the temporal dimention in cooperation with a attention module.
   - The attention selects important reference spectrogram frames to synthesize target speech.
   - Pre-train the model with a single speaker dataset first (LJSpeech for 30k iters.)
   - Fine-tune the model with a multi-speaker dataset. (VCTK for 70k iters.)
   - It achieves slightly better metrics in comparison to using x-vectors from speaker classification model and VAE based reference audio encoder. 
   

  Demo page: https://hyperconnect.github.io/Attentron/ <br>
  ![image](https://user-images.githubusercontent.com/1402048/105180385-cc57eb00-5b2a-11eb-9b9b-201153ee2029.png)
  ![image](https://user-images.githubusercontent.com/1402048/105180441-e1347e80-5b2a-11eb-8968-3731a0119ff4.png)
</details>

## Attention
- LOCATION-RELATIVE ATTENTION MECHANISMS FOR ROBUST LONG-FORMSPEECH SYNTHESIS : https://arxiv.org/pdf/1910.10288.pdf

## Vocoders
- MelGAN: https://arxiv.org/pdf/1910.06711.pdf
- ParallelWaveGAN: https://arxiv.org/pdf/1910.11480.pdf
    - Multi scale STFT loss
    - ~1M model parameters (very small)
    - Slightly worse than WaveRNN 
- Improving FFTNEt
    - https://www.okamotocamera.com/slt_2018.pdfF
    - https://www.okamotocamera.com/slt_2018.pdf
- FFTnet
    - https://gfx.cs.princeton.edu/pubs/Jin_2018_FAR/clips/clips.php
    - https://gfx.cs.princeton.edu/pubs/Jin_2018_FAR/fftnet-jin2018.pdf
- SPEECH WAVEFORM RECONSTRUCTION USING CONVOLUTIONAL NEURALNETWORKS WITH NOISE AND PERIODIC INPUTS
    - 150.162.46.34:8080/icassp2019/ICASSP2019/pdfs/0007045.pdf
- Towards Achieveing Robust Universal Vocoding
    - https://arxiv.org/pdf/1811.06292.pdf
- LPCNet
    - https://arxiv.org/pdf/1810.11846.pdf
    - https://arxiv.org/pdf/2001.11686.pdf
- ExciteNet
    - https://arxiv.org/pdf/1811.04769v3.pdf
- GELP: GAN-Excited Linear Prediction for Speech Synthesis fromMel-spectrogram
    - https://arxiv.org/pdf/1904.03976v3.pdf
- High Fidelity Speech Synthesis with Adversarial Networks: https://arxiv.org/abs/1909.11646
    - GAN-TTS, end-to-end speech synthesis
    - Uses duration and linguistic features
    - Duration and acoustic features are predicted by additional models.
    - Random Window Discriminator: Ingest not the whole Voice sample but random
    windows.
    - Multiple RWDs. Some conditional and some unconditional. (conditioned on 
    input features)
    - Punchline: Use randomly sampled windows with different window sizes for D.
    - Shared results sounds mechanical that shows the limits of non-neural
    acoustic features.
- Multi-Band MelGAN: https://arxiv.org/abs/2005.05106
    - Use PWGAN losses instead of feature-matching loss.
    - Using a larger receptive field boosts model performance significantly.
    - Generator pretraining for 200k iters.
    - Multi-Band voice signal prediction. The output is summation of 4 different
    band predictions with PQMF synthesis filters.
    - Multi-band model has 1.9m parameters (quite small).
    - Claimed to be 7x faster than MelGAN
    - On a Chinese dataset: MOS 4.22
 - WaveGLow: https://arxiv.org/abs/1811.00002
 	- Very large model (268M parameters)
	- Hard to train since on 12GB GPU it can only takes batch size 1.
	- Real-time inference due to the use of convolutions.
	- Based on Invertable Normalizing Flow. (Great tutorial https://blog.evjang.com/2018/01/nf1.html
)
	- Model learns and invetible mapping of audio samples to mel-spectrograms with Max Likelihood loss.
	- In inference network runs in reverse direction and give mel-specs are converted to audio samples.
	- Training has been done using 8 Nvidia V100 with 32GB ram, batch size 24. (Expensive)
	
 - SqueezeWave: https://arxiv.org/pdf/2001.05685.pdf , code: https://github.com/tianrengao/SqueezeWave 
 	- ~5-13x faster than real-time
	- WaveGlow redanduncies: Long audio samples, upsamples mel-specs, large channel dimensions in WN function.
	- Fixes: More but shorter audio samples as input,  (L=2000, C=8 vs L=64, C=256)
	- L=64 matchs the mel-spec resolution so no upsampling necessary.
	- Use depth-wise separable convolutions in WN modules.
	- Use regular convolution instead of dilated since audio samples are shorter.
	- Do not split module outputs to residual and network output, assuming these vectors are almost identical.
	- Training has been done using Titan RTX 24GB batch size 96 for 600k iterations.
	- MOS on LJSpeech: WaveGLow - 4.57, SqueezeWave (L=128 C=256) - 4.07 and SqueezeWave (L=64 C=256) - 3.77
	- Smallest model has 21K samples per second on Raspi3.
	
<details>
<summary>WaveGrad: https://arxiv.org/pdf/2009.00713.pdf </summary> 
	
  - It is based on Probability Diffusion and Lagenvin Dynamics
  - The base idea is to learn a function that maps a known distribution to target data distribution iteratively.
  - They report 0.2 real-time factor on a GPU but CPU performance is not shared.
  - In the example code below, the author reports that the model converges after 2 days of training on a single GPU.
  - MOS scores on the paper are not compherensive enough but shows comparable performance to known models like WaveRNN and WaveNet.
  
  Code: https://github.com/ivanvovk/WaveGrad
  ![image](https://user-images.githubusercontent.com/1402048/93461311-e071ae80-f8e4-11ea-82c6-e631301bbd27.png)
</details>



# From the Internet (Blogs, Videos etc)

## Videos
### Paper Discussion
- Tacotron 2 : https://www.youtube.com/watch?v=2iarxxm-v9w

### Talks
- End-to-End Text-to-Speech Synthesis, Part 1 : https://www.youtube.com/watch?v=RNKrq26Z0ZQ
- Speech synthesis from neural decoding of spoken sentences | AISC : https://www.youtube.com/watch?v=MNDtMDPmnMo
- Generative Text-to-Speech Synthesis : https://www.youtube.com/watch?v=j4mVEAnKiNg
- SPEECH SYNTHESIS FOR THE GAMING INDUSTRY : https://www.youtube.com/watch?v=aOHAYe4A-2Q

### General
- Modern Text-to-Speech Systems Review : https://www.youtube.com/watch?v=8rXLSc-ZcRY

## Blogs
- Text to Speech Deep Learning Architectures : http://www.erogol.com/text-speech-deep-learning-architectures/






<hr>

GNU General Public License
==========================

_Version 3, 29 June 2007_  
_Copyright © 2007 Free Software Foundation, Inc. &lt;<http://fsf.org/>&gt;_

Everyone is permitted to copy and distribute verbatim copies of this license
document, but changing it is not allowed.

## Preamble

The GNU General Public License is a free, copyleft license for software and other
kinds of works.

The licenses for most software and other practical works are designed to take away
your freedom to share and change the works. By contrast, the GNU General Public
License is intended to guarantee your freedom to share and change all versions of a
program--to make sure it remains free software for all its users. We, the Free
Software Foundation, use the GNU General Public License for most of our software; it
applies also to any other work released this way by its authors. You can apply it to
your programs, too.

When we speak of free software, we are referring to freedom, not price. Our General
Public Licenses are designed to make sure that you have the freedom to distribute
copies of free software (and charge for them if you wish), that you receive source
code or can get it if you want it, that you can change the software or use pieces of
it in new free programs, and that you know you can do these things.

To protect your rights, we need to prevent others from denying you these rights or
asking you to surrender the rights. Therefore, you have certain responsibilities if
you distribute copies of the software, or if you modify it: responsibilities to
respect the freedom of others.

For example, if you distribute copies of such a program, whether gratis or for a fee,
you must pass on to the recipients the same freedoms that you received. You must make
sure that they, too, receive or can get the source code. And you must show them these
terms so they know their rights.

Developers that use the GNU GPL protect your rights with two steps: **(1)** assert
copyright on the software, and **(2)** offer you this License giving you legal permission
to copy, distribute and/or modify it.

For the developers' and authors' protection, the GPL clearly explains that there is
no warranty for this free software. For both users' and authors' sake, the GPL
requires that modified versions be marked as changed, so that their problems will not
be attributed erroneously to authors of previous versions.

Some devices are designed to deny users access to install or run modified versions of
the software inside them, although the manufacturer can do so. This is fundamentally
incompatible with the aim of protecting users' freedom to change the software. The
systematic pattern of such abuse occurs in the area of products for individuals to
use, which is precisely where it is most unacceptable. Therefore, we have designed
this version of the GPL to prohibit the practice for those products. If such problems
arise substantially in other domains, we stand ready to extend this provision to
those domains in future versions of the GPL, as needed to protect the freedom of
users.

Finally, every program is threatened constantly by software patents. States should
not allow patents to restrict development and use of software on general-purpose
computers, but in those that do, we wish to avoid the special danger that patents
applied to a free program could make it effectively proprietary. To prevent this, the
GPL assures that patents cannot be used to render the program non-free.

The precise terms and conditions for copying, distribution and modification follow.

## TERMS AND CONDITIONS

### 0. Definitions

“This License” refers to version 3 of the GNU General Public License.

“Copyright” also means copyright-like laws that apply to other kinds of
works, such as semiconductor masks.

“The Program” refers to any copyrightable work licensed under this
License. Each licensee is addressed as “you”. “Licensees” and
“recipients” may be individuals or organizations.

To “modify” a work means to copy from or adapt all or part of the work in
a fashion requiring copyright permission, other than the making of an exact copy. The
resulting work is called a “modified version” of the earlier work or a
work “based on” the earlier work.

A “covered work” means either the unmodified Program or a work based on
the Program.

To “propagate” a work means to do anything with it that, without
permission, would make you directly or secondarily liable for infringement under
applicable copyright law, except executing it on a computer or modifying a private
copy. Propagation includes copying, distribution (with or without modification),
making available to the public, and in some countries other activities as well.

To “convey” a work means any kind of propagation that enables other
parties to make or receive copies. Mere interaction with a user through a computer
network, with no transfer of a copy, is not conveying.

An interactive user interface displays “Appropriate Legal Notices” to the
extent that it includes a convenient and prominently visible feature that **(1)**
displays an appropriate copyright notice, and **(2)** tells the user that there is no
warranty for the work (except to the extent that warranties are provided), that
licensees may convey the work under this License, and how to view a copy of this
License. If the interface presents a list of user commands or options, such as a
menu, a prominent item in the list meets this criterion.

### 1. Source Code

The “source code” for a work means the preferred form of the work for
making modifications to it. “Object code” means any non-source form of a
work.

A “Standard Interface” means an interface that either is an official
standard defined by a recognized standards body, or, in the case of interfaces
specified for a particular programming language, one that is widely used among
developers working in that language.

The “System Libraries” of an executable work include anything, other than
the work as a whole, that **(a)** is included in the normal form of packaging a Major
Component, but which is not part of that Major Component, and **(b)** serves only to
enable use of the work with that Major Component, or to implement a Standard
Interface for which an implementation is available to the public in source code form.
A “Major Component”, in this context, means a major essential component
(kernel, window system, and so on) of the specific operating system (if any) on which
the executable work runs, or a compiler used to produce the work, or an object code
interpreter used to run it.

The “Corresponding Source” for a work in object code form means all the
source code needed to generate, install, and (for an executable work) run the object
code and to modify the work, including scripts to control those activities. However,
it does not include the work's System Libraries, or general-purpose tools or
generally available free programs which are used unmodified in performing those
activities but which are not part of the work. For example, Corresponding Source
includes interface definition files associated with source files for the work, and
the source code for shared libraries and dynamically linked subprograms that the work
is specifically designed to require, such as by intimate data communication or
control flow between those subprograms and other parts of the work.

The Corresponding Source need not include anything that users can regenerate
automatically from other parts of the Corresponding Source.

The Corresponding Source for a work in source code form is that same work.

### 2. Basic Permissions

All rights granted under this License are granted for the term of copyright on the
Program, and are irrevocable provided the stated conditions are met. This License
explicitly affirms your unlimited permission to run the unmodified Program. The
output from running a covered work is covered by this License only if the output,
given its content, constitutes a covered work. This License acknowledges your rights
of fair use or other equivalent, as provided by copyright law.

You may make, run and propagate covered works that you do not convey, without
conditions so long as your license otherwise remains in force. You may convey covered
works to others for the sole purpose of having them make modifications exclusively
for you, or provide you with facilities for running those works, provided that you
comply with the terms of this License in conveying all material for which you do not
control copyright. Those thus making or running the covered works for you must do so
exclusively on your behalf, under your direction and control, on terms that prohibit
them from making any copies of your copyrighted material outside their relationship
with you.

Conveying under any other circumstances is permitted solely under the conditions
stated below. Sublicensing is not allowed; section 10 makes it unnecessary.

### 3. Protecting Users' Legal Rights From Anti-Circumvention Law

No covered work shall be deemed part of an effective technological measure under any
applicable law fulfilling obligations under article 11 of the WIPO copyright treaty
adopted on 20 December 1996, or similar laws prohibiting or restricting circumvention
of such measures.

When you convey a covered work, you waive any legal power to forbid circumvention of
technological measures to the extent such circumvention is effected by exercising
rights under this License with respect to the covered work, and you disclaim any
intention to limit operation or modification of the work as a means of enforcing,
against the work's users, your or third parties' legal rights to forbid circumvention
of technological measures.

### 4. Conveying Verbatim Copies

You may convey verbatim copies of the Program's source code as you receive it, in any
medium, provided that you conspicuously and appropriately publish on each copy an
appropriate copyright notice; keep intact all notices stating that this License and
any non-permissive terms added in accord with section 7 apply to the code; keep
intact all notices of the absence of any warranty; and give all recipients a copy of
this License along with the Program.

You may charge any price or no price for each copy that you convey, and you may offer
support or warranty protection for a fee.

### 5. Conveying Modified Source Versions

You may convey a work based on the Program, or the modifications to produce it from
the Program, in the form of source code under the terms of section 4, provided that
you also meet all of these conditions:

* **a)** The work must carry prominent notices stating that you modified it, and giving a
relevant date.
* **b)** The work must carry prominent notices stating that it is released under this
License and any conditions added under section 7. This requirement modifies the
requirement in section 4 to “keep intact all notices”.
* **c)** You must license the entire work, as a whole, under this License to anyone who
comes into possession of a copy. This License will therefore apply, along with any
applicable section 7 additional terms, to the whole of the work, and all its parts,
regardless of how they are packaged. This License gives no permission to license the
work in any other way, but it does not invalidate such permission if you have
separately received it.
* **d)** If the work has interactive user interfaces, each must display Appropriate Legal
Notices; however, if the Program has interactive interfaces that do not display
Appropriate Legal Notices, your work need not make them do so.

A compilation of a covered work with other separate and independent works, which are
not by their nature extensions of the covered work, and which are not combined with
it such as to form a larger program, in or on a volume of a storage or distribution
medium, is called an “aggregate” if the compilation and its resulting
copyright are not used to limit the access or legal rights of the compilation's users
beyond what the individual works permit. Inclusion of a covered work in an aggregate
does not cause this License to apply to the other parts of the aggregate.

### 6. Conveying Non-Source Forms

You may convey a covered work in object code form under the terms of sections 4 and
5, provided that you also convey the machine-readable Corresponding Source under the
terms of this License, in one of these ways:

* **a)** Convey the object code in, or embodied in, a physical product (including a
physical distribution medium), accompanied by the Corresponding Source fixed on a
durable physical medium customarily used for software interchange.
* **b)** Convey the object code in, or embodied in, a physical product (including a
physical distribution medium), accompanied by a written offer, valid for at least
three years and valid for as long as you offer spare parts or customer support for
that product model, to give anyone who possesses the object code either **(1)** a copy of
the Corresponding Source for all the software in the product that is covered by this
License, on a durable physical medium customarily used for software interchange, for
a price no more than your reasonable cost of physically performing this conveying of
source, or **(2)** access to copy the Corresponding Source from a network server at no
charge.
* **c)** Convey individual copies of the object code with a copy of the written offer to
provide the Corresponding Source. This alternative is allowed only occasionally and
noncommercially, and only if you received the object code with such an offer, in
accord with subsection 6b.
* **d)** Convey the object code by offering access from a designated place (gratis or for
a charge), and offer equivalent access to the Corresponding Source in the same way
through the same place at no further charge. You need not require recipients to copy
the Corresponding Source along with the object code. If the place to copy the object
code is a network server, the Corresponding Source may be on a different server
(operated by you or a third party) that supports equivalent copying facilities,
provided you maintain clear directions next to the object code saying where to find
the Corresponding Source. Regardless of what server hosts the Corresponding Source,
you remain obligated to ensure that it is available for as long as needed to satisfy
these requirements.
* **e)** Convey the object code using peer-to-peer transmission, provided you inform
other peers where the object code and Corresponding Source of the work are being
offered to the general public at no charge under subsection 6d.

A separable portion of the object code, whose source code is excluded from the
Corresponding Source as a System Library, need not be included in conveying the
object code work.

A “User Product” is either **(1)** a “consumer product”, which
means any tangible personal property which is normally used for personal, family, or
household purposes, or **(2)** anything designed or sold for incorporation into a
dwelling. In determining whether a product is a consumer product, doubtful cases
shall be resolved in favor of coverage. For a particular product received by a
particular user, “normally used” refers to a typical or common use of
that class of product, regardless of the status of the particular user or of the way
in which the particular user actually uses, or expects or is expected to use, the
product. A product is a consumer product regardless of whether the product has
substantial commercial, industrial or non-consumer uses, unless such uses represent
the only significant mode of use of the product.

“Installation Information” for a User Product means any methods,
procedures, authorization keys, or other information required to install and execute
modified versions of a covered work in that User Product from a modified version of
its Corresponding Source. The information must suffice to ensure that the continued
functioning of the modified object code is in no case prevented or interfered with
solely because modification has been made.

If you convey an object code work under this section in, or with, or specifically for
use in, a User Product, and the conveying occurs as part of a transaction in which
the right of possession and use of the User Product is transferred to the recipient
in perpetuity or for a fixed term (regardless of how the transaction is
characterized), the Corresponding Source conveyed under this section must be
accompanied by the Installation Information. But this requirement does not apply if
neither you nor any third party retains the ability to install modified object code
on the User Product (for example, the work has been installed in ROM).

The requirement to provide Installation Information does not include a requirement to
continue to provide support service, warranty, or updates for a work that has been
modified or installed by the recipient, or for the User Product in which it has been
modified or installed. Access to a network may be denied when the modification itself
materially and adversely affects the operation of the network or violates the rules
and protocols for communication across the network.

Corresponding Source conveyed, and Installation Information provided, in accord with
this section must be in a format that is publicly documented (and with an
implementation available to the public in source code form), and must require no
special password or key for unpacking, reading or copying.

### 7. Additional Terms

“Additional permissions” are terms that supplement the terms of this
License by making exceptions from one or more of its conditions. Additional
permissions that are applicable to the entire Program shall be treated as though they
were included in this License, to the extent that they are valid under applicable
law. If additional permissions apply only to part of the Program, that part may be
used separately under those permissions, but the entire Program remains governed by
this License without regard to the additional permissions.

When you convey a copy of a covered work, you may at your option remove any
additional permissions from that copy, or from any part of it. (Additional
permissions may be written to require their own removal in certain cases when you
modify the work.) You may place additional permissions on material, added by you to a
covered work, for which you have or can give appropriate copyright permission.

Notwithstanding any other provision of this License, for material you add to a
covered work, you may (if authorized by the copyright holders of that material)
supplement the terms of this License with terms:

* **a)** Disclaiming warranty or limiting liability differently from the terms of
sections 15 and 16 of this License; or
* **b)** Requiring preservation of specified reasonable legal notices or author
attributions in that material or in the Appropriate Legal Notices displayed by works
containing it; or
* **c)** Prohibiting misrepresentation of the origin of that material, or requiring that
modified versions of such material be marked in reasonable ways as different from the
original version; or
* **d)** Limiting the use for publicity purposes of names of licensors or authors of the
material; or
* **e)** Declining to grant rights under trademark law for use of some trade names,
trademarks, or service marks; or
* **f)** Requiring indemnification of licensors and authors of that material by anyone
who conveys the material (or modified versions of it) with contractual assumptions of
liability to the recipient, for any liability that these contractual assumptions
directly impose on those licensors and authors.

All other non-permissive additional terms are considered “further
restrictions” within the meaning of section 10. If the Program as you received
it, or any part of it, contains a notice stating that it is governed by this License
along with a term that is a further restriction, you may remove that term. If a
license document contains a further restriction but permits relicensing or conveying
under this License, you may add to a covered work material governed by the terms of
that license document, provided that the further restriction does not survive such
relicensing or conveying.

If you add terms to a covered work in accord with this section, you must place, in
the relevant source files, a statement of the additional terms that apply to those
files, or a notice indicating where to find the applicable terms.

Additional terms, permissive or non-permissive, may be stated in the form of a
separately written license, or stated as exceptions; the above requirements apply
either way.

### 8. Termination

You may not propagate or modify a covered work except as expressly provided under
this License. Any attempt otherwise to propagate or modify it is void, and will
automatically terminate your rights under this License (including any patent licenses
granted under the third paragraph of section 11).

However, if you cease all violation of this License, then your license from a
particular copyright holder is reinstated **(a)** provisionally, unless and until the
copyright holder explicitly and finally terminates your license, and **(b)** permanently,
if the copyright holder fails to notify you of the violation by some reasonable means
prior to 60 days after the cessation.

Moreover, your license from a particular copyright holder is reinstated permanently
if the copyright holder notifies you of the violation by some reasonable means, this
is the first time you have received notice of violation of this License (for any
work) from that copyright holder, and you cure the violation prior to 30 days after
your receipt of the notice.

Termination of your rights under this section does not terminate the licenses of
parties who have received copies or rights from you under this License. If your
rights have been terminated and not permanently reinstated, you do not qualify to
receive new licenses for the same material under section 10.

### 9. Acceptance Not Required for Having Copies

You are not required to accept this License in order to receive or run a copy of the
Program. Ancillary propagation of a covered work occurring solely as a consequence of
using peer-to-peer transmission to receive a copy likewise does not require
acceptance. However, nothing other than this License grants you permission to
propagate or modify any covered work. These actions infringe copyright if you do not
accept this License. Therefore, by modifying or propagating a covered work, you
indicate your acceptance of this License to do so.

### 10. Automatic Licensing of Downstream Recipients

Each time you convey a covered work, the recipient automatically receives a license
from the original licensors, to run, modify and propagate that work, subject to this
License. You are not responsible for enforcing compliance by third parties with this
License.

An “entity transaction” is a transaction transferring control of an
organization, or substantially all assets of one, or subdividing an organization, or
merging organizations. If propagation of a covered work results from an entity
transaction, each party to that transaction who receives a copy of the work also
receives whatever licenses to the work the party's predecessor in interest had or
could give under the previous paragraph, plus a right to possession of the
Corresponding Source of the work from the predecessor in interest, if the predecessor
has it or can get it with reasonable efforts.

You may not impose any further restrictions on the exercise of the rights granted or
affirmed under this License. For example, you may not impose a license fee, royalty,
or other charge for exercise of rights granted under this License, and you may not
initiate litigation (including a cross-claim or counterclaim in a lawsuit) alleging
that any patent claim is infringed by making, using, selling, offering for sale, or
importing the Program or any portion of it.

### 11. Patents

A “contributor” is a copyright holder who authorizes use under this
License of the Program or a work on which the Program is based. The work thus
licensed is called the contributor's “contributor version”.

A contributor's “essential patent claims” are all patent claims owned or
controlled by the contributor, whether already acquired or hereafter acquired, that
would be infringed by some manner, permitted by this License, of making, using, or
selling its contributor version, but do not include claims that would be infringed
only as a consequence of further modification of the contributor version. For
purposes of this definition, “control” includes the right to grant patent
sublicenses in a manner consistent with the requirements of this License.

Each contributor grants you a non-exclusive, worldwide, royalty-free patent license
under the contributor's essential patent claims, to make, use, sell, offer for sale,
import and otherwise run, modify and propagate the contents of its contributor
version.

In the following three paragraphs, a “patent license” is any express
agreement or commitment, however denominated, not to enforce a patent (such as an
express permission to practice a patent or covenant not to sue for patent
infringement). To “grant” such a patent license to a party means to make
such an agreement or commitment not to enforce a patent against the party.

If you convey a covered work, knowingly relying on a patent license, and the
Corresponding Source of the work is not available for anyone to copy, free of charge
and under the terms of this License, through a publicly available network server or
other readily accessible means, then you must either **(1)** cause the Corresponding
Source to be so available, or **(2)** arrange to deprive yourself of the benefit of the
patent license for this particular work, or **(3)** arrange, in a manner consistent with
the requirements of this License, to extend the patent license to downstream
recipients. “Knowingly relying” means you have actual knowledge that, but
for the patent license, your conveying the covered work in a country, or your
recipient's use of the covered work in a country, would infringe one or more
identifiable patents in that country that you have reason to believe are valid.

If, pursuant to or in connection with a single transaction or arrangement, you
convey, or propagate by procuring conveyance of, a covered work, and grant a patent
license to some of the parties receiving the covered work authorizing them to use,
propagate, modify or convey a specific copy of the covered work, then the patent
license you grant is automatically extended to all recipients of the covered work and
works based on it.

A patent license is “discriminatory” if it does not include within the
scope of its coverage, prohibits the exercise of, or is conditioned on the
non-exercise of one or more of the rights that are specifically granted under this
License. You may not convey a covered work if you are a party to an arrangement with
a third party that is in the business of distributing software, under which you make
payment to the third party based on the extent of your activity of conveying the
work, and under which the third party grants, to any of the parties who would receive
the covered work from you, a discriminatory patent license **(a)** in connection with
copies of the covered work conveyed by you (or copies made from those copies), or **(b)**
primarily for and in connection with specific products or compilations that contain
the covered work, unless you entered into that arrangement, or that patent license
was granted, prior to 28 March 2007.

Nothing in this License shall be construed as excluding or limiting any implied
license or other defenses to infringement that may otherwise be available to you
under applicable patent law.

### 12. No Surrender of Others' Freedom

If conditions are imposed on you (whether by court order, agreement or otherwise)
that contradict the conditions of this License, they do not excuse you from the
conditions of this License. If you cannot convey a covered work so as to satisfy
simultaneously your obligations under this License and any other pertinent
obligations, then as a consequence you may not convey it at all. For example, if you
agree to terms that obligate you to collect a royalty for further conveying from
those to whom you convey the Program, the only way you could satisfy both those terms
and this License would be to refrain entirely from conveying the Program.

### 13. Use with the GNU Affero General Public License

Notwithstanding any other provision of this License, you have permission to link or
combine any covered work with a work licensed under version 3 of the GNU Affero
General Public License into a single combined work, and to convey the resulting work.
The terms of this License will continue to apply to the part which is the covered
work, but the special requirements of the GNU Affero General Public License, section
13, concerning interaction through a network will apply to the combination as such.

### 14. Revised Versions of this License

The Free Software Foundation may publish revised and/or new versions of the GNU
General Public License from time to time. Such new versions will be similar in spirit
to the present version, but may differ in detail to address new problems or concerns.

Each version is given a distinguishing version number. If the Program specifies that
a certain numbered version of the GNU General Public License “or any later
version” applies to it, you have the option of following the terms and
conditions either of that numbered version or of any later version published by the
Free Software Foundation. If the Program does not specify a version number of the GNU
General Public License, you may choose any version ever published by the Free
Software Foundation.

If the Program specifies that a proxy can decide which future versions of the GNU
General Public License can be used, that proxy's public statement of acceptance of a
version permanently authorizes you to choose that version for the Program.

Later license versions may give you additional or different permissions. However, no
additional obligations are imposed on any author or copyright holder as a result of
your choosing to follow a later version.

### 15. Disclaimer of Warranty

THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW.
EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES
PROVIDE THE PROGRAM “AS IS” WITHOUT WARRANTY OF ANY KIND, EITHER
EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE
QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE
DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

### 16. Limitation of Liability

IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING WILL ANY
COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS THE PROGRAM AS
PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL,
INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE
PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE
OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE
WITH ANY OTHER PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES.

### 17. Interpretation of Sections 15 and 16

If the disclaimer of warranty and limitation of liability provided above cannot be
given local legal effect according to their terms, reviewing courts shall apply local
law that most closely approximates an absolute waiver of all civil liability in
connection with the Program, unless a warranty or assumption of liability accompanies
a copy of the Program in return for a fee.

_END OF TERMS AND CONDITIONS_

## How to Apply These Terms to Your New Programs

If you develop a new program, and you want it to be of the greatest possible use to
the public, the best way to achieve this is to make it free software which everyone
can redistribute and change under these terms.

To do so, attach the following notices to the program. It is safest to attach them
to the start of each source file to most effectively state the exclusion of warranty;
and each file should have at least the “copyright” line and a pointer to
where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

Also add information on how to contact you by electronic and paper mail.

If the program does terminal interaction, make it output a short notice like this
when it starts in an interactive mode:

    DOCUMENTS  Copyright (C) 2021  FISCRATCH
    This program comes with ABSOLUTELY NO WARRANTY; for details type 'show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type 'show c' for details.

The hypothetical commands `show w` and `show c` should show the appropriate parts of
the General Public License. Of course, your program's commands might be different;
for a GUI interface, you would use an “about box”.

You should also get your employer (if you work as a programmer) or school, if any, to
sign a “copyright disclaimer” for the program, if necessary. For more
information on this, and how to apply and follow the GNU GPL, see
&lt;<http://www.gnu.org/licenses/>&gt;.

The GNU General Public License does not permit incorporating your program into
proprietary programs. If your program is a subroutine library, you may consider it
more useful to permit linking proprietary applications with the library. If this is
what you want to do, use the GNU Lesser General Public License instead of this
License. But first, please read
&lt;<http://www.gnu.org/philosophy/why-not-lgpl.html>&gt;.
