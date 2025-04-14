**Master Degree in Computer Science** 

**Master Degree in Data Science for Economics and Health**

# Natural Language Processing

#### **Prof. Alfio Ferrara**

**Dott. Sergio Picascia, Dott.ssa Elisabetta Rocchetti**

*Department of Computer Science, Università degli Studi di Milano
Room 7012 via Celoria 18, 20133 Milano, Italia <a href="mailto:alfio.ferrara@unimi.it">alfio.ferrara@unimi.it</a>*



# Ideas for final projects

[TOC]



## Instructions

The final project consists in the preparation of a short study on one of the topics of the course, identifying a precise research question and measurable objectives. The project will propose a methodology for solving the research question and provide an experimental verification of the results obtained according to results evaluation metrics. The emphasis is not on obtaining high performance but rather on the critical discussion of the results obtained in order to understand the potential effectiveness of the proposed methodology.

The results must be documented in a short article of not less than 4 pages and no more than 8, composed according to the guidelines available here: [template](https://preview.springer.com/gp/livingreviews/latex-templates) and using the corresponding $ \ LaTeX $ or MS Word templates. Students have also to provide access to a <a href="https://github.com/">GitHub</a> repository containing the code and reproducible experimental results.

Finally, the project will be discussed after a **10 minutes presentation in English with slides**. 

## Procedure

Exam dates are just for the registration of the final grade. The project discussion will be set by appointment, according to the following procedure:

1. Subscribe to any available date
2. Contact Prof. Ferrara as soon as
   1. The project is finished and ready to be discussed
   2. After the date of your subscription is expired

3. Setup an appointment and discuss your work

**Example**: you subscribe the exam date of [Month] [Day]. **Anytime after [Month] [Day]**, when the **project is ready**, you will contact Prof. Ferrara and set an appointment. You discuss the project during the appointment.

If you are **interested in doing your final master thesis on these topics**, the final project may be a preliminary work in view of the thesis. In this case, discuss the contents with Prof. Ferrara.

## Structure of the paper

1. **Introduction**

   Provides an overview of the project and a short dicsussion on the pertinent literature 

2. **Research question and methodology**

   Provides a clear statement on the goals of the project, an overview of the proposed approach, and a formal definition of the problem

3. **Experimental results**

   Provides an overview of the dataset used for experiments, the metrics used for evaluating performances, and the experimental methodology. Presents experimental results as plots and/or tables

4. **Concluding remarks**

   Provides a critical discussion on the experimental results and some ideas for future work

## AI Usage Disclaimer

Parts of this projects have been developed with the assistance of **OpenAI’s ChatGPT (GPT-4)**. The AI was used to support the **development of project ideas, the structuring of methodological workflows, the drafting of descriptive texts**, and the **identification of relevant datasets and references**. All content produced with AI assistance has been **carefully reviewed, edited, and validated** by me. I take full responsibility for the final content and its accuracy, relevance, and academic integrity.

## Using AI (for students)

Generative AI tools (such as ChatGPT, Claude, Mistral, or similar models) **may be used in this project**, both as an object of investigation and as a tool to support the development process. Students are encouraged to explore how these models function, interact with them creatively, and leverage them as **inspiration or assistance in ideation, drafting, or experimentation**.

However, **AI should not be used as a substitute for original work**. The responsibility for the structure, reasoning, and understanding of the project remains entirely with the student.

If generative AI has been used at any stage of the project, it is **mandatory to include a disclaimer** clearly specifying:

- **Which models** have been used (e.g., GPT-4, Claude 3, etc.)
- **For what purposes** (e.g., drafting text, summarizing ideas, generating code or examples)
- **To what extent** the outputs were modified, verified, or integrated into the final submission

The project will be assessed not only based on its output, but also on the **student’s ability to explain and justify all choices made**. A final **interview will evaluate the depth of understanding**, and any lack of clarity or over-reliance on AI-generated material without proper insight may negatively affect the evaluation.

Generative AI should be seen as a **creativity support tool**, not as a replacement for critical thinking, problem solving, or technical development.

## Project ideas

The following are ideas for projects. For each idea, a short description, example of datasets that can be used, and bibliographic references are provided. Students may **choose one of the following** as their project theme or **they can propose their own idea**, structuring the proposal as those presented in this document. In the latter case, just send the project description to Prof. Ferrara.

### Stop it, it's forbidden! (P1)

Commercial Large Language Models (LLMs), such as ChatGPT, CoPilot, Gemini, have become ubiquitous in various applications, from chatbots to content generation. However, concerns persist regarding their ideological biases, potential censorship, and the need for effective safeguards VS the risk of safeguards as tools for censoring information. This project aims to explore one or more of the following aspects, always by proposing a statistical approach for performing large scale tests, measuring the obeserved evidences and provide set of preliminary results to show up the effectiveness of the proposed approach.

- **Ideological Biases:** Investigate whether commercial LLMs exhibit biases related to political, cultural, or social ideologies. Analyze their responses to prompts on sensitive topics and assess any inherent bias.
- **Safeguards Evaluation:** Examine existing safeguards implemented by commercial LLM providers. Evaluate their effectiveness in preventing harmful or biased outputs. Consider transparency, explainability, and adaptability of these safeguards.
- **Censorship Risks:** Assess the risk of inadvertent or intentional censorship by LLMs. Explore scenarios where information is withheld due to political pressure, corporate interests, or other factors.
- **Comparative Analysis:** Compare different commercial LLMs (such as ChatGPT, CoPilot, Gemini, etc.) in terms of biases, safeguards, and censorship risks. Investigate variations across languages, regions, and user queries.

Censorship and idealogical biases can be studied on one or more of this tasks:

- **Text-to-Text generation:** Observe the effect of different prompt inputs and how they may trigger safeguards or ideologically biased answers in the LLM(s). *Some examples: observe how LLMs paraphrase input texts containing "sensitive" words; ask for information about different books or contents with different levels of "ideological risk" and observe the answer of the LLM.*
- **Text-to-Image generation:** Explore the limits of the type and kind of images that are generated according to different prompts introducing different levels of sensistivity in the request. *Some examples: observe the reaction to different requests to generate images containing naked people, violence, unethical contents and so on.*
- **Image-to-Image transformation:** Explore patch generation in image impainting according to the level of sensitivity of the deleted original patch. *Some examples, observe the inpaining of images containing naked people, violence, unethical contents and so on.*
- **Image-to-Text generation:** observe and analyze the text produced to describe images containing contents with different levels of sensitivity. *Some examples, observe the text generated from images containing naked people, violence, unethical contents and so on.*

#### Dataset

- Any dataset containing sensitive contents as well as non sensitive contents, including automatically generated datasets.

#### References

- Glukhov, D., Shumailov, I., Gal, Y., Papernot, N., & Papyan, V. (2023). Llm censorship: A machine learning challenge or a computer security problem?. *arXiv preprint arXiv:2307.10719*.
- Deng, Y., & Chen, H. (2023). Divide-and-Conquer Attack: Harnessing the Power of LLM to Bypass the Censorship of Text-to-Image Generation Model. *arXiv preprint arXiv:2312.07130*.
- Urman, A., & Makhortykh, M. (2023). The Silence of the LLMs: Cross-Lingual Analysis of Political Bias and False Information Prevalence in ChatGPT, Google Bard, and Bing Chat.
- Zhou, X., Wang, Q., Wang, X., Tang, H., & Liu, X. (2023). Large Language Model Soft Ideologization via AI-Self-Consciousness. *arXiv preprint arXiv:2309.16167*.
- Naous, T., Ryan, M. J., Ritter, A., & Xu, W. (2023). Having beer after prayer? measuring cultural bias in large language models. *arXiv preprint arXiv:2305.14456*.

### Mind the gap (P2)

This project aims to identify, measure, and mitigate social biases, such as gender, race, or profession-related stereotypes, in lightweight transformer models through hands-on fine-tuning and evaluation on targeted NLP tasks. More specifically, the project should implement a four-step methodology, defined as follows:

1. Choose a lightweight pre-trained transformer model (e.g., DistilBERT, ALBERT, RoBERTa-base) suitable for local fine-tuning and evaluation.
2. Evaluate the presence and extent of social bias (e.g., gender, racial, or occupational stereotypes) using dedicated benchmark datasets. Both quantitative metrics and qualitative outputs should be evaluated.
3. Apply a bias mitigation technique, such as **fine-tuning on curated counter-stereotypical data**, integrating **adapter layers**, or employing **contrastive learning**, while keeping the solution computationally efficient and transparent.
4. Re-assess the model using the same benchmark(s) to measure improvements. Students should compare pre- and post-intervention results, discuss trade-offs (e.g., performance vs. fairness), and visualize the impact of their approach.

#### Dataset

- [StereoSet: Measuring stereotypical bias in pretrained language models](https://github.com/moinnadeem/StereoSet). Nadeem, M., Bethke, A., & Reddy, S. (2020). StereoSet: Measuring stereotypical bias in pretrained language models. *arXiv preprint arXiv:2004.09456*.

#### References

- Zhang, Y., & Zhou, F. (2024). Bias mitigation in fine-tuning pre-trained models for enhanced fairness and efficiency. *arXiv preprint arXiv:2403.00625*.
- Fu, C. L., Chen, Z. C., Lee, Y. R., & Lee, H. Y. (2022). Adapterbias: Parameter-efficient token-dependent representation shift for adapters in nlp tasks. *arXiv preprint arXiv:2205.00305*.
- Park, K., Oh, S., Kim, D., & Kim, J. (2024, June). Contrastive Learning as a Polarizer: Mitigating Gender Bias by Fair and Biased sentences. In *Findings of the Association for Computational Linguistics: NAACL 2024* (pp. 4725-4736).

### Politics of emotions or propaganda? (P3)

This project explores how emotional language is used strategically in political texts, such as speeches, social media posts, or debates—to influence perception and manipulate audience response. Students will design a pipeline using transformer-based models to detect emotional framing, categorize tone (e.g., fear, pride, outrage), and highlight shifts in sentiment across political stances or media sources. The goal is not just to classify emotion, but to *interpret its rhetorical function* within the discourse. In order to perform the task, the project should:

1. Use pre-trained transformer models (e.g. RoBERTa, BERT fine-tuned on GoEmotions) to classify emotions expressed in each text.
2. Examine how specific emotions (e.g. fear, anger, pride) are used across parties, time periods, or topics to shape opinion.
3. Create plots or dashboards comparing emotional tone across actors, media types, or ideological groups.
4. Apply explainability methods (e.g. SHAP, attention heatmaps) to highlight emotional trigger words and rhetorical patterns.

#### Dataset

- [GoEmotions: A Dataset of Fine-Grained Emotions](https://github.com/google-research/google-research/tree/master/goemotions). Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A dataset of fine-grained emotions. *arXiv preprint arXiv:2005.00547*.
- [Media Frames Corpus](https://github.com/dallascard/media_frames_corpus). Card, D., Boydstun, A., Gross, J. H., Resnik, P., & Smith, N. A. (2015, July). The media frames corpus: Annotations of frames across issues. In *Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)* (pp. 438-444).

#### References

- Şeref, M. M., Şeref, O., Abrahams, A. S., Hill, S. B., & Warnick, Q. (2023). Rhetoric Mining: A New Text-Analytics Approach for Quantifying Persuasion. *INFORMS Journal on Data Science*, *2*(1), 24-44.
- Chefer, H., Gur, S., & Wolf, L. (2021). Transformer interpretability beyond attention visualization. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition* (pp. 782-791).
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in neural information processing systems*, *30*.

### To Bot or not to Bot? (P4)

This project investigates the detection of AI-generated fake personas on social media by analyzing the coherence and authenticity of their multimodal footprint, from profile descriptions and post content to image consistency and interaction patterns. The goal is to build a system that can flag suspicious accounts through language and/or vision-based features.

1. Collect real social profiles through available datasets and mix them with fake profiles. Those can be provided by the dataset itself or be generated through LLMs for the sake of the project.
2. Use transformer models (e.g., RoBERTa, BERT) to analyze linguistic cues like unnatural phrasing, semantic drift across posts, or repetition patterns.
3. (Optional) Use CLIP or other visual transformers to check semantic alignment between profile images, bios, and posted content (e.g., does the image match the described profession or lifestyle?).
4. Combine signals from text and image to train a classifier for detecting suspicious or AI-generated personas.

#### Dataset

- [TwitterGAN dataset: Fake Twitter accounts using GAN-generated faces as profiles](https://zenodo.org/records/10436889). 
- [TweepFake: about Detecting Deepfake Tweets](https://github.com/tizfa/tweepfake_deepfake_text_detection). Fagni, T., Falchi, F., Gambini, M., Martella, A., & Tesconi, M. (2021). TweepFake: About detecting deepfake tweets. *Plos one*, *16*(5), e0251415.
- [TwiBot-22](https://github.com/LuoUndergradXJTU/TwiBot-22). TwiBot-22 is the largest and most comprehensive Twitter bot detection benchmark to date.

#### References

- Habib, A. R. R., Akpan, E. E., Ghosh, B., & Dutta, I. K. (2024, January). Techniques to detect fake profiles on social media using the new age algorithms-A Survey. In *2024 IEEE 14th Annual Computing and Communication Workshop and Conference (CCWC)* (pp. 0329-0335). IEEE.
- Boato, G., Pasquini, C., Stefani, A. L., Verde, S., & Miorandi, D. (2022, October). TrueFace: A dataset for the detection of synthetic face images from social networks. In *2022 IEEE International Joint Conference on Biometrics (IJCB)* (pp. 1-7). IEEE.
- Khaled, S., El-Tazi, N., & Mokhtar, H. M. (2018, December). Detecting fake accounts on social media. In *2018 IEEE international conference on big data (big data)* (pp. 3672-3681). IEEE.

### If You Ask Nicely... (P5)

This project aims at exploring and comparing different prompt engineering strategies (e.g., explicit instructions, few-shot examples, role prompts, zero-shot) by applying them to a creative task across multiple LLMs (e.g., GPT-4, Claude, Mistral, LLaMA). The core objective is to analyze how different parts of the prompt influence the model’s output, using explainability tools to identify which components are actually being attended to or utilized by the model during generation. The project shoul be based on creative generation tasks, in order to focus on the capability of models to handle non trivial generation tasks. Possible examples could be: 1) asking a historical figure to comment about some modern trend or issue; 2) asking the model to explain some scientific concept in a wrong but plausible way; 3) asking the model to generate fictional stories or characters with different styles; 4) asking the model to generate prompts for other models; 5) asking the model to act as a counselor for fictional characters (e.g., the Joker, Darth Vater, Barry Lyndon, ...). 

The methodological steps are:

1. Design multiple prompts (at least 4–5 per group), varying in structure, tone, and technique, all targeting the same generation task.
2. Prompts are submitted to 2–3 different LLMs (both open and closed if possible) to observe differences in how each model interprets and responds to the same inputs.
3. Generated texts are collected and annotated based on features like coherence, creativity, adherence to prompt instructions, style, and hidden elements (if applicable).
4. Enforce explainability methods (such as [Logprobs](https://cookbook.openai.com/examples/using_logprobs) or [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) just to give a couple of examples) in order to analyze which sections of the prompt influenced the generation most, and which were ignored or misunderstood.
5. Provide a critical comparison of different prompting techniques.

#### Dataset

The dataset generation is one of the goals of the project

#### References

- Schulhoff, S., Ilie, M., Balepur, N., Kahadze, K., Liu, A., Si, C., ... & Resnik, P. (2024). The prompt report: A systematic survey of prompting techniques. *arXiv preprint arXiv:2406.06608*, *5*.
- Li, C., Wang, J., Zhang, Y., Zhu, K., Hou, W., Lian, J., ... & Xie, X. (2023). Large language models understand and can be enhanced by emotional stimuli. *arXiv preprint arXiv:2307.11760*.
- Zheng, M., Pei, J., Logeswaran, L., Lee, M., & Jurgens, D. (2024, November). When” A Helpful Assistant” Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of Large Language Models. In *Findings of the Association for Computational Linguistics: EMNLP 2024* (pp. 15126-15154).

### Boardgames are made of rules (P6)

The goal of the project is to challenge LLMs to understand the logic of real boardgames, generate playable rulesets, and even invent new games, using prompt engineering, rule extraction, and reasoning. Students will test the model’s ability to grasp and manipulate structured rules, explain them clearly, and generate novel (but coherent) mechanics. There are several tasks that can be used to test this LLM ability. The project may try to explore all, just some of them, or focus specifically on one of these tasks. Here, we provide some examples:

- Give the model a rulebook and prompt it to explain the game in simple, conversational terms to a child or other audiences.
- Feed the model a version of a game with missing or intentionally flawed rules and ask it to identify the issue and suggest a correction.
- Ask the model to invent a boardgame based on a concept or theme.
- Feed the model the rules of a game and a current game state (in natural language), and ask it to narrate or recommend the next move.
- Give the model a rulebook and ask it to classify the mechanics, evaluate the complexity, suggests the perfect number of players and estimate the duration. These model predictions can be compared with the data provided by BoardGameGeek ([BGG](https://boardgamegeek.com/)).

#### Dataset

-  BGG API ([https://boardgamegeek.com/wiki/page/BGG_XML_API2](https://boardgamegeek.com/wiki/page/BGG_XML_API2)).
- Any collection of rulebooks that can be downloaded from the publishers websites.

#### References

- Hu, C., Zhao, Y., & Liu, J. (2024, August). Game generation via large language models. In *2024 IEEE Conference on Games (CoG)* (pp. 1-4). IEEE.
- Todd, G., Padula, A. G., Stephenson, M., Piette, É., Soemers, D., & Togelius, J. (2024). GAVEL: Generating games via evolution and language models. *Advances in Neural Information Processing Systems*, *37*, 110723-110745.
- Li, D., Zhang, S., Sohn, S. S., Hu, K., Usman, M., & Kapadia, M. (2025). Cardiverse: Harnessing LLMs for Novel Card Game Prototyping. *arXiv preprint arXiv:2502.07128*.

### Elementary, my Dear Watson! (P7)

This project explores the reasoning capabilities of large language models (LLMs) by challenging them with custom-designed logical problems of varying complexity. Students will formulate these problems in natural language and compare LLM responses with the output of formal logic solvers such as [Prolog](https://www.swi-prolog.org/) or [Z3](https://github.com/Z3Prover/z3). The goal is to investigate to what extent LLMs can simulate symbolic reasoning, what kinds of logical structures they struggle with, and how prompt formulation affects their performance. Through explainability tools, students will analyze which parts of the prompt guide the model’s reasoning and whether the model’s answers reflect genuine understanding or shallow pattern matching. Methodological steps:

1. Create custom logic puzzles in natural language, varying complexity and structure to test different types of reasoning.
2. Each problem is formalized and solved using solvers to provide a ground-truth reference.
3. The same problems are posed to LLMs using different prompting strategies to test sensitivity and variation in model behavior.
4. Responses are evaluated against the symbolic solutions to assess correctness, consistency, and patterns of failure.
5. Tools like attention maps or token attribution are used to identify which parts of the prompt influenced the model’s response.

#### Dataset

- [LogicQA](https://huggingface.co/datasets/lucasmccabe/logiqa): LogiQA is constructed from the logical comprehension problems from publically available questions of the National Civil Servants Examination of China, which are designed to test the civil servant candidates’ critical thinking and problem solving.
- [Project Euler](https://www.kaggle.com/datasets/dheerajmpai/projecteuler): this dataset contains a comprehensive collection of questions and solutions from Project Euler, spanning the period from 2007 to July 4th, 2023. Project Euler is a popular online platform that offers a wide range of challenging mathematical and computational problems, designed to encourage problem-solving and algorithmic thinking.
- Tafjord, O., Mishra, B. D., & Clark, P. (2020). ProofWriter: Generating implications, proofs, and abductive statements over natural language. *arXiv preprint arXiv:2012.13048*.

#### References

- Tang, X., Zheng, Z., Li, J., Meng, F., Zhu, S. C., Liang, Y., & Zhang, M. (2023). Large language models are in-context semantic reasoners rather than symbolic reasoners. *arXiv preprint arXiv:2305.14825*.
- Fang, M., Deng, S., Zhang, Y., Shi, Z., Chen, L., Pechenizkiy, M., & Wang, J. (2024, March). Large language models are neurosymbolic reasoners. In *Proceedings of the AAAI conference on artificial intelligence* (Vol. 38, No. 16, pp. 17985-17993).
- Sullivan, R., & Elsayed, N. (2024). Can Large Language Models Act as Symbolic Reasoners?. *arXiv preprint arXiv:2410.21490*.

### The Boundary of Meaning (P8)

This project investigates how figurative and literal language are encoded in the embedding space of transformer models such as BERT or RoBERTa. The goal is to determine whether a semantic boundary can be identified between figurative and non-figurative expressions, and to analyze how models internally represent idioms, metaphors, and literal phrases. Students will use clustering, dimensionality reduction, and linear probing techniques to explore separability in the latent space and evaluate how well figurative meaning aligns with surface form. Steps:

1. Students collect or use an existing dataset of sentence pairs or short phrases labeled as figurative or literal (e.g., idioms vs literal uses, metaphor detection datasets).
2. Use contextualized embeddings from BERT or similar models (e.g., CLS token, average token embedding, or layer-wise representation) to generate vectors for each expression.
3. Apply dimensionality reduction (e.g., PCA, t-SNE, UMAP) to visualize distribution of figurative vs literal examples. Use clustering algorithms or a classifier to see if the two categories form separable groups.
4. Use a simple classifier to test whether a boundary exists that separates figurative from literal expressions in the vector space.

#### Dataset

- [VU Amsterdam Metaphor Corpus](http://www.vismet.org/metcor/documentation/home.html). Steen, G.J., Dorst A.G., Herrmann, J.B., Kaal, A.A., Krennmayr, T., Pasma, T. (2010). A method for linguistic metaphor identification. From MIP to MIPVU. Amsterdam: John Benjamins.
- [FigLang 2024](https://aclanthology.org/2024.figlang-1.16/): Kulkarni, S., Saakyan, A., Chakrabarty, T., & Muresan, S. (2024, June). A report on the FigLang 2024 shared task on multimodal figurative language. In *Proceedings of the 4th Workshop on Figurative Language Processing (FigLang 2024)* (pp. 115-119).

#### References

- Choi, M., Lee, S., Choi, E., Park, H., Lee, J., Lee, D., & Lee, J. (2021). MelBERT: Metaphor detection via contextualized late interaction using metaphorical identification theories. *arXiv preprint arXiv:2104.13615*.
- Lin, Y., Liu, J., Gao, Y., Wang, A., & Su, J. (2025, April). A Dual-Perspective Metaphor Detection Framework Using Large Language Models. In *ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* (pp. 1-5). IEEE.
- Yang, C., Li, Z., Liu, Z., & Huang, Q. (2023). Deep Learning-Based Knowledge Injection for Metaphor Detection: A Comprehensive Review. *arXiv preprint arXiv:2308.04306*.

### Once Upon a Time (P9)

This project investigates how large language models can be used to generate interactive narratives with strong internal consistency and responsiveness to user input. While role-playing game (RPG)-style adventures are one possible format, students are free to experiment with other narrative genres, such as mystery, science fiction, or surreal fiction, so long as the focus remains on maintaining narrative coherence, continuity, and structure over multiple interactions. The goal is to explore how prompt design, memory management, and semantic control can influence a model’s ability to generate believable, engaging, and logically grounded stories. A specific goal is to distinguish between explicit and implicit narrative rules, as well as recognize violations of story consistency. An explicit rule is one directly stated in the prompt or setup, for example, if the protagonist is allergic to sunlight, it would be inconsistent for the model to later describe them enjoying a sunny morning. Implicit rules, on the other hand, are based on context or world-building assumptions. For instance, if a character is a detective in 1920s London, the model should not include smartphones or modern slang, even if these rules aren’t spelled out. Consistency violations can also arise over time, for example, if a magic ring is destroyed in one scene but used again in a later turn, the narrative breaks its own logic. These types of inconsistencies are central to the project’s investigation into model memory and control. 

Methodological steps:

1. Select a narrative genre or structure (e.g., RPG-style adventure, detective story, sci-fi simulation) to explore through generation.
2. Different prompt formats are developed to guide the model’s storytelling behavior, including opening setups, scene descriptions, and user-model interactions.
3. A strategy for preserving narrative memory across turns is defined, using techniques like prompt concatenation, external memory buffers, or structured notes.
4. Students run multi-turn story sessions with the model, simulating user interaction and tracking how well the model maintains coherence and adapts to new inputs. User interaction can also be simulated by a second LLM but this way we need to check consistency in both the models.
5. Stories are evaluated for narrative continuity, logical flow, character consistency, and ability to follow implicit or explicit rules.
6. Reflect on and evaluate how different prompt structures, memory strategies, or model types affect the level of creative and structural control they can achieve.

#### Dataset

- [Story Cloze Test and ROCStories Corpora](https://cs.rochester.edu/nlp/rocstories/): 'Story Cloze Test' is a new commonsense reasoning framework for evaluating story understanding, story generation, and script learning. This test requires a system to choose the correct ending to a four-sentence story.
- [PersonaChat](https://huggingface.co/datasets/AlekseyKorshuk/persona-chat): designed for consistency in dialogues can be adapted to storytelling
- Wu, Y., Mei, J., Yan, M., Li, C., Lai, S., Ren, Y., ... & Huang, F. (2025). WritingBench: A Comprehensive Benchmark for Generative Writing. *arXiv preprint [arXiv:2503.05244](https://arxiv.org/abs/2503.05244)*.

#### References

- Mostafazadeh, N., Chambers, N., He, X., Parikh, D., Batra, D., Vanderwende, L., ... & Allen, J. (2016). A corpus and evaluation framework for deeper understanding of commonsense stories. *arXiv preprint arXiv:1604.01696*.
- Mostafazadeh, N., Grealish, A., Chambers, N., Allen, J., & Vanderwende, L. (2016, June). CaTeRS: Causal and temporal relation scheme for semantic annotation of event structures. In *Proceedings of the Fourth Workshop on Events* (pp. 51-61).
- Mostafazadeh, N., Vanderwende, L., Yih, W. T., Kohli, P., & Allen, J. (2016, August). Story cloze evaluator: Vector space representation evaluation by predicting what happens next. In *Proceedings of the 1st Workshop on Evaluating Vector-Space Representations for NLP* (pp. 24-29).

### CTRL + Style (P10)

This project investigates how large language models encode literary style and whether distinct stylistic voices can be separated or blended in vector space. Students will explore style transfer in text generation, focusing on how models imitate, transform, or interpolate between literary styles (e.g., Hemingway vs. Austen, noir vs. satire). The primary focus is on text, but students may optionally explore neural style transfer in images using visual or multimodal models. The ultimate goal is to identify how style is represented internally, whether as distinct clusters, continuous gradients, or manipulable embeddings, and to test whether stylistic transformations are consistent, controllable, and interpretable. Methods:

1. Choose two or more distinct literary styles or authorial voices to compare and work with, for example, classic authors, genres, or tonal extremes (e.g., gothic vs minimalist).
2. A corpus of texts or passages is gathered for each style, either from public datasets or self-curated sources, to serve as input for embedding analysis and generation.
3. Texts are embedded using a transformer model (e.g., BERT, RoBERTa, or SentenceTransformers) to explore whether stylistic differences are reflected in the vector space.
4. Dimensionality reduction (e.g., PCA, t-SNE, UMAP) and clustering methods are used to test whether different styles form separable regions in the embedding space.
5. Using prompt-based control or editing techniques, rewrite a given text in a different style, analyzing changes in tone, syntax, and structure.
6. Outputs are evaluated for consistency, fidelity to target style, and content preservation. Reflect on whether "style" is a learnable or transferable concept for LLMs, and how it is internally encoded.

#### Dataset

- [Comprehensive Literary Greats Dataset](https://www.kaggle.com/datasets/thedevastator/comprehensive-literary-greats-dataset). 50,000+ Books Rated and Awarded Across Language, Genre, and Format.
- [BOOKSUM: A Collection of Datasets for Long-form Narrative Summarization](https://huggingface.co/datasets/kmfoda/booksum): The majority of available text summarization datasets include short-form source documents that lack long-range causal and temporal dependencies, and often contain strong layout and stylistic biases.
- [The CONLIT Dataset of Contemporary Literature](https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.88). This dataset includes derived data on a collection of ca. 2,700 books in English published between 2001–2021 and spanning 12 different genres.
- [Wikiart dataset](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset). Tan, W. R., Chan, C. S., Aguirre, H. E., & Tanaka, K. (2018). Improved ArtGAN for conditional synthesis of natural image and artwork. IEEE Transactions on Image Processing, 28(1), 394-409.

#### References

- Jing, Y., Yang, Y., Feng, Z., Ye, J., Yu, Y., & Song, M. (2019). Neural style transfer: A review. *IEEE transactions on visualization and computer graphics*, *26*(11), 3365-3385.
- Cai, Q., Ma, M., Wang, C., & Li, H. (2023). Image neural style transfer: A review. *Computers and Electrical Engineering*, *108*, 108723.
- Pan, L., Lan, Y., Li, Y., & Qian, W. (2024). Unsupervised Text Style Transfer via LLMs and Attention Masking with Multi-way Interactions. *arXiv preprint arXiv:2402.13647*.
