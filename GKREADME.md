*This is a readme file from Greg Kalman regarding adapting IssuBench to probe issues of diveristy and other alignments.*

**Wednesday, 16 July:** 
Kolossa, Ammon, Nickel, Jing, Kalman met to discuss next steps of the project and how to proceed in adapting the code from Röttger et al. 2025 to look at diversity. GK looked at code based, forked repo, and began initial exploration.

**Thursday, 17 July:**
Following the protocols set forth in Röttger et al. 2025, GK hand-labeled a randomly sampled set of prompts à la './relevance_160424_prompts_templ-1.csv' and sent them off to Clara to be annotated a second time

**Saturday, 19 July**
Friday 18 July Greg did not progress on this because I was waiting for CH to send back the labeled data. GK then collapsed the 1000 gold label datapoints into a single .csv file (currently named 'final_GK_CH_annotations.csv'). Then, with some tweaking, executed 1_create_eval_prompts.ipynb to the folder './eval_prompts'. Now, writing script to load those files to the models and have them complete the analysis, then I will run '2_analyse_responses.ipynb' and get some results. Script is in really shit time complexity but it is running nonetheless. Going to let it run overnight and I'll see what happens tomorrow. 

**Sunday, 20 July**
Completed in 170m. Running now with a different model, llama3:8b. Ran and compiled in '2_relevance_filtering/eval_prompts_responsesKalmanLlama.' Ran '2_analyse_responses'. Moving onto '3_writing_assistance_filtering'.

**Monday, 21 July**
Met w SA, DK, CH about preliminary findings. Decided to pivot away from small llama models and toward bigger models like DeepSeek. Also need to remove human-annotations from the dataset and feed that new data, with just the prompt and without the human annotations, through the bigger models. 