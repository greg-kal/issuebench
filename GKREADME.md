*This is a readme file from Greg Kalman regarding adapting IssuBench to probe issues of diveristy and other alignments.*

**Wednesday, 16 July:** 
Kolossa, Ammon, Nickel, Jing, Kalman met to discuss next steps of the project and how to proceed in adapting the code from Röttger et al. 2025 to look at diversity. GK looked at code based, forked repo, and began initial exploration.

**Thursday, 17 July:**
Following the protocols set forth in Röttger et al. 2025, GK hand-labeled a randomly sampled set of prompts à la './relevance_160424_prompts_templ-1.csv' and sent them off to Clara to be annotated a second time

**Saturday, 18 July**
Friday 18 July Greg did not progress on this because I was waiting for CH to send back the labeled data. GK then collapsed the 1000 gold label datapoints into a single .csv file (currently named 'final_GK_Ch_annotations.csv'). Then, with some tweaking, executed 1_create_eval_prompts.ipynb to the folder './eval_prompts'. Now, writing script to load those files to the models and have them complete the analysis, then I will run '2_analyse_responses.ipynb' and get some results.