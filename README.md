# Guide

## Contents
1. user study: User study conducted on AMTurk

## Files
### User Study 

Note: The necessary files are "objective_data/", "subjective_data/", and data_analysis.py

Figures: Figures plotted by running data_analysis.py

Objective data:
  - SET-A_CONTEXT = True, then for first set of 44 questions the agent picked player using FairCB and next set of 44 questions the agent picked player using non-contextual fair FTRL.
  - SET-B_CONTEXT = True, then for first set of 44 questions the agent picked player using non-contextual fair FTRL and next set of 44 questions the agent picked player using FairCB.
  - Set: Set number in ./questions.json
  - lossVal: 0 = correct response, 1 = incorrect response
  - time: time taken in secs to select response (max. time is 10 secs)
  - loss-c1p1: Importance weighted loss estimator (l_hat) for context 1 - player 1.
  - actual_prob-c1p2: Probability of selecting player 2 in context 1 (as computed by the algorithm)
  - sch_prob-c2p1: Probability of selecting player 1 in context 2 rounded off to the tenth place for the next 10 turns to be divided amongst the players.
  - Turns: question assignment for the next 10 questions. 0 - player 1 (USA) and 1 - player 2 (India)

Subjective data:
  - Ratings are on 1-7 Likert Scale
  - For half the trials FairCB was run in Part 1 (SET A) of the study and for the other half FairCB was run in Part 2 (SET A) for counter-balancing.

data_analysis.py: 
  - This file loads the objective and subjective data --> removes users that did not complete the quiz or did not answer any survey responses --> calculates and plots performance and fairness rating.

questions.json:
   ```
    question_img: link for the image
    question_no: question number from for that set
    set: there are 2 sets set-0 and set-1
    category: (0: India) and (1: USA)
    a: option
    b: option
    c: option
    d: option
    answer: which of the option is correct
        
  ```
